"""
Canli iki model karsilastirma goruntuleme.

Ayni frame'de iki model paralel calisir, ekran ikiye bolunur:
  SOL  → Model A
  SAG  → Model B

Kullanim ornekleri:
  # Helmet: crop_30 vs normal_30
  python scripts/live_compare.py --left crophelmet_30 --right normal_helmet_30

  # Helmet: crophelmet_final vs vinayak
  python scripts/live_compare.py --left crophelmet_final --right vinayak

  # Vest: cropvest_30 vs normal_vest_30
  python scripts/live_compare.py --left cropvest_30 --right normal_vest_30

  # Belirli video:
  python scripts/live_compare.py --left crophelmet_final --right vinayak --source C:/video.mp4

  # Kaydet:
  python scripts/live_compare.py --left crophelmet_final --right vinayak --save

Tuslari:
  q  : cik
  s  : ekran goruntusunu kaydet
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

import cv2
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model kayit defteri
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "crophelmet_30": {
        "path": "models/crophelmet_agent_30_best.pt",
        "classes": ["Hardhat", "NO-Hardhat"],
        "label": "CropHelmet-30",
        "ppe_type": "helmet",
    },
    "crophelmet_final": {
        "path": "models/crophelmet_agent_final_best.pt",
        "classes": ["Hardhat", "NO-Hardhat"],
        "label": "CropHelmet-Final",
        "ppe_type": "helmet",
    },
    "normal_helmet_30": {
        "path": "models/helmet_agent_30_best.pt",
        "classes": ["Hardhat", "NO-Hardhat"],
        "label": "NormalHelmet-30",
        "ppe_type": "helmet",
    },
    "cropvest_30": {
        "path": "models/cropvest_agent_30_best.pt",
        "classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "CropVest-30",
        "ppe_type": "vest",
    },
    "normal_vest_30": {
        "path": "models/vest_agent_30_best.pt",
        "classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "NormalVest-30",
        "ppe_type": "vest",
    },
    "vinayak_vest": {
        "path": "models/pretrained/vinayakmane/ppe.pt",
        "classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "vinayak_vest",
        "ppe_type": "vest",
    },
    "vinayak": {
        "path": "models/pretrained/vinayakmane/ppe.pt",
        "classes": ["Hardhat", "NO-Hardhat"],
        "label": "Vinayak-PPE",
        "ppe_type": "helmet",
    },
    "bera_vest_best": {
        "path": "models/vest_agent_final_best.pt",
        "classes": ["Safety Vest", "NO-Safety Vest"],
        "label": "beraVest-best",
        "ppe_type": "vest",
    },
    "helmet_agent_final": {
        "path": "models/helmet_agent_final_best.pt",
        "classes": ["Hardhat", "NO-Hardhat"],
        "label": "HelmetAgent-Final",
        "ppe_type": "helmet",
    },
}

PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"
PERSON_CONF = 0.25
PPE_CONF = 0.30
TRACKER = "bytetrack.yaml"
IMGSZ = 640
TEMPORAL_WINDOW = 10
CROP_PADDING = 0.30

# Renk paleti (BGR)
COLOR_POSITIVE = (0, 200, 0)    # yesil  - kask/yelek var
COLOR_NEGATIVE = (0, 0, 220)    # kirmizi - kask/yelek yok
COLOR_UNKNOWN  = (0, 200, 255)  # sari   - bilinmiyor

POSITIVE_LABELS = {"Hardhat", "Safety Vest"}
NEGATIVE_LABELS = {"NO-Hardhat", "NO-Safety Vest"}


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root() / p).resolve()


def choose_source() -> Path:
    desktop = Path.home() / "Desktop"
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Karsilastirilacak kaynak (video veya gorsel)",
        initialdir=str(desktop if desktop.exists() else Path.home()),
        filetypes=[
            ("Video ve gorseller", "*.mp4 *.avi *.mov *.mkv *.webm *.jpg *.jpeg *.png *.bmp"),
            ("Videolar", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Gorseller", "*.jpg *.jpeg *.png *.bmp"),
            ("Tum dosyalar", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise SystemExit("Kaynak secilmedi.")
    return Path(selected)


def select_device(requested: str) -> str:
    if requested.lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    raise SystemExit("GPU secildi ama CUDA aktif degil. --device cpu kullan.")


def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [name_to_id[n] for n in names]


def crop_with_padding(frame, box, pad_ratio: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    pw = int(max(1, x2 - x1) * pad_ratio)
    ph = int(max(1, y2 - y1) * pad_ratio)
    return frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]


def best_detection(model: YOLO, result, allowed_ids: list[int]) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for box in result.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < PPE_CONF:
            continue
        label = str(model.names[cid])
        if best is None or conf > best[1]:
            best = (label, conf)
    return best if best else ("unknown", 0.0)


def vote(q: deque) -> str:
    if not q:
        return "unknown"
    return Counter(q).most_common(1)[0][0]


def label_color(label: str) -> tuple[int, int, int]:
    if label in POSITIVE_LABELS:
        return COLOR_POSITIVE
    if label in NEGATIVE_LABELS:
        return COLOR_NEGATIVE
    return COLOR_UNKNOWN


def readable_fg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    b = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]
    return (0, 0, 0) if b > 140 else (255, 255, 255)


def draw_box_label(frame, x1, y1, x2, y2, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    scale, thick, pad = 1.4, 3, 10
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bx1 = max(0, x1)
    by2 = max(th + bl + pad * 2, y1)
    by1 = by2 - th - bl - pad * 2
    bx2 = min(frame.shape[1] - 1, bx1 + tw + pad * 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad - bl),
                cv2.FONT_HERSHEY_SIMPLEX, scale, readable_fg(color), thick, cv2.LINE_AA)


def draw_header(frame, model_label: str, fps: float, track_count: int, side: str) -> None:
    h, w = frame.shape[:2]
    bg = (30, 30, 30)
    bar_h = 36
    cv2.rectangle(frame, (0, 0), (w, bar_h), bg, -1)

    fps_txt = f"FPS:{fps:.1f}  Tracks:{track_count}"
    side_tag = f"[{'LEFT' if side == 'left' else 'RIGHT'}]"
    text = f"{side_tag} {model_label}   {fps_txt}"
    cv2.putText(frame, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_stats_panel(frame, known: int, unknown: int, pos: int, neg: int) -> None:
    lines = [
        f"Known  : {known}",
        f"Unknown: {unknown}",
        f"Pos    : {pos}",
        f"Neg    : {neg}",
    ]
    scale, thick, pad, lh = 0.55, 1, 8, 22
    max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines)
    pw = max_w + pad * 2
    ph = lh * len(lines) + pad * 2
    x0, y0 = 8, frame.shape[0] - ph - 8
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (180, 180, 180), 1)
    for i, line in enumerate(lines):
        cy = y0 + pad + 16 + i * lh
        cv2.putText(frame, line, (x0 + pad, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (220, 220, 220), thick, cv2.LINE_AA)


def resize_half(frame, max_w: int, max_h: int):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


# ---------------------------------------------------------------------------
# Ana isleme
# ---------------------------------------------------------------------------

def process_frame(
    frame,
    person_model: YOLO,
    person_ids: list[int],
    model_a: YOLO,
    ids_a: list[int],
    states_a,
    model_b: YOLO,
    ids_b: list[int],
    states_b,
    device: str,
    padding: float = CROP_PADDING,
) -> tuple[Any, Any, int]:
    """
    Ayni frame'i iki kopyaya ayir, her birine farkli modelin
    sonuclarini ciz. Ortak person tracking bir kez calisir.
    """
    frame_a = frame.copy()
    frame_b = frame.copy()

    person_result = person_model.track(
        frame,
        classes=person_ids,
        tracker=TRACKER,
        persist=True,
        imgsz=IMGSZ,
        conf=PERSON_CONF,
        device=device,
        verbose=False,
    )[0]

    boxes = person_result.boxes
    if boxes is None or boxes.id is None:
        return frame_a, frame_b, 0

    track_count = len(boxes.id)

    for box, tid in zip(boxes.xyxy, boxes.id):
        track_id = int(tid)
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = crop_with_padding(frame, [x1, y1, x2, y2], padding)

        # --- Model A ---
        res_a = model_a.predict(crop, classes=ids_a, imgsz=IMGSZ, conf=PPE_CONF, device=device, verbose=False)[0]
        label_a, conf_a = best_detection(model_a, res_a, ids_a)
        states_a[track_id].append(label_a)
        voted_a = vote(states_a[track_id])
        color_a = label_color(voted_a)
        draw_box_label(frame_a, x1, y1, x2, y2, f"#{track_id} {voted_a}", color_a)

        # --- Model B ---
        res_b = model_b.predict(crop, classes=ids_b, imgsz=IMGSZ, conf=PPE_CONF, device=device, verbose=False)[0]
        label_b, conf_b = best_detection(model_b, res_b, ids_b)
        states_b[track_id].append(label_b)
        voted_b = vote(states_b[track_id])
        color_b = label_color(voted_b)
        draw_box_label(frame_b, x1, y1, x2, y2, f"#{track_id} {voted_b}", color_b)

    return frame_a, frame_b, track_count


def make_side_by_side(frame_a, frame_b, max_w: int = 1280, max_h: int = 720) -> Any:
    import numpy as np
    half_w = max_w // 2
    fa = resize_half(frame_a, half_w, max_h)
    fb = resize_half(frame_b, half_w, max_h)

    # Yukseklikleri esitle
    ha, wa = fa.shape[:2]
    hb, wb = fb.shape[:2]
    target_h = max(ha, hb)
    if ha < target_h:
        fa = cv2.copyMakeBorder(fa, 0, target_h - ha, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if hb < target_h:
        fb = cv2.copyMakeBorder(fb, 0, target_h - hb, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    divider = np.full((target_h, 3, 3), 100, dtype="uint8")
    canvas = np.hstack([fa, divider, fb])

    # Toplam canvas ekrana sigmazsa tekrar kucult
    ch, cw = canvas.shape[:2]
    scale = min(max_w / cw, max_h / ch, 1.0)
    if scale < 1.0:
        canvas = cv2.resize(canvas, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    return canvas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    choices = list(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Canli iki model karsilastirma.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--left",  choices=choices, default="crophelmet_final",
                        help=f"Sol panel modeli. Secenekler: {choices}")
    parser.add_argument("--right", choices=choices, default="normal_helmet_30",
                        help=f"Sag panel modeli. Secenekler: {choices}")
    parser.add_argument("--source", help="Video/gorsel yolu veya webcam index. Bos birakinca dosya secici acilir.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--save", action="store_true", help="Sonucu video olarak kaydet.")
    parser.add_argument("--max-frames", type=int, help="Maksimum frame sayisi (video kaydi icin).")
    parser.add_argument("--display-width", type=int, default=1280,
                        help="Toplam pencere genisligi piksel (varsayilan: 1280). Kucuk ekran icin 1024 veya 960 dene.")
    parser.add_argument("--display-height", type=int, default=720,
                        help="Toplam pencere yuksekligi piksel (varsayilan: 720).")
    parser.add_argument("--padding", type=float, default=CROP_PADDING,
                        help=f"Crop padding orani (varsayilan: {CROP_PADDING}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source is None:
        source = choose_source()
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = resolve_path(args.source)

    device = select_device(args.device)

    info_a = MODEL_REGISTRY[args.left]
    info_b = MODEL_REGISTRY[args.right]

    # Modelleri yukle
    person_path = resolve_path(PERSON_MODEL_PATH)
    if not person_path.exists():
        raise SystemExit(f"Person model bulunamadi: {person_path}")

    path_a = resolve_path(info_a["path"])
    path_b = resolve_path(info_b["path"])
    for p, name in [(path_a, args.left), (path_b, args.right)]:
        if not p.exists():
            raise SystemExit(f"Model bulunamadi [{name}]: {p}")

    print(f"Person model : {person_path.name}")
    print(f"Sol  [{args.left}] : {path_a.name}  classes={info_a['classes']}")
    print(f"Sag  [{args.right}] : {path_b.name}  classes={info_b['classes']}")
    print(f"Device: {device}")

    person_model = YOLO(str(person_path))
    model_a = YOLO(str(path_a))
    model_b = YOLO(str(path_b))

    person_ids = class_ids(person_model, ["person"])
    ids_a = class_ids(model_a, info_a["classes"])
    ids_b = class_ids(model_b, info_b["classes"])

    states_a = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))
    states_b = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_image = isinstance(source, Path) and source.suffix.lower() in IMAGE_EXT

    cap = None
    writer = None
    window = f"LEFT: {info_a['label']}   |   RIGHT: {info_b['label']}   (q=cik, s=screenshot)"

    if is_image:
        frame_raw = cv2.imread(str(source))
        if frame_raw is None:
            raise SystemExit(f"Gorsel acilamadi: {source}")
        fa, fb, tc = process_frame(
            frame_raw, person_model, person_ids,
            model_a, ids_a, states_a,
            model_b, ids_b, states_b,
            device,
            padding=args.padding,
        )
        draw_header(fa, info_a["label"], 0, tc, "left")
        draw_header(fb, info_b["label"], 0, tc, "right")
        canvas = make_side_by_side(fa, fb, args.display_width, args.display_height)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print("Gorsel gosteriliyor. q ile cik, s ile kaydet.")
        while True:
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out_path = resolve_path("runs/live_compare") / f"screenshot_{int(time.time())}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), canvas)
                print(f"Screenshot: {out_path}")
        cv2.destroyAllWindows()
        return

    # Video veya kamera
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise SystemExit(f"Kaynak acilamadi: {source}")

    if args.save:
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = resolve_path("runs/live_compare")
        out_dir.mkdir(parents=True, exist_ok=True)
        src_stem = source.stem if isinstance(source, Path) else f"cam{source}"
        out_path = out_dir / f"{src_stem}_{args.left}_vs_{args.right}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (fw * 2 + 3, fh))
        print(f"Kaydediliyor: {out_path}")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print("Live view basladi. q ile cik, s ile screenshot.")

    frame_idx = 0
    t_prev = time.perf_counter()
    fps_display = 0.0

    while cap.isOpened():
        ok, frame_raw = cap.read()
        if not ok:
            break
        frame_idx += 1

        fa, fb, tc = process_frame(
            frame_raw, person_model, person_ids,
            model_a, ids_a, states_a,
            model_b, ids_b, states_b,
            device,
            padding=args.padding,
        )

        # FPS hesapla
        now = time.perf_counter()
        fps_display = 0.8 * fps_display + 0.2 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        # Istatistik paneli
        def _counts(states, keys):
            all_votes = [vote(q) for q in states.values() if q]
            c = Counter(all_votes)
            pos = sum(c[k] for k in keys if k in POSITIVE_LABELS)
            neg = sum(c[k] for k in keys if k in NEGATIVE_LABELS)
            unk = c.get("unknown", 0)
            known = pos + neg
            return known, unk, pos, neg

        kna, una, posa, nega = _counts(states_a, info_a["classes"])
        knb, unb, posb, negb = _counts(states_b, info_b["classes"])

        draw_header(fa, info_a["label"], fps_display, tc, "left")
        draw_header(fb, info_b["label"], fps_display, tc, "right")
        draw_stats_panel(fa, kna, una, posa, nega)
        draw_stats_panel(fb, knb, unb, posb, negb)

        canvas = make_side_by_side(fa, fb, args.display_width, args.display_height)
        cv2.imshow(window, canvas)

        if writer is not None:
            writer.write(canvas if canvas.shape[1] == fw * 2 + 3 else cv2.resize(canvas, (fw * 2 + 3, fh)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ss_path = resolve_path("runs/live_compare") / f"screenshot_{int(time.time())}.jpg"
            ss_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ss_path), canvas)
            print(f"Screenshot: {ss_path}")

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if writer:
        print(f"Video kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
