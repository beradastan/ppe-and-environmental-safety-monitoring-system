# -*- coding: utf-8 -*-
"""
scene_ppe_demo.py
=================
Gerçek sahne tabanlı PPE ihlal tespiti.
Kişi crop'u YOK — PPE modeli doğrudan tam kareye uygulanır.
İhlal bbox'ları ekranda gösterilir, temporal filtreleme uygulanır.

Kullanım:
    python scripts/scene_ppe_demo.py --video test/noppe_test.mp4
    python scripts/scene_ppe_demo.py --video test/nohat_test.mp4 --model vinayak
    python scripts/scene_ppe_demo.py --video test/novest_test.mp4 --conf 0.30
    python scripts/scene_ppe_demo.py --camera 0
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Model tanımları (id → dosya + violation sınıf isimleri)
# ---------------------------------------------------------------------------

MODELS = {
    # Helmet + vest odaklı — cross-class FP en düşük (6.6k görüntü)
    "yihong": {
        "file":      "models/yihong/models/yolo11/pt/yolo11m.pt",
        "no_helmet": ["NO-Hardhat"],
        "no_vest":   ["NO-Safety Vest"],
        "no_mask":   [],           # mask ajan ayrıca eklenecek
    },
    "vinayak": {
        "file":      "models/pretrained/vinayakmane/ppe.pt",
        "no_helmet": ["NO-Hardhat"],
        "no_vest":   ["NO-Safety Vest"],
        "no_mask":   ["NO-Mask"],
    },
    "voxdroid": {
        "file":      "models/voxdroid_200epoch_best.pt",
        "no_helmet": ["NO-Hardhat"],
        "no_vest":   ["NO-Safety Vest"],
        "no_mask":   ["NO-Mask"],
    },
    "hansung": {
        "file":      "models/hansung_yolov8_ppe.pt",
        "no_helmet": ["NO-Hardhat"],
        "no_vest":   ["NO-Safety Vest"],
        "no_mask":   ["NO-Mask"],
    },
}

# Görüntüleme renkleri
COLOR_HELMET = (0,  80, 255)   # turuncu-kırmızı
COLOR_VEST   = (0, 200, 255)   # sarı
COLOR_MASK   = (255, 60,  60)  # mavi
COLOR_OK     = (0, 210,  0)    # yeşil

VIOLATION_COLORS = {
    "helmet": COLOR_HELMET,
    "vest":   COLOR_VEST,
    "mask":   COLOR_MASK,
}
VIOLATION_LABELS = {
    "helmet": "NO HELMET",
    "vest":   "NO VEST",
    "mask":   "NO MASK",
}


def _draw_box(frame, x1, y1, x2, y2, label: str, color, conf: float) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tx, ty = x1, max(y1 - 4, th + 4)
    cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 6, ty + 2), color, -1)
    cv2.putText(frame, text, (tx + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_panel(frame, lines: list[str]) -> None:
    if not lines:
        return
    scale, thick = 0.6, 2
    lh = 24
    mw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines)
    pw, ph = mw + 20, lh * len(lines) + 16
    cv2.rectangle(frame, (8, 8), (8 + pw, 8 + ph), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (8 + pw, 8 + ph), (180, 180, 180), 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, 8 + 18 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thick, cv2.LINE_AA)


def run(args) -> None:
    from ultralytics import YOLO
    import torch

    meta = MODELS.get(args.model)
    if meta is None:
        print(f"Bilinmeyen model: {args.model}. Seçenekler: {list(MODELS)}")
        return

    model_path = ROOT / meta["file"]
    if not model_path.exists():
        print(f"Model dosyası bulunamadı: {model_path}")
        return

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_half = device == "cuda"
    print(f"Model yükleniyor: {model_path.name}  device={device}")
    yolo = YOLO(str(model_path))

    # Sınıf adı → id
    name_to_id = {v.lower(): k for k, v in yolo.names.items()}

    def ids(labels: list[str]) -> list[int]:
        return [name_to_id[l.lower()] for l in labels if l.lower() in name_to_id]

    h_ids = ids(meta["no_helmet"])
    v_ids = ids(meta["no_vest"])
    m_ids = ids(meta["no_mask"])
    all_viol_ids = h_ids + v_ids + m_ids or None  # None = tüm sınıflar

    viol_id_map: dict[int, str] = {}
    for cid in h_ids: viol_id_map[cid] = "helmet"
    for cid in v_ids: viol_id_map[cid] = "vest"
    for cid in m_ids: viol_id_map[cid] = "mask"

    # Temporal smoothing: son N frame'de kaç kez görüldü
    SMOOTH_WINDOW = args.smooth
    viol_history: dict[str, deque] = {
        "helmet": deque(maxlen=SMOOTH_WINDOW),
        "vest":   deque(maxlen=SMOOTH_WINDOW),
        "mask":   deque(maxlen=SMOOTH_WINDOW),
    }
    SMOOTH_THRESH = max(1, SMOOTH_WINDOW // 3)  # %33 frame'de görülmeli

    # Kaynak: video veya kamera
    source: int | str
    if args.camera is not None:
        source = args.camera
    elif args.video:
        src = ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
        if not src.exists():
            print(f"Video bulunamadı: {src}")
            return
        source = str(src)
    else:
        print("--video veya --camera belirtilmeli")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Kaynak açılamadı: {source}")
        return

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Kaynak: {Path(source).name if isinstance(source, str) else f'kamera {source}'}  "
          f"fps={fps_cap:.0f}  frames={total_frames}")

    frame_idx  = 0
    t_start    = time.perf_counter()
    fps_smooth = deque(maxlen=30)

    win = "Scene PPE Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # İstatistik sayaçları
    total_h = total_v = total_m = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            t0 = time.perf_counter()
            preds = yolo.predict(
                frame,
                classes=all_viol_ids,
                conf=args.conf,
                device=device,
                half=use_half,
                verbose=False,
            )
            ms = (time.perf_counter() - t0) * 1000
            fps_smooth.append(1000 / ms if ms > 0 else 0)

            draw = frame.copy()
            pred = preds[0] if preds else None

            # Bu frame'deki ihlaller
            frame_viols: dict[str, list[tuple]] = {"helmet": [], "vest": [], "mask": []}

            if pred is not None and pred.boxes is not None:
                for box in pred.boxes:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    vtype = viol_id_map.get(cid)
                    if vtype is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    frame_viols[vtype].append((x1, y1, x2, y2, conf))

            # Temporal smoothing
            for vtype in ("helmet", "vest", "mask"):
                viol_history[vtype].append(1 if frame_viols[vtype] else 0)

            smoothed: dict[str, bool] = {
                vtype: sum(viol_history[vtype]) >= SMOOTH_THRESH
                for vtype in ("helmet", "vest", "mask")
            }

            # Kutu çiz (sadece smoothed ihlaller için)
            for vtype, dets in frame_viols.items():
                if not smoothed[vtype]:
                    continue
                color = VIOLATION_COLORS[vtype]
                label = VIOLATION_LABELS[vtype]
                for x1, y1, x2, y2, conf in dets:
                    _draw_box(draw, x1, y1, x2, y2, label, color, conf)

            # Sayaç
            if smoothed["helmet"]: total_h += 1
            if smoothed["vest"]:   total_v += 1
            if smoothed["mask"]:   total_m += 1

            # Panel
            cur_fps  = sum(fps_smooth) / len(fps_smooth) if fps_smooth else 0
            elapsed  = time.perf_counter() - t_start
            active   = [VIOLATION_LABELS[vt] for vt in ("helmet","vest","mask") if smoothed[vt]]
            status   = ", ".join(active) if active else "OK"

            panel_lines = [
                f"Model: {args.model}  conf={args.conf:.2f}  smooth={SMOOTH_WINDOW}",
                f"FPS: {cur_fps:.0f}  Frame: {frame_idx}/{total_frames}  t={elapsed:.0f}s",
                f"Durum: {status}",
                f"Toplam ihlal-frame: H={total_h} V={total_v} M={total_m}",
            ]
            if not meta["no_vest"]:
                panel_lines.append("  [!] Bu modelde vest sinifi yok")

            _draw_panel(draw, panel_lines)
            cv2.imshow(win, draw)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            # Tuş kısayolları: + / - conf ayarlama
            if key == ord('+') or key == ord('='):
                args.conf = min(0.95, args.conf + 0.05)
            if key == ord('-'):
                args.conf = max(0.05, args.conf - 0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\nBitti: {frame_idx} frame  {elapsed:.1f}s  "
          f"ihlal-frame: H={total_h} V={total_v} M={total_m}")


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--video",  help="Video dosyası yolu")
    grp.add_argument("--camera", type=int, help="Kamera index (örn: 0)")
    ap.add_argument("--model",  default="yihong", choices=list(MODELS))
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Temporal smoothing penceresi (frame sayısı, varsayılan=5)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
