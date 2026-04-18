# -*- coding: utf-8 -*-
"""
scripts/wesjos_agent_demo.py
============================
Vinayak PPE pipeline demo.

Kullanim:
    python scripts/wesjos_agent_demo.py --model vinayak_ppe
    python scripts/wesjos_agent_demo.py --model vinayak_ppe --person-model small --tracker bytetrack.yaml --crop-padding 0.30

Klavye (pencere acikken):
    Q / ESC   -> Cik
    R         -> Menuye don
    SPACE     -> Duraklat / Devam
    F         -> Tam ekran / Pencere
    + / -     -> PPE guven esigini ayarla (%5)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import yaml

# ── Display sabitleri ────────────────────────────────────────────────────────
DISPLAY_W = 1280
DISPLAY_H = 720
FONT      = cv2.FONT_HERSHEY_SIMPLEX
WIN       = "Wesjos | Vinayak PPE Demo"

# BGR renk paleti
C_GREEN  = (0, 200, 0)      # hardhat + vest tamam
C_ORANGE = (0, 140, 255)    # bir PPE eksik
C_RED    = (0, 0, 220)      # hardhat + vest her ikisi eksik
C_YELLOW = (0, 210, 210)    # bilinmiyor / belirsiz
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_GRAY   = (160, 160, 160)
C_BLUE   = (220, 100, 0)

# ── Yardimci fonksiyonlar ────────────────────────────────────────────────────

def _label(img, text, x, y, color=C_WHITE, scale=0.48):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 2), C_BLACK, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _resize(frame):
    oh, ow = frame.shape[:2]
    scale  = min(DISPLAY_W / ow, DISPLAY_H / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    px = (DISPLAY_W - nw) // 2
    py = (DISPLAY_H - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return canvas, scale, scale, px, py


# ── Registry okuyucu ─────────────────────────────────────────────────────────

def load_registry(path="models/registry.yaml") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("models", [])


def find_model(registry: list[dict], model_id: str) -> dict:
    for m in registry:
        if m["id"] == model_id:
            return m
    raise ValueError(f"Registry'de model bulunamadi: {model_id!r}")


def build_label_map(model_meta: dict) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for key, labels in model_meta.get("ppe_classes", {}).items():
        if key.endswith("_present"):
            cat, status = key[:-8], "present"
        elif key.endswith("_missing"):
            cat, status = key[:-8], "missing"
        else:
            continue
        for lbl in labels:
            mapping[lbl.strip().lower()] = (cat, status)
    return mapping


# ── Model secimleri ──────────────────────────────────────────────────────────

PERSON_MODEL_MAP = {
    "small": "models/pretrained/person/person_yolov8s-seg.pt",
    "nano":  "models/yakhyo_yolov8n_best.pt",
}

PERSON_CONF_MAP = {
    "small": 0.25,
    "nano":  0.40,
}


# ── Terminal menusu ──────────────────────────────────────────────────────────

def _menu_select(title: str, options: list[str]) -> int:
    SEP = "=" * 58
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print(SEP)
    while True:
        try:
            raw = input(f"  Secim (1-{len(options)}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            print("  Gecersiz numara.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\n  Iptal.")
            sys.exit(0)


def pick_video() -> str:
    test_dir = Path("test")
    videos = sorted(test_dir.glob("*.mp4")) + sorted(test_dir.glob("*.avi"))
    if not videos:
        print("test/ dizininde video bulunamadi.")
        sys.exit(1)
    idx = _menu_select("VIDEO SEC", [v.name for v in videos])
    return str(videos[idx])


# ── PPE agent wrapper ────────────────────────────────────────────────────────

def build_ppe_agent(model_meta: dict, device: str, conf: float):
    from ultralytics import YOLO
    import torch

    yolo  = YOLO(model_meta["file"])
    lmap  = build_label_map(model_meta)
    cids  = [cid for cid, nm in yolo.names.items() if nm.strip().lower() in lmap]
    if device == "cuda" and torch.cuda.is_available():
        yolo.to("cuda")

    class _PPEAgent:
        def __init__(self):
            self.model          = yolo
            self.confidence     = conf
            self.device         = device
            self._ppe_class_ids = cids
            self._label_map     = lmap

        def detect(self, frame):
            return self._run([frame])[0]

        def detect_batch(self, crops):
            return self._run(crops) if crops else []

        def _run(self, imgs):
            try:
                res = self.model(
                    imgs,
                    conf=self.confidence,
                    classes=self._ppe_class_ids or None,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                return [[] for _ in imgs]
            out = []
            for r in res:
                dets = []
                if r.boxes is not None:
                    for box in r.boxes:
                        nm = self.model.names.get(int(box.cls[0]), "").strip().lower()
                        mp = self._label_map.get(nm)
                        if mp is None:
                            continue
                        cat, status = mp
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        dets.append({
                            "label":      nm,
                            "bbox":       [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "category":   cat,
                            "status":     status,
                        })
                out.append(dets)
            return out

    return _PPEAgent()


# ── Person agent wrapper (seg model) ────────────────────────────────────────

def build_person_agent(model_path: str, device: str, conf: float, tracker: str):
    from ultralytics import YOLO
    import torch

    yolo = YOLO(model_path)
    if device == "cuda" and torch.cuda.is_available():
        yolo.to("cuda")

    hit_counts: dict[int, int] = {}
    MIN_HITS = 3

    class _PersonAgent:
        def __init__(self):
            self.model      = yolo
            self.confidence = conf
            self.device     = device

        def detect(self, frame) -> list[dict]:
            try:
                results = self.model.track(
                    source=frame,
                    conf=self.confidence,
                    classes=[0],        # person only
                    tracker=tracker,
                    persist=True,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                return []

            if not results or results[0].boxes is None:
                return []

            out = []
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid  = int(box.id[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cf   = float(box.conf[0])
                hit_counts[tid] = hit_counts.get(tid, 0) + 1
                out.append({
                    "track_id":    tid,
                    "bbox":        [x1, y1, x2, y2],
                    "confidence":  cf,
                    "is_confirmed": hit_counts[tid] >= MIN_HITS,
                })
            return out

        def reset(self):
            hit_counts.clear()
            if hasattr(self.model, "predictor") and self.model.predictor:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers = None

    return _PersonAgent()


# ── Kisi rengi ──────────────────────────────────────────────────────────────

def _person_color(helmet_status: str, vest_status: str, check_helmet: bool, check_vest: bool) -> tuple:
    helmet_missing = check_helmet and helmet_status == "missing"
    vest_missing   = check_vest   and vest_status   == "missing"

    if helmet_status == "unknown" and vest_status == "unknown":
        return C_YELLOW
    if helmet_missing and vest_missing:
        return C_RED
    if helmet_missing or vest_missing:
        return C_ORANGE
    return C_GREEN


# ── Frame cizici ─────────────────────────────────────────────────────────────

def draw_frame(
    frame:          np.ndarray,
    person_results: list,
    ppe_dets:       list,
    event_result:   dict,
    fps:            float,
    model_id:       str,
    person_model:   str,
    conf_ppe:       float,
    paused:         bool,
    check_helmet:   bool,
    check_vest:     bool,
) -> np.ndarray:
    out, sx, sy, px, py = _resize(frame)

    # PPE kutulari
    for det in ppe_dets:
        x1 = int(det["bbox"][0] * sx + px); y1 = int(det["bbox"][1] * sy + py)
        x2 = int(det["bbox"][2] * sx + px); y2 = int(det["bbox"][3] * sy + py)
        col = C_RED if det["status"] == "missing" else C_GREEN
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        _label(out, f"{det['label']} {det['confidence']:.0%}", x1, y2 + 14, col, 0.36)

    # Kisi kutulari
    for p in person_results:
        ox1, oy1, ox2, oy2 = p["person_bbox"]
        x1 = int(ox1 * sx + px); y1 = int(oy1 * sy + py)
        x2 = int(ox2 * sx + px); y2 = int(oy2 * sy + py)

        col = _person_color(p["helmet_status"], p["vest_status"], check_helmet, check_vest)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        tid = p.get("track_id")
        viols = p.get("violations", [])
        tag = f"#{tid}"
        if viols:
            tag += " " + ",".join(v.replace("no_", "!") for v in viols)
        _label(out, tag, x1, y1 - 5, col)

        # PPE durum satirlari (kisi kutusunun alt kosesi)
        line_y = y2 + 15
        for item, status_key in [("H", "helmet_status"), ("V", "vest_status")]:
            st = p.get(status_key, "unknown")
            c  = C_GREEN if st == "present" else (C_RED if st == "missing" else C_YELLOW)
            _label(out, f"{item}:{st[0].upper()}", x1, line_y, c, 0.38)
            line_y += 14

    # Olay durumu paneli (ust sol)
    sig = event_result.get("signature", {})
    status = event_result.get("event_status", "idle")
    reason = event_result.get("reason", "")

    panel_color = C_RED if status in ("new", "active") else C_GRAY
    _label(out, f"Event: {status.upper()}", 8, 22, panel_color, 0.55)
    _label(out, reason[:55], 8, 42, C_GRAY, 0.40)

    h_viol = sig.get("helmet_violation", False)
    v_viol = sig.get("vest_violation",   False)
    _label(out, f"Helmet: {'IHLAL' if h_viol else 'OK'}", 8, 62,
           C_RED if h_viol else C_GREEN, 0.42)
    _label(out, f"Vest:   {'IHLAL' if v_viol else 'OK'}", 8, 78,
           C_RED if v_viol else C_GREEN, 0.42)

    # Ust sag: FPS + model bilgisi
    info = f"{fps:.0f} FPS | PPE:{model_id} | Person:{person_model}"
    _label(out, info, 8, DISPLAY_H - 36, C_WHITE, 0.40)
    _label(out, f"conf={conf_ppe:.0%}  [+/-]  |  SPACE=pause  R=menu  Q=quit",
           8, DISPLAY_H - 18, C_GRAY, 0.38)

    if paused:
        _label(out, "DURAKLADI", DISPLAY_W // 2 - 55, DISPLAY_H // 2, C_YELLOW, 0.9)

    return out


# ── Ana video dongusu ─────────────────────────────────────────────────────────

def run_video(
    video_path:   str,
    ppe_agent,
    person_agent,
    args:         argparse.Namespace,
    model_meta:   dict,
) -> str:
    """Videoyu isle. 'menu' veya 'quit' dondur."""
    from matching.crop_ppe_matcher import CropPPEMatcher
    from event_manager import PersonEventManager

    crop_matcher = CropPPEMatcher(
        crop_pad=args.crop_padding,
        check_helmet=True,
        check_vest=True,
        check_mask=False,
    )
    em = PersonEventManager(check_helmet=True, check_vest=True, check_mask=False)
    person_agent.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video acilamadi: {video_path}")
        return "menu"

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paused   = False
    fullscr  = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, DISPLAY_H)

    fps_hist: list[float] = []
    t_last   = time.perf_counter()
    result   = "menu"
    event_result: dict = {"event_status": "idle", "signature": {}, "reason": "baslatiliyor"}

    frame_delay = max(1, int(1000 / vid_fps))

    while True:
        key = cv2.waitKey(frame_delay) & 0xFF

        if key in (ord("q"), 27):          # Q / ESC
            result = "quit"
            break
        if key == ord("r"):                # R — menu
            result = "menu"
            break
        if key == ord(" "):                # SPACE — pause
            paused = not paused
        if key == ord("f"):                # F — fullscreen
            fullscr = not fullscr
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscr else cv2.WINDOW_NORMAL)
        if key == ord("+") or key == 43:   # + — conf artir
            ppe_agent.confidence = min(0.95, ppe_agent.confidence + 0.05)
        if key == ord("-") or key == 45:   # - — conf azalt
            ppe_agent.confidence = max(0.05, ppe_agent.confidence - 0.05)

        if paused:
            continue

        ret, frame = cap.read()
        if not ret:
            # Video bitti — basından başla
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            person_agent.reset()
            em = PersonEventManager(check_helmet=True, check_vest=True, check_mask=False)
            continue

        # Pipeline
        persons   = person_agent.detect(frame)
        confirmed = [p for p in persons if p.get("is_confirmed", True)]

        person_results, ppe_dets = (
            crop_matcher.match(frame, confirmed, ppe_agent)
            if confirmed else ([], [])
        )
        event_result = em.process_frame(person_results)

        # FPS hesaplama
        now  = time.perf_counter()
        fps_hist.append(1.0 / max(now - t_last, 1e-6))
        t_last = now
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        display = draw_frame(
            frame, person_results, ppe_dets, event_result, fps,
            model_id=model_meta["id"],
            person_model=args.person_model,
            conf_ppe=ppe_agent.confidence,
            paused=paused,
            check_helmet=True,
            check_vest=True,
        )
        cv2.imshow(WIN, display)

    cap.release()
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wesjos | Vinayak PPE agent demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",         default="vinayak_ppe",
                   help="Registry'deki PPE model ID'si.")
    p.add_argument("--person-model",  default="small",
                   choices=list(PERSON_MODEL_MAP),
                   dest="person_model",
                   help="Kisi takip modeli: small=yolov8s-seg, nano=yolov8n.")
    p.add_argument("--tracker",       default="bytetrack.yaml",
                   help="Ultralytics tracker config.")
    p.add_argument("--crop-padding",  type=float, default=0.30,
                   dest="crop_padding",
                   help="Kisi bbox crop padding orani.")
    p.add_argument("--device",        default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--conf-ppe",      type=float, default=0.25, dest="conf_ppe",
                   help="PPE model guven esigi.")
    p.add_argument("--video",         default=None,
                   help="Video dosyasi. Belirtilmezse menu acilir.")
    return p.parse_args()


def main() -> None:
    import torch
    args = parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"\n  Wesjos | Vinayak PPE Demo")
    print(f"  Device: {'GPU 0' if device == 'cuda' else 'CPU'}  |  "
          f"Image size: 640  |  Display: {DISPLAY_W}x{DISPLAY_H}")
    print(f"  Tracker: {args.tracker}  |  Crop padding: {args.crop_padding}\n")

    registry   = load_registry()
    model_meta = find_model(registry, args.model)

    if not Path(model_meta["file"]).exists():
        print(f"  HATA: PPE model dosyasi bulunamadi: {model_meta['file']}")
        sys.exit(1)

    person_path = PERSON_MODEL_MAP[args.person_model]
    if not Path(person_path).exists():
        print(f"  HATA: Kisi model dosyasi bulunamadi: {person_path}")
        sys.exit(1)

    print(f"  Modeller yukleniyor...")
    ppe_agent    = build_ppe_agent(model_meta, device, args.conf_ppe)
    person_agent = build_person_agent(
        model_path=person_path,
        device=device,
        conf=PERSON_CONF_MAP[args.person_model],
        tracker=args.tracker,
    )
    print(f"  PPE model   : {model_meta['id']}  ({model_meta['file']})")
    print(f"  Person model: {args.person_model}  ({person_path})")
    print(f"  Hazir.\n")

    while True:
        video = args.video if args.video else pick_video()
        print(f"  Video: {video}")

        action = run_video(video, ppe_agent, person_agent, args, model_meta)

        cv2.destroyAllWindows()
        if action == "quit":
            break
        # action == "menu" → dongu basina don


if __name__ == "__main__":
    main()
