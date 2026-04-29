# -*- coding: utf-8 -*-
"""
compare_person_live.py
Vinayak PPE modeli (Person class) vs COCO yolov8m (person class) — yan yana canli karsilastirma.
Kullanim:
    python scripts/compare_person_live.py --video test/noppe_test.mp4
    python scripts/compare_person_live.py --video test/nohat_test.mp4 --conf 0.25
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VINAYAK_PATH = ROOT / "models/pretrained/vinayakmane/ppe.pt"
COCO_PATH    = ROOT / "models/yolov8m.pt"

PERSON_LABELS = {"Person", "person"}

COLOR_VIN  = (50, 200, 50)    # yesil — vinayak
COLOR_COCO = (50, 100, 255)   # turuncu — coco


def _draw_persons(img, preds, names, color, label_prefix: str):
    if not preds or preds[0].boxes is None:
        return 0
    count = 0
    for box in preds[0].boxes:
        lbl = names[int(box.cls[0])]
        if lbl not in PERSON_LABELS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_bg = max(y1, th + 4)
        cv2.rectangle(img, (x1, y_bg - th - 4), (x1 + tw + 6, y_bg + 2), color, -1)
        bright = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        tc = (0, 0, 0) if bright > 120 else (255, 255, 255)
        cv2.putText(img, text, (x1 + 3, y_bg - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2, cv2.LINE_AA)
        count += 1
    return count


def _hud(img, title: str, n_persons: int, fps: float, conf: float, color):
    lines = [title, f"Kisi: {n_persons}  FPS: {fps:.0f}  conf: {conf:.2f}"]
    scale, thick, lh = 0.52, 2, 20
    mw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines)
    pw, ph = mw + 14, lh * len(lines) + 10
    cv2.rectangle(img, (5, 5), (5 + pw, 5 + ph), (20, 20, 20), -1)
    cv2.rectangle(img, (5, 5), (5 + pw, 5 + ph), color, 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 5 + 17 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    use_half = (device == "cuda")

    from ultralytics import YOLO
    print(f"Model yukleniyor  device={device}")
    print(f"  Vinayak : {VINAYAK_PATH.name}")
    vin_model = YOLO(str(VINAYAK_PATH))
    print(f"  COCO    : {COCO_PATH.name}")
    coco_model = YOLO(str(COCO_PATH))

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / args.video

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Video acilamadi: {video_path}")
        return

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    win_name = "Person Karsilastirma  [Q=cikis  SPACE=duraklat]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(fw * 2, 1920), min(fh, 1080))

    print(f"Video: {video_path.name}  ({total_f} frame  {fw}x{fh})")
    print("Q veya ESC ile cikis, SPACE ile duraklat.")

    fps_ring: list[float] = []
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            frame_idx += 1

        t0 = time.perf_counter()

        left  = frame.copy()
        right = frame.copy()

        vin_preds  = vin_model.predict(frame,  conf=args.conf, device=device,
                                       half=use_half, verbose=False)
        coco_preds = coco_model.predict(frame, conf=args.conf, device=device,
                                        half=use_half, verbose=False)

        n_vin  = _draw_persons(left,  vin_preds,  vin_model.names,  COLOR_VIN,  "V")
        n_coco = _draw_persons(right, coco_preds, coco_model.names, COLOR_COCO, "C")

        elapsed = time.perf_counter() - t0
        fps_ring.append(1.0 / elapsed if elapsed > 0 else 0)
        if len(fps_ring) > 20:
            fps_ring.pop(0)
        avg_fps = sum(fps_ring) / len(fps_ring)

        _hud(left,  "Vinayak PPE (Person)", n_vin,  avg_fps, args.conf, COLOR_VIN)
        _hud(right, "COCO yolov8m (person)", n_coco, avg_fps, args.conf, COLOR_COCO)

        # Ortaya dikey ayirac ciz
        combined = cv2.hconcat([left, right])
        mid = fw
        cv2.line(combined, (mid, 0), (mid, fh), (200, 200, 200), 2)

        # Frame sayaci
        cv2.putText(combined, f"{frame_idx}/{total_f}",
                    (fw - 80, fh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(win_name, combined)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
