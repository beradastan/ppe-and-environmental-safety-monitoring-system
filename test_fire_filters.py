# -*- coding: utf-8 -*-
"""
test_fire_filters.py
====================
fire_test.mp4 üzerinde farklı filtre kombinasyonlarını çalıştırır,
hangi frame'lerin alarm ürettiğini ve raw detection vs filtered detection
farkını raporlar.

Kullanım:
    python test_fire_filters.py
"""
from __future__ import annotations

import sys
import os
from collections import deque
from pathlib import Path

os.chdir(Path(__file__).parent)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch
from ultralytics import YOLO

VIDEO      = "test/fire_test.mp4"
MODEL_PATH = "models/bera/fire_smoke_other_agent_final_best.pt"
FIRE_CONF  = 0.75
IMGSZ      = 640
INFER_EVERY = 5   # run_live_video.py ile aynı

CONFIGS = [
    {"name": "Mevcut   (0.01 alan / 1.5x büyüme)",  "area": 0.01, "factor": 1.5, "window": 10},
    {"name": "Gevşek   (0.005 alan / 1.3x büyüme)", "area": 0.005,"factor": 1.3, "window": 10},
    {"name": "Sıkı     (0.02 alan / 2.0x büyüme)",  "area": 0.02, "factor": 2.0, "window": 10},
    {"name": "Sadece alan (0.01, büyüme kapalı)",   "area": 0.01, "factor": 999, "window": 10},
    {"name": "Sadece büyüme (alan kapalı / 1.5x)",  "area": 0.0,  "factor": 1.5, "window": 10},
]


def run_config(model, frames_data, cfg: dict) -> dict:
    area_thr  = cfg["area"]
    factor    = cfg["factor"]
    window    = cfg["window"]
    history: deque = deque(maxlen=window * 2)

    raw_alarms      = 0
    filtered_alarms = 0
    alarm_frames:   list[int] = []
    raw_frames:     list[int] = []
    max_area_seen   = 0.0

    for frame_idx, (frame, frame_area) in enumerate(frames_data):
        if frame_idx % INFER_EVERY != 0:
            continue

        res = model.predict(frame, imgsz=IMGSZ, conf=FIRE_CONF,
                            device=DEVICE, verbose=False)[0]

        fire_raw   = False
        max_area   = 0.0
        for box in res.boxes:
            cid  = int(box.cls[0])
            name = model.names[cid]
            if name != "fire":
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
            max_area   = max(max_area, area_ratio)
            fire_raw   = True

        history.append(max_area)
        max_area_seen = max(max_area_seen, max_area)

        if not fire_raw:
            continue

        raw_alarms += 1
        raw_frames.append(frame_idx)

        is_large   = max_area >= area_thr
        is_growing = False
        if max_area > 0 and len(history) >= window:
            half      = window // 2
            hist      = list(history)
            older_avg = sum(hist[-window:-half]) / half if half else 0
            newer_avg = sum(hist[-half:]) / half if half else 0
            is_growing = older_avg > 0 and (newer_avg / older_avg) >= factor

        if is_large or is_growing:
            filtered_alarms += 1
            alarm_frames.append(frame_idx)

    return {
        "raw":      raw_alarms,
        "filtered": filtered_alarms,
        "max_area": max_area_seen,
        "frames":   alarm_frames,
        "raw_f":    raw_frames,
    }


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Video açılamadı: {VIDEO}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO}  |  {total} frame  |  {fps:.1f} FPS")
    print(f"Model: {MODEL_PATH}  |  FIRE_CONF={FIRE_CONF}\n")

    # Tüm frame'leri oku (küçük video varsayımı)
    frames_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fh, fw = frame.shape[:2]
        frames_data.append((frame, fh * fw or 1))
    cap.release()
    print(f"{len(frames_data)} frame okundu.\n")

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    header = f"{'Konfigürasyon':<44} {'Ham':>5} {'Filtreli':>9} {'MaxAlan':>9}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        r = run_config(model, frames_data, cfg)
        print(f"{cfg['name']:<44} {r['raw']:>5} {r['filtered']:>9} {r['max_area']:>9.4f}")

    # Mevcut config için detaylı alarm zamanları
    print("\n--- Mevcut config alarm frame'leri (ilk 20) ---")
    r0 = run_config(model, frames_data, CONFIGS[0])
    times = [f"{int(f/fps)}s" for f in r0["frames"][:20]]
    print("Alarm :", ", ".join(times) if times else "(yok)")
    raw_t  = [f"{int(f/fps)}s" for f in r0["raw_f"][:20]]
    print("Ham   :", ", ".join(raw_t) if raw_t else "(yok)")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    main()
