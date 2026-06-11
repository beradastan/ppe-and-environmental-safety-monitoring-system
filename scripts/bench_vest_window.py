# -*- coding: utf-8 -*-
"""
bench_vest_window.py — novest_test üzerinde vest known_rate vs temporal_window
Crop modu: windows [5,10,15,20,30,50], PPE_INFER_EVERY=4, kesinlestirilmis conf
Scene modu: windows [5,10,20,30,40,50], SCENE_PPE_INFER_EVERY=2, kesinlestirilmis conf
"""
from __future__ import annotations

import gc
import sys
import time
from collections import Counter, deque, defaultdict
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HALF        = DEVICE == "cuda"
MAX_WIDTH   = 1920
WARMUP_F    = 30
MAX_FRAMES  = 300
VIDEO       = ROOT / "test" / "novest_test.mp4"  # --video ile override edilebilir
TRACKER     = "bytetrack.yaml"
IMGSZ       = 640

# --- Crop sabitleri ---
CROP_WINDOWS        = [5, 10, 15, 20, 30, 50]
CROP_PPE_EVERY      = 4
CROP_VEST_CONF      = 0.30
CROP_PERSON_CONF    = 0.25
CROP_MIN_CROP_PX    = 40
CROP_MIN_KNOWN      = 3

PERSON_MODEL_PATH   = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"
CROP_VEST_PATH      = ROOT / "models/bera/cropvest_agent_final_best.pt"

# --- Scene sabitleri ---
SCENE_WINDOWS       = [5, 10, 20, 30, 40, 50]
SCENE_PPE_EVERY     = 2
SCENE_VEST_CONF     = 0.30
SCENE_PERSON_CONF   = 0.25
SCENE_INSIDE_FRAC   = 0.20
SCENE_MIN_KNOWN     = 2

SCENE_VEST_PATH     = ROOT / "models/vinayak_trained_byBera/vest_agent_final_best.pt"

VEST_CLASSES = ["Safety Vest", "NO-Safety Vest"]


def _class_ids(model, names):
    n2id = {n: i for i, n in model.names.items()}
    return [n2id[n] for n in names if n in n2id]


def vote(q: deque, min_known: int) -> str:
    if not q:
        return "unknown"
    top = Counter(q).most_common(1)[0][0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    top_l, top_c = Counter(known).most_common(1)[0]
    return top_l if top_c / len(known) >= 0.5 else "unknown"


def _best_det(result, allowed_ids, min_conf):
    if result.boxes is None:
        return None
    best_label, best_conf = None, -1.0
    for box in result.boxes:
        cid = int(box.cls[0]); conf = float(box.conf[0])
        if cid in allowed_ids and conf >= min_conf and conf > best_conf:
            best_label = result.names[cid]; best_conf = conf
    return best_label


def _crop_vest(frame, x1, y1, x2, y2):
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    return frame[max(0, y1 + int(ph * .10)):min(fh, y1 + int(ph * .90)),
                 max(0, x1 - int(pw * .15)):min(fw, x2 + int(pw * .15))]


def _inside_frac(ppe_box, person_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area  = (bx2 - bx1) * (by2 - by1)
    return inter / area if area > 0 else 0.0


def run_crop(person_model, vest_model, window: int) -> float:
    v_ids = _class_ids(vest_model, VEST_CLASSES)
    _pcls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = _class_ids(person_model, [_pcls])

    cap = cv2.VideoCapture(str(VIDEO))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    v_known = v_total = 0
    frame_idx = 0

    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame.shape[1] > MAX_WIDTH:
            s = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * s)))

        do_ppe = (frame_idx % CROP_PPE_EVERY == 0)

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=CROP_PERSON_CONF, device=DEVICE, half=HALF, verbose=False,
        )[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        if do_ppe:
            v_batch = []
            for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
                vc = _crop_vest(frame, x1, y1, x2, y2)
                if vc is not None and vc.size > 0:
                    h, w = vc.shape[:2]
                    if h >= CROP_MIN_CROP_PX and w >= CROP_MIN_CROP_PX:
                        v_batch.append((tid, vc))
            if v_batch:
                for (tid, _), res in zip(
                    v_batch,
                    vest_model.predict(
                        [b[1] for b in v_batch],
                        imgsz=IMGSZ, conf=CROP_VEST_CONF,
                        device=DEVICE, half=HALF, verbose=False,
                    ),
                ):
                    v_deqs[tid].append(_best_det(res, v_ids, CROP_VEST_CONF) or "unknown")

        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            vv = vote(v_deqs[tid], CROP_MIN_KNOWN)
            v_total += 1
            if vv != "unknown":
                v_known += 1

    cap.release()
    return round(100.0 * v_known / v_total, 1) if v_total > 0 else 0.0


def run_scene(person_model, vest_model, window: int) -> float:
    v_ids = set(_class_ids(vest_model, VEST_CLASSES))
    _pcls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = _class_ids(person_model, [_pcls])

    cap = cv2.VideoCapture(str(VIDEO))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    v_known = v_total = 0
    frame_idx = 0

    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame.shape[1] > MAX_WIDTH:
            s = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * s)))

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=SCENE_PERSON_CONF, device=DEVICE, half=HALF, verbose=False,
        )[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        do_ppe = (frame_idx % SCENE_PPE_EVERY == 0)
        if do_ppe:
            v_res = vest_model.predict(
                frame, imgsz=IMGSZ, conf=SCENE_VEST_CONF,
                device=DEVICE, half=HALF, verbose=False,
            )[0]
            v_dets = []
            if v_res.boxes:
                for b in v_res.boxes:
                    cid = int(b.cls[0])
                    if cid in v_ids:
                        v_dets.append((str(vest_model.names[cid]), float(b.conf[0]), b.xyxy[0].tolist()))
        else:
            v_dets = []

        for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
            person_box = [x1, y1, x2, y2]
            best = None
            for lbl, c, bbox in v_dets:
                if _inside_frac(bbox, person_box) >= SCENE_INSIDE_FRAC:
                    if best is None or c > best[1]:
                        best = (lbl, c)
            vlabel = best[0] if best else "unknown"
            v_deqs[tid].append(vlabel)

        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            vv = vote(v_deqs[tid], SCENE_MIN_KNOWN)
            v_total += 1
            if vv != "unknown":
                v_known += 1

    cap.release()
    return round(100.0 * v_known / v_total, 1) if v_total > 0 else 0.0


def main():
    import argparse
    global VIDEO
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=VIDEO)
    args = ap.parse_args()
    VIDEO = args.video
    if not VIDEO.exists():
        sys.exit(f"Video bulunamadi: {VIDEO}")

    # ── CROP MODU ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("CROP MODU — vest_known_rate vs temporal_window")
    print(f"Video: {VIDEO.name}  max_frames={MAX_FRAMES}  ppe_every={CROP_PPE_EVERY}")
    print("=" * 55)
    print("Crop modelleri yukleniyor...")
    p_model_crop = YOLO(str(PERSON_MODEL_PATH))
    v_model_crop = YOLO(str(CROP_VEST_PATH))
    print("  Hazir.\n")

    crop_results = []
    for win in CROP_WINDOWS:
        print(f"  window={win:>2} ...", end="", flush=True)
        t0 = time.perf_counter()
        rate = run_crop(p_model_crop, v_model_crop, win)
        elapsed = time.perf_counter() - t0
        crop_results.append((win, rate))
        print(f" vest_known={rate:.1f}%  ({elapsed:.0f}s)")

    del p_model_crop, v_model_crop
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ── SCENE MODU ─────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("SCENE MODU — vest_known_rate vs temporal_window")
    print(f"Video: {VIDEO.name}  max_frames={MAX_FRAMES}  ppe_every={SCENE_PPE_EVERY}")
    print("=" * 55)
    print("Scene modelleri yukleniyor...")
    p_model_scene = YOLO(str(PERSON_MODEL_PATH))
    v_model_scene = YOLO(str(SCENE_VEST_PATH))
    print("  Hazir.\n")

    scene_results = []
    for win in SCENE_WINDOWS:
        print(f"  window={win:>2} ...", end="", flush=True)
        t0 = time.perf_counter()
        rate = run_scene(p_model_scene, v_model_scene, win)
        elapsed = time.perf_counter() - t0
        scene_results.append((win, rate))
        print(f" vest_known={rate:.1f}%  ({elapsed:.0f}s)")

    del p_model_scene, v_model_scene
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── ÖZET ───────────────────────────────────────────────────────────────
    print()
    print("=" * 40)
    print("ÖZET — Crop modu vest_known_rate")
    print(f"{'window':>8}  {'vest_known%':>12}")
    for win, rate in crop_results:
        print(f"{win:>8}  {rate:>11.1f}%")

    print()
    print("ÖZET — Scene modu vest_known_rate")
    print(f"{'window':>8}  {'vest_known%':>12}")
    for win, rate in scene_results:
        print(f"{win:>8}  {rate:>11.1f}%")


if __name__ == "__main__":
    main()
