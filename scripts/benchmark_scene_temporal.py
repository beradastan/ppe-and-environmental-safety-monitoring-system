# -*- coding: utf-8 -*-
"""
benchmark_scene_temporal.py — Scene modu temporal_window optimizasyonu
=======================================================================
Farklı temporal window boyutlarında karar kalitesini ölçer.
Conf ve inside_frac değerleri üretim değerlerinde sabit tutulur.
Scene-tabanlı algılama: tam kareye YOLO predict + inside_frac eşleştirme.

Metrikler:
  known_rate     — "unknown" olmayan temporal oy oranı (%)
  violation_rate — ground truth ihlalinin doğru tespiti (%)
  first_known_f  — ilk kararlı kararın geldiği ortalama frame numarası

Kullanım:
    python scripts/benchmark_scene_temporal.py
    python scripts/benchmark_scene_temporal.py --max-frames 300
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque, defaultdict
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

_AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSON_MODEL = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"
HELMET_MODEL = ROOT / "models/vinayak_trained_byBera/helmet_agent_final_best.pt"
VEST_MODEL   = ROOT / "models/vinayak_trained_byBera/vest_agent_final_best.pt"
MASK_MODEL   = ROOT / "models/mask_agent_scene_200ep_yolov8m_best.pt"

PERSON_CONF     = 0.25
HELMET_CONF     = 0.25   # benchmark_scene_conf sonucu
VEST_CONF       = 0.20   # benchmark_scene_conf sonucu
MASK_CONF       = 0.10   # benchmark_scene_conf sonucu
INSIDE_FRAC_THR = 0.20   # benchmark_scene_frac sonucu
IMGSZ           = 640
TRACKER         = "bytetrack.yaml"
WARMUP_F        = 30
H_MIN_KNOWN     = 2
V_MIN_KNOWN     = 2
M_MIN_KNOWN     = 1

WINDOW_VALUES = [5, 10, 15, 20, 30, 40, 50]

HELMET_CLASSES = ["Hardhat",     "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",        "NO-Mask"]

GROUND_TRUTH: dict[str, dict[str, str]] = {
    "nohat_test":  {"helmet": "NO-Hardhat",      "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest",   "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat",       "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}

DEFAULT_MAX_FRAMES = 300


def _class_ids(model: YOLO, names: list[str]) -> set[int]:
    n2id = {n: i for i, n in model.names.items()}
    return {n2id[n] for n in names if n in n2id}


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


def _inside_frac(ppe_box: list, person_box: list) -> float:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area  = (bx2 - bx1) * (by2 - by1)
    return inter / area if area > 0 else 0.0


def _scene_dets(model: YOLO, frame, allowed_ids: set[int], min_conf: float) -> list[tuple]:
    res = model.predict(frame, imgsz=IMGSZ, conf=min_conf,
                        device=_AUTO_DEVICE, verbose=False)[0]
    if not res.boxes:
        return []
    return [
        (str(model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist())
        for b in res.boxes
        if int(b.cls[0]) in allowed_ids
    ]


def _best_scene(dets: list[tuple], person_box: list,
                frac_thr: float) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for lbl, c, bbox in dets:
        if _inside_frac(bbox, person_box) >= frac_thr:
            if best is None or c > best[1]:
                best = (lbl, c)
    return best if best else ("unknown", 0.0)


def run_one(video_path: Path, models: dict, window: int, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    person_model = models["person"]
    helmet_model = models["helmet"]
    vest_model   = models["vest"]
    mask_model   = models["mask"]

    _pcls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = {i for i, n in person_model.names.items() if n == _pcls}
    h_ids = _class_ids(helmet_model, HELMET_CLASSES)
    v_ids = _class_ids(vest_model,   VEST_CLASSES)
    m_ids = _class_ids(mask_model,   MASK_CLASSES)

    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    h_known = h_total = v_known = v_total = m_known = m_total = 0
    h_viol  = v_viol  = m_viol = 0
    h_first: dict[int, int] = {}
    v_first: dict[int, int] = {}
    m_first: dict[int, int] = {}

    gt = GROUND_TRUTH.get(video_path.stem, {})
    h_exp = gt.get("helmet")
    v_exp = gt.get("vest")
    m_exp = gt.get("mask")

    frame_idx = 0
    t0 = time.perf_counter()

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        p_res = person_model.track(
            frame, classes=list(p_ids), tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_AUTO_DEVICE, verbose=False,
        )[0]

        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        h_dets = _scene_dets(helmet_model, frame, h_ids, HELMET_CONF)
        v_dets = _scene_dets(vest_model,   frame, v_ids, VEST_CONF)
        m_dets = _scene_dets(mask_model,   frame, m_ids, MASK_CONF)

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
            person_box = [x1, y1, x2, y2]
            hlabel, _ = _best_scene(h_dets, person_box, INSIDE_FRAC_THR)
            vlabel, _ = _best_scene(v_dets, person_box, INSIDE_FRAC_THR)
            mlabel, _ = _best_scene(m_dets, person_box, INSIDE_FRAC_THR)
            h_deqs[tid].append(hlabel)
            v_deqs[tid].append(vlabel)
            m_deqs[tid].append(mlabel)

        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            hv = vote(h_deqs[tid], H_MIN_KNOWN)
            vv = vote(v_deqs[tid], V_MIN_KNOWN)
            mv = vote(m_deqs[tid], M_MIN_KNOWN)
            h_total += 1; v_total += 1; m_total += 1
            if hv != "unknown":
                h_known += 1
                if h_exp and hv == h_exp:
                    h_viol += 1
                if tid not in h_first:
                    h_first[tid] = frame_idx
            if vv != "unknown":
                v_known += 1
                if v_exp and vv == v_exp:
                    v_viol += 1
                if tid not in v_first:
                    v_first[tid] = frame_idx
            if mv != "unknown":
                m_known += 1
                if m_exp and mv == m_exp:
                    m_viol += 1
                if tid not in m_first:
                    m_first[tid] = frame_idx

    elapsed = time.perf_counter() - t0
    cap.release()

    def _r(n, d): return round(100.0 * n / d, 1) if d > 0 else 0.0
    def _first(d): return round(sum(d.values()) / len(d), 1) if d else None

    return {
        "fps":                   round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
        "helmet_known_rate":     _r(h_known, h_total),
        "vest_known_rate":       _r(v_known, v_total),
        "mask_known_rate":       _r(m_known, m_total),
        "helmet_violation_rate": _r(h_viol, h_known) if h_exp else None,
        "vest_violation_rate":   _r(v_viol, v_known) if v_exp else None,
        "mask_violation_rate":   _r(m_viol, m_known) if m_exp else None,
        "helmet_first_known":    _first(h_first),
        "vest_first_known":      _first(v_first),
        "mask_first_known":      _first(m_first),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--window-values", nargs="+", type=int, default=WINDOW_VALUES)
    args = ap.parse_args()

    test_dir = ROOT / "test"
    test_videos = sorted(test_dir.glob("*.mp4"))
    if not test_videos:
        sys.exit(f"test/ klasöründe .mp4 bulunamadı: {test_dir}")

    print("Scene modelleri yükleniyor...")
    models = {
        "person": YOLO(str(PERSON_MODEL)),
        "helmet": YOLO(str(HELMET_MODEL)),
        "vest":   YOLO(str(VEST_MODEL)),
        "mask":   YOLO(str(MASK_MODEL)),
    }
    print(f"  Hazır. {len(test_videos)} video, window değerleri: {args.window_values}\n")

    all_rows: list[dict] = []

    for win in args.window_values:
        for vid in test_videos:
            print(f"  window={win:>2}  {vid.name:<25}", end="", flush=True)
            t0 = time.perf_counter()
            m = run_one(vid, models, win, args.max_frames)
            print(f"→ fps={m.get('fps','?')}  "
                  f"H-known={m.get('helmet_known_rate')}%  "
                  f"H-1st={m.get('helmet_first_known')}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            all_rows.append({"window": win, "video": vid.stem, **m})

    hdr = (f"{'win':>4}  {'video':<20}  {'fps':>6}  "
           f"{'H-know%':>8}  {'V-know%':>8}  {'M-know%':>8}  "
           f"{'H-viol%':>8}  {'V-viol%':>8}  {'M-viol%':>8}  "
           f"{'H-1st':>6}  {'V-1st':>6}  {'M-1st':>6}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    prev_win = None
    for row in all_rows:
        if prev_win is not None and row["window"] != prev_win:
            print()
        def f(v): return f"{v:6.1f}" if v is not None and not isinstance(v, str) else "   -  "
        print(f"{row['window']:>4}  {row['video']:<20}  "
              f"{f(row['fps'])}  "
              f"{f(row['helmet_known_rate'])}  {f(row['vest_known_rate'])}  {f(row['mask_known_rate'])}  "
              f"{f(row['helmet_violation_rate'])}  {f(row['vest_violation_rate'])}  {f(row['mask_violation_rate'])}  "
              f"{f(row['helmet_first_known'])}  {f(row['vest_first_known'])}  {f(row['mask_first_known'])}")
        prev_win = row["window"]

    out_dir = ROOT / "runs" / "benchmarks" / "scene_temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir / f"scene_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSonuçlar kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
