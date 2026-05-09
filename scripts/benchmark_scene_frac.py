# -*- coding: utf-8 -*-
"""
benchmark_scene_frac.py — Scene modu inside_frac eşiği optimizasyonu
====================================================================
`INSIDE_FRAC_THR` (PPE bbox'un kişi bbox içinde kalması gereken minimum oran)
farklı değerlerde taranır. Conf ve temporal_window üretim değerlerinde sabit.

Bu parametre scene moduna özgüdür — crop modunda karşılığı yoktur.

Metrikler:
  known_rate     — "unknown" olmayan temporal oy oranı (%)  → yüksek = iyi
  violation_rate — ground truth ihlalinin doğru tespiti (%)
  fps            — işlem hızı

Kullanım:
    python scripts/benchmark_scene_frac.py
    python scripts/benchmark_scene_frac.py --max-frames 200
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

# ── Sabitler ────────────────────────────────────────────────────────────────
PERSON_MODEL = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"
HELMET_MODEL = ROOT / "models/vinayak_trained_byBera/helmet_agent_final_best.pt"
VEST_MODEL   = ROOT / "models/vinayak_trained_byBera/vest_agent_final_best.pt"
MASK_MODEL   = ROOT / "models/mask_agent_scene_200ep_yolov8m_best.pt"

PERSON_CONF  = 0.25
HELMET_CONF  = 0.20   # üretim değeri
VEST_CONF    = 0.30   # üretim değeri
MASK_CONF    = 0.25   # üretim değeri
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"
TEMPORAL_WIN = 20
WARMUP_F     = 30
H_MIN_KNOWN  = 2
V_MIN_KNOWN  = 2
M_MIN_KNOWN  = 1

# Tarama aralığı
FRAC_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

HELMET_CLASSES = ["Hardhat",     "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",        "NO-Mask"]

GROUND_TRUTH: dict[str, dict[str, str]] = {
    "nohat_test":  {"helmet": "NO-Hardhat",      "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest",   "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat",       "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}

DEFAULT_MAX_FRAMES = 200


# ── Yardımcılar ──────────────────────────────────────────────────────────────

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


# ── Ana ölçüm fonksiyonu ─────────────────────────────────────────────────────

def run_one(video_path: Path, models: dict, frac_thr: float, max_frames: int) -> dict:
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

    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))

    h_known = h_total = v_known = v_total = m_known = m_total = 0
    h_viol  = v_viol  = m_viol = 0

    gt = GROUND_TRUTH.get(video_path.stem, {})
    h_expected = gt.get("helmet")
    v_expected = gt.get("vest")
    m_expected = gt.get("mask")

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
            hlabel, _ = _best_scene(h_dets, person_box, frac_thr)
            vlabel, _ = _best_scene(v_dets, person_box, frac_thr)
            mlabel, _ = _best_scene(m_dets, person_box, frac_thr)
            h_deqs[tid].append(hlabel)
            v_deqs[tid].append(vlabel)
            m_deqs[tid].append(mlabel)

        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            hvote = vote(h_deqs[tid], H_MIN_KNOWN)
            vvote = vote(v_deqs[tid], V_MIN_KNOWN)
            mvote = vote(m_deqs[tid], M_MIN_KNOWN)
            h_total += 1; v_total += 1; m_total += 1
            if hvote != "unknown":
                h_known += 1
                if h_expected and hvote == h_expected:
                    h_viol += 1
            if vvote != "unknown":
                v_known += 1
                if v_expected and vvote == v_expected:
                    v_viol += 1
            if mvote != "unknown":
                m_known += 1
                if m_expected and mvote == m_expected:
                    m_viol += 1

    elapsed = time.perf_counter() - t0
    cap.release()

    def _rate(num, den): return round(100.0 * num / den, 1) if den > 0 else 0.0

    return {
        "fps":                   round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
        "helmet_known_rate":     _rate(h_known, h_total),
        "vest_known_rate":       _rate(v_known, v_total),
        "mask_known_rate":       _rate(m_known, m_total),
        "helmet_violation_rate": _rate(h_viol, h_known) if h_expected else None,
        "vest_violation_rate":   _rate(v_viol, v_known) if v_expected else None,
        "mask_violation_rate":   _rate(m_viol, m_known) if m_expected else None,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--frac-values", nargs="+", type=float, default=FRAC_SWEEP)
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
    print(f"  Hazır. {len(test_videos)} video, frac değerleri: {args.frac_values}\n")

    all_rows: list[dict] = []

    for frac in args.frac_values:
        for vid in test_videos:
            print(f"  frac={frac:.2f}  {vid.name:<25}", end="", flush=True)
            m = run_one(vid, models, frac_thr=frac, max_frames=args.max_frames)
            print(f"→ fps={m.get('fps','?')}  "
                  f"H-known={m.get('helmet_known_rate')}%  "
                  f"H-viol={m.get('helmet_violation_rate')}%  "
                  f"V-known={m.get('vest_known_rate')}%")
            all_rows.append({"frac_thr": frac, "video": vid.stem, **m})

    # Özet tablo
    hdr = (f"{'frac':>5}  {'video':<20}  {'fps':>6}  "
           f"{'H-know%':>8}  {'V-know%':>8}  {'M-know%':>8}  "
           f"{'H-viol%':>8}  {'V-viol%':>8}  {'M-viol%':>8}")
    print(f"\n{'='*len(hdr)}")
    print("  INSIDE_FRAC_THR TARAMASI (SCENE)")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print("-" * len(hdr))
    prev_frac = None
    for row in all_rows:
        if prev_frac is not None and row["frac_thr"] != prev_frac:
            print()
        def f(v): return f"{v:6.1f}" if v is not None and not isinstance(v, str) else "   -  "
        print(f"{row['frac_thr']:>5.2f}  {row['video']:<20}  "
              f"{f(row['fps'])}  "
              f"{f(row['helmet_known_rate'])}  {f(row['vest_known_rate'])}  {f(row['mask_known_rate'])}  "
              f"{f(row['helmet_violation_rate'])}  {f(row['vest_violation_rate'])}  {f(row['mask_violation_rate'])}")
        prev_frac = row["frac_thr"]
    print()

    out_dir = ROOT / "runs" / "benchmarks" / "scene_frac"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir / f"scene_frac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Sonuçlar kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
