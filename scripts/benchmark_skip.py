# -*- coding: utf-8 -*-
"""
benchmark_skip.py — PPE_INFER_EVERY değeri optimizasyonu
=========================================================
Farklı frame-skip değerlerinde hem FPS hem de tespit kalitesini ölçer.

Metrikler (her skip × video × ppe_tipi için):
  fps             — işlem hızı (frame/sn)
  known_rate      — "unknown" olmayan oyların oranı (%)  → daha yüksek = daha iyi
  violation_rate  — ihlal tespiti oranı (bilinen videolarda)  (%)
  first_known_f   — ilk "unknown" olmayan oy için gereken frame sayısı  → daha düşük = daha iyi

Kullanım:
    python scripts/benchmark_skip.py
    python scripts/benchmark_skip.py --skip-values 1 2 3 4 6 8
    python scripts/benchmark_skip.py --max-frames 300 --video nohat_test
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

# ── Üretim sabitleri (run_live_video ile senkron) ────────────────────────────
PERSON_MODEL = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"
HELMET_MODEL = ROOT / "models/bera/crophelmet_agent_final_best.pt"
VEST_MODEL   = ROOT / "models/bera/vest_agent_final_best.pt"
MASK_MODEL   = ROOT / "models/bera/cropmask_agent_final_best.pt"

PERSON_CONF = 0.25
HELMET_CONF = 0.15
VEST_CONF   = 0.30
MASK_CONF   = 0.10
MASK_IMGSZ  = 640   # benchmark: skip karşılaştırması — 1280 CPU'da çok yavaş
IMGSZ       = 640
TRACKER     = "bytetrack.yaml"
TEMPORAL_WIN = 20
MIN_KNOWN    = 3       # vote() için minimum bilinen örnek
WARMUP_F     = 30     # ilk N frame atlanır (tracking stabilizasyonu)
MIN_CROP_PX  = 40

HELMET_CLASSES = ["Hardhat",     "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",        "NO-Mask"]

# Video → beklenen ihlal bilgisi (ground truth)
# {video_stem: {ppe: expected_violation_label}}
GROUND_TRUTH: dict[str, dict[str, str]] = {
    "nohat_test":  {"helmet": "NO-Hardhat"},
    "novest_test": {"vest":   "NO-Safety Vest"},
    "noppe_test":  {"helmet": "NO-Hardhat", "vest": "NO-Safety Vest"},
}

DEFAULT_SKIP_VALUES = [1, 2, 3, 4, 6, 8, 10]
DEFAULT_MAX_FRAMES  = 200

# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _class_ids(model: YOLO, names: list[str]) -> list[int]:
    n2id = {n: i for i, n in model.names.items()}
    return [n2id[n] for n in names if n in n2id]


def _crop_ok(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX


def crop_ppe(frame, x1, y1, x2, y2, ppe: str):
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if ppe == "helmet":
        cx1 = max(0, x1 - int(pw * 0.10))
        cy1 = max(0, y1 - int(ph * 0.15))
        cx2 = min(fw, x2 + int(pw * 0.10))
        cy2 = min(fh, y1 + int(ph * 0.40))
    elif ppe == "vest":
        cx1 = max(0, x1 - int(pw * 0.15))
        cy1 = max(0, y1 + int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.90))
    else:  # mask
        cx1 = max(0, x1 - int(pw * 0.15))
        cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.45))
    return frame[cy1:cy2, cx1:cx2]


def vote(q: deque) -> str:
    if not q:
        return "unknown"
    top = Counter(q).most_common(1)[0][0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < MIN_KNOWN:
        return "unknown"
    top_l, top_c = Counter(known).most_common(1)[0]
    return top_l if top_c / len(known) >= 0.5 else "unknown"


def _best_det(result, allowed_ids: list[int], min_conf: float):
    """En yüksek conf'lu tespit label'ını döndür ya da None."""
    if result.boxes is None:
        return None
    best_label, best_conf = None, -1.0
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid in allowed_ids and conf >= min_conf and conf > best_conf:
            best_label = result.names[cid]
            best_conf  = conf
    return best_label


# ── Ana benchmark fonksiyonu ─────────────────────────────────────────────────

def run_one(video_path: Path, skip: int, models: dict, max_frames: int) -> dict:
    """
    Tek video + tek skip değeri için metrikleri döndür.
    Dönen dict: {fps, helmet_known_rate, vest_known_rate, mask_known_rate,
                 helmet_violation_rate, vest_violation_rate, mask_violation_rate,
                 helmet_first_known, vest_first_known, mask_first_known}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    person_model  = models["person"]
    helmet_model  = models["helmet"]
    vest_model    = models["vest"]
    mask_model    = models["mask"]

    _pcls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = _class_ids(person_model, [_pcls])
    h_ids = _class_ids(helmet_model, HELMET_CLASSES)
    v_ids = _class_ids(vest_model,   VEST_CLASSES)
    m_ids = _class_ids(mask_model,   MASK_CLASSES)

    # Per-track temporal deques
    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))

    # Metrik toplayıcılar (warmup sonrası)
    h_known = h_total = 0
    v_known = v_total = 0
    m_known = m_total = 0
    h_viol  = v_viol  = m_viol = 0

    h_first: dict[int, int | None] = {}
    v_first: dict[int, int | None] = {}
    m_first: dict[int, int | None] = {}

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
        _do_ppe = (frame_idx % skip == 0)

        # Person tracking
        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_AUTO_DEVICE, verbose=False,
        )[0]

        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        # Batch PPE inference
        if _do_ppe:
            h_batch: list[tuple] = []  # (tid, crop)
            v_batch: list[tuple] = []
            m_batch: list[tuple] = []

            for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
                hc = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                vc = crop_ppe(frame, x1, y1, x2, y2, "vest")
                mc = crop_ppe(frame, x1, y1, x2, y2, "mask")
                if _crop_ok(hc): h_batch.append((tid, hc))
                if _crop_ok(vc): v_batch.append((tid, vc))
                if _crop_ok(mc): m_batch.append((tid, mc))

            # Helmet
            if h_batch:
                h_results = helmet_model.predict(
                    [b[1] for b in h_batch], imgsz=IMGSZ, conf=HELMET_CONF,
                    device=_AUTO_DEVICE, verbose=False,
                )
                for (tid, _), res in zip(h_batch, h_results):
                    lbl = _best_det(res, h_ids, HELMET_CONF) or "unknown"
                    h_deqs[tid].append(lbl)

            # Vest
            if v_batch:
                v_results = vest_model.predict(
                    [b[1] for b in v_batch], imgsz=IMGSZ, conf=VEST_CONF,
                    device=_AUTO_DEVICE, verbose=False,
                )
                for (tid, _), res in zip(v_batch, v_results):
                    lbl = _best_det(res, v_ids, VEST_CONF) or "unknown"
                    v_deqs[tid].append(lbl)

            # Mask
            if m_batch:
                m_results = mask_model.predict(
                    [b[1] for b in m_batch], imgsz=MASK_IMGSZ, conf=MASK_CONF,
                    device=_AUTO_DEVICE, verbose=False,
                )
                for (tid, _), res in zip(m_batch, m_results):
                    lbl = _best_det(res, m_ids, MASK_CONF) or "unknown"
                    m_deqs[tid].append(lbl)

        # Metrik toplama (warmup sonrası)
        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            hvote = vote(h_deqs[tid])
            vvote = vote(v_deqs[tid])
            mvote = vote(m_deqs[tid])

            h_total += 1
            v_total += 1
            m_total += 1

            if hvote != "unknown":
                h_known += 1
                if h_expected and hvote == h_expected:
                    h_viol += 1
                if tid not in h_first:
                    h_first[tid] = frame_idx

            if vvote != "unknown":
                v_known += 1
                if v_expected and vvote == v_expected:
                    v_viol += 1
                if tid not in v_first:
                    v_first[tid] = frame_idx

            if mvote != "unknown":
                m_known += 1
                if m_expected and mvote == m_expected:
                    m_viol += 1
                if tid not in m_first:
                    m_first[tid] = frame_idx

    elapsed = time.perf_counter() - t0
    cap.release()

    fps = frame_idx / elapsed if elapsed > 0 else 0.0

    def _rate(num, den):  return round(100.0 * num / den, 1) if den > 0 else 0.0
    def _first(d: dict): return round(sum(d.values()) / len(d), 1) if d else None

    return {
        "fps":                   round(fps, 1),
        "helmet_known_rate":     _rate(h_known, h_total),
        "vest_known_rate":       _rate(v_known, v_total),
        "mask_known_rate":       _rate(m_known, m_total),
        "helmet_violation_rate": _rate(h_viol, h_known) if h_expected else None,
        "vest_violation_rate":   _rate(v_viol, v_known) if v_expected else None,
        "mask_violation_rate":   _rate(m_viol, m_known) if m_expected else None,
        "helmet_first_known":    _first(h_first),
        "vest_first_known":      _first(v_first),
        "mask_first_known":      _first(m_first),
    }


# ── Tablo yazdırma ────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return f"{v:6.1f}"
    return str(v)


def print_table(rows: list[dict], skip_values: list[int], videos: list[str]):
    ppetypes = ["helmet", "vest", "mask"]
    header_cols = ["skip", "video", "fps",
                   "H-known%", "V-known%", "M-known%",
                   "H-viol%",  "V-viol%",  "M-viol%",
                   "H-1st",    "V-1st",    "M-1st"]
    widths = [5, 16, 6, 9, 9, 9, 8, 8, 8, 7, 7, 7]

    def row_str(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    sep = "  ".join("-" * w for w in widths)
    print("\n" + row_str(header_cols))
    print(sep)
    prev_skip = None
    for row in rows:
        if prev_skip is not None and row["skip"] != prev_skip:
            print(sep)
        cells = [
            row["skip"], row["video"],
            _fmt(row["fps"]),
            _fmt(row["helmet_known_rate"]),
            _fmt(row["vest_known_rate"]),
            _fmt(row["mask_known_rate"]),
            _fmt(row["helmet_violation_rate"]),
            _fmt(row["vest_violation_rate"]),
            _fmt(row["mask_violation_rate"]),
            _fmt(row["helmet_first_known"]),
            _fmt(row["vest_first_known"]),
            _fmt(row["mask_first_known"]),
        ]
        print(row_str(cells))
        prev_skip = row["skip"]
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-values", nargs="+", type=int, default=DEFAULT_SKIP_VALUES,
                    metavar="N", help="Test edilecek skip değerleri")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--video", help="Tek video stem (örn. nohat_test)")
    args = ap.parse_args()

    # Test videoları
    test_dir = ROOT / "test"
    all_videos = sorted(test_dir.glob("*.mp4"))
    if args.video:
        all_videos = [v for v in all_videos if v.stem == args.video]
        if not all_videos:
            sys.exit(f"Video bulunamadı: {args.video}")

    if not all_videos:
        sys.exit(f"test/ klasöründe .mp4 bulunamadı: {test_dir}")

    print(f"Modeller yükleniyor...")
    models = {
        "person": YOLO(str(PERSON_MODEL)),
        "helmet": YOLO(str(HELMET_MODEL)),
        "vest":   YOLO(str(VEST_MODEL)),
        "mask":   YOLO(str(MASK_MODEL)),
    }
    print(f"  Hazır. {len(all_videos)} video, skip değerleri: {args.skip_values}")
    print(f"  Max frame / run: {args.max_frames}\n")

    all_rows: list[dict] = []

    for skip in args.skip_values:
        for vid in all_videos:
            print(f"  skip={skip}  {vid.name}  ", end="", flush=True)
            t0 = time.perf_counter()
            metrics = run_one(vid, skip, models, args.max_frames)
            elapsed = time.perf_counter() - t0
            print(f"→ fps={metrics.get('fps', '?')}  "
                  f"H-known={metrics.get('helmet_known_rate')}%  "
                  f"V-known={metrics.get('vest_known_rate')}%  "
                  f"({elapsed:.0f}s)")
            row = {"skip": skip, "video": vid.stem, **metrics}
            all_rows.append(row)

    print_table(all_rows, args.skip_values, [v.stem for v in all_videos])

    # CSV kaydet
    out_dir = ROOT / "runs" / "benchmarks" / "skip"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir / f"skip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Sonuçlar kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
