# -*- coding: utf-8 -*-
"""
benchmark_conf.py — PPE tespit confidence eşiği optimizasyonu
=============================================================
Her PPE türü için confidence eşiği bağımsız olarak taranır;
diğer iki PPE türünün conf değeri üretim değerinde (config.yaml) sabit tutulur.
PPE_INFER_EVERY = 4 (üretim değeri) sabittir.

Metrikler:
  known_rate     — "unknown" olmayan temporal oy oranı (%)  → yüksek = iyi
  violation_rate — ground truth ihlalinin doğru kararla örtüşme oranı (%)
  fps            — işlem hızı

Kullanım:
    python scripts/benchmark_conf.py
    python scripts/benchmark_conf.py --max-frames 200
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
HELMET_MODEL = ROOT / "models/bera/crophelmet_agent_final_best.pt"
VEST_MODEL   = ROOT / "models/bera/cropvest_agent_final_best.pt"
MASK_MODEL   = ROOT / "models/bera/cropmask_agent_final_best.pt"

PERSON_CONF   = 0.25
IMGSZ         = 640
MASK_IMGSZ    = 640
TRACKER       = "bytetrack.yaml"
TEMPORAL_WIN  = 20
MIN_KNOWN     = 3
WARMUP_F      = 30
MIN_CROP_PX   = 40
PPE_INFER_EVERY = 4   # üretim değeri — sabit

# Üretim conf değerleri (diğer PPE'ler bu değerde tutulur)
PROD_HELMET_CONF = 0.20
PROD_VEST_CONF   = 0.30
PROD_MASK_CONF   = 0.25

# Tarama aralıkları
HELMET_CONF_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
VEST_CONF_SWEEP   = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
MASK_CONF_SWEEP   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

HELMET_CLASSES = ["Hardhat",     "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",        "NO-Mask"]

GROUND_TRUTH: dict[str, dict[str, str]] = {
    "nohat_test":  {"helmet": "NO-Hardhat",       "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest",    "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat",        "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}
# mask_test: vest=ok, mask=ok — FP ölçümü için ayrı kategoride

DEFAULT_MAX_FRAMES = 200

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
        cx1 = max(0, x1 - int(pw * 0.10)); cy1 = max(0, y1 - int(ph * 0.15))
        cx2 = min(fw, x2 + int(pw * 0.10)); cy2 = min(fh, y1 + int(ph * 0.40))
    elif ppe == "vest":
        cx1 = max(0, x1 - int(pw * 0.15)); cy1 = max(0, y1 + int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15)); cy2 = min(fh, y1 + int(ph * 0.90))
    else:
        cx1 = max(0, x1 - int(pw * 0.15)); cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15)); cy2 = min(fh, y1 + int(ph * 0.45))
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


# ── Ana ölçüm fonksiyonu ─────────────────────────────────────────────────────

def run_one(video_path: Path, models: dict,
            helmet_conf: float, vest_conf: float, mask_conf: float,
            max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    person_model = models["person"]
    helmet_model = models["helmet"]
    vest_model   = models["vest"]
    mask_model   = models["mask"]

    _pcls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = _class_ids(person_model, [_pcls])
    h_ids = _class_ids(helmet_model, HELMET_CLASSES)
    v_ids = _class_ids(vest_model,   VEST_CLASSES)
    m_ids = _class_ids(mask_model,   MASK_CLASSES)

    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))

    h_known = h_total = 0
    v_known = v_total = 0
    m_known = m_total = 0
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
        _do_ppe = (frame_idx % PPE_INFER_EVERY == 0)

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_AUTO_DEVICE, verbose=False,
        )[0]

        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        if _do_ppe:
            h_batch, v_batch, m_batch = [], [], []
            for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
                hc = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                vc = crop_ppe(frame, x1, y1, x2, y2, "vest")
                mc = crop_ppe(frame, x1, y1, x2, y2, "mask")
                if _crop_ok(hc): h_batch.append((tid, hc))
                if _crop_ok(vc): v_batch.append((tid, vc))
                if _crop_ok(mc): m_batch.append((tid, mc))

            if h_batch:
                for (tid, _), res in zip(h_batch, helmet_model.predict(
                        [b[1] for b in h_batch], imgsz=IMGSZ,
                        conf=helmet_conf, device=_AUTO_DEVICE, verbose=False)):
                    h_deqs[tid].append(_best_det(res, h_ids, helmet_conf) or "unknown")

            if v_batch:
                for (tid, _), res in zip(v_batch, vest_model.predict(
                        [b[1] for b in v_batch], imgsz=IMGSZ,
                        conf=vest_conf, device=_AUTO_DEVICE, verbose=False)):
                    v_deqs[tid].append(_best_det(res, v_ids, vest_conf) or "unknown")

            if m_batch:
                for (tid, _), res in zip(m_batch, mask_model.predict(
                        [b[1] for b in m_batch], imgsz=MASK_IMGSZ,
                        conf=mask_conf, device=_AUTO_DEVICE, verbose=False)):
                    m_deqs[tid].append(_best_det(res, m_ids, mask_conf) or "unknown")

        if frame_idx <= WARMUP_F:
            continue

        for tid in track_ids:
            hvote = vote(h_deqs[tid])
            vvote = vote(v_deqs[tid])
            mvote = vote(m_deqs[tid])
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


# ── Yazdırma ─────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None: return "  -  "
    return f"{v:6.1f}"


def print_section(title: str, rows: list[dict], conf_key: str, conf_values: list[float]):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'conf':>6}  {'video':<20}  {'fps':>6}  {'H-know%':>8}  {'V-know%':>8}  {'M-know%':>8}  {'H-viol%':>8}  {'V-viol%':>8}  {'M-viol%':>8}"
    print(header)
    print("-" * len(header))
    prev_conf = None
    for row in rows:
        if prev_conf is not None and row[conf_key] != prev_conf:
            print()
        cells = (
            f"{row[conf_key]:>6.2f}  {row['video']:<20}  "
            f"{_fmt(row['fps'])}  "
            f"{_fmt(row['helmet_known_rate'])}  "
            f"{_fmt(row['vest_known_rate'])}  "
            f"{_fmt(row['mask_known_rate'])}  "
            f"{_fmt(row['helmet_violation_rate'])}  "
            f"{_fmt(row['vest_violation_rate'])}  "
            f"{_fmt(row['mask_violation_rate'])}"
        )
        print(cells)
        prev_conf = row[conf_key]
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = ap.parse_args()

    test_dir = ROOT / "test"
    test_videos = [v for stem in ["nohat_test", "novest_test", "noppe_test", "mask_test"]
                   for v in [test_dir / f"{stem}.mp4"] if v.exists()]
    if not test_videos:
        sys.exit(f"test/ klasöründe test videoları bulunamadı: {test_dir}")

    print("Modeller yükleniyor...")
    models = {
        "person": YOLO(str(PERSON_MODEL)),
        "helmet": YOLO(str(HELMET_MODEL)),
        "vest":   YOLO(str(VEST_MODEL)),
        "mask":   YOLO(str(MASK_MODEL)),
    }
    print(f"  Hazır. {len(test_videos)} video, PPE_INFER_EVERY={PPE_INFER_EVERY} (sabit)\n")

    all_rows: list[dict] = []

    # ── 1. Helmet conf taraması ──────────────────────────────────────────────
    print(">>> Helmet conf taraması başlıyor...")
    h_rows = []
    for hconf in HELMET_CONF_SWEEP:
        for vid in test_videos:
            print(f"  helmet_conf={hconf:.2f}  {vid.name}  ", end="", flush=True)
            m = run_one(vid, models, helmet_conf=hconf,
                        vest_conf=PROD_VEST_CONF, mask_conf=PROD_MASK_CONF,
                        max_frames=args.max_frames)
            print(f"→ fps={m.get('fps','?')}  H-known={m.get('helmet_known_rate')}%  H-viol={m.get('helmet_violation_rate')}%")
            row = {"sweep": "helmet", "helmet_conf": hconf,
                   "vest_conf": PROD_VEST_CONF, "mask_conf": PROD_MASK_CONF,
                   "video": vid.stem, **m}
            h_rows.append(row)
            all_rows.append(row)

    # ── 2. Vest conf taraması ────────────────────────────────────────────────
    print(">>> Vest conf taraması başlıyor...")
    v_rows = []
    for vconf in VEST_CONF_SWEEP:
        for vid in test_videos:
            print(f"  vest_conf={vconf:.2f}  {vid.name}  ", end="", flush=True)
            m = run_one(vid, models, helmet_conf=PROD_HELMET_CONF,
                        vest_conf=vconf, mask_conf=PROD_MASK_CONF,
                        max_frames=args.max_frames)
            print(f"→ fps={m.get('fps','?')}  V-known={m.get('vest_known_rate')}%  V-viol={m.get('vest_violation_rate')}%")
            row = {"sweep": "vest", "helmet_conf": PROD_HELMET_CONF,
                   "vest_conf": vconf, "mask_conf": PROD_MASK_CONF,
                   "video": vid.stem, **m}
            v_rows.append(row)
            all_rows.append(row)

    # ── 3. Mask conf taraması ────────────────────────────────────────────────
    print(">>> Mask conf taraması başlıyor...")
    m_rows = []
    for mconf in MASK_CONF_SWEEP:
        for vid in test_videos:
            print(f"  mask_conf={mconf:.2f}  {vid.name}  ", end="", flush=True)
            m = run_one(vid, models, helmet_conf=PROD_HELMET_CONF,
                        vest_conf=PROD_VEST_CONF, mask_conf=mconf,
                        max_frames=args.max_frames)
            print(f"→ fps={m.get('fps','?')}  M-known={m.get('mask_known_rate')}%  M-viol={m.get('mask_violation_rate')}%")
            row = {"sweep": "mask", "helmet_conf": PROD_HELMET_CONF,
                   "vest_conf": PROD_VEST_CONF, "mask_conf": mconf,
                   "video": vid.stem, **m}
            m_rows.append(row)
            all_rows.append(row)

    # ── Tablolar ─────────────────────────────────────────────────────────────
    print_section("HELMET CONF TARAMASI", h_rows, "helmet_conf", HELMET_CONF_SWEEP)
    print_section("VEST CONF TARAMASI",   v_rows, "vest_conf",   VEST_CONF_SWEEP)
    print_section("MASK CONF TARAMASI",   m_rows, "mask_conf",   MASK_CONF_SWEEP)

    # ── CSV kaydet ───────────────────────────────────────────────────────────
    out_dir = ROOT / "runs" / "benchmarks" / "conf"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir / f"conf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSonuçlar kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
