# -*- coding: utf-8 -*-
"""
benchmark_scene_conf.py — Scene modu confidence eşiği optimizasyonu
====================================================================
Her PPE türü için confidence eşiği bağımsız olarak taranır;
diğer iki PPE türünün conf değeri üretim değerinde sabit tutulur.
Scene-tabanlı algılama: tam kareye YOLO predict + inside_frac eşleştirme.
PPE_INFER_EVERY yok — scene modu her frame'de çalışır.

Metrikler:
  known_rate     — "unknown" olmayan temporal oy oranı (%)  → yüksek = iyi
  violation_rate — ground truth ihlalinin doğru kararla örtüşme oranı (%)
  fps            — işlem hızı

Kullanım:
    python scripts/benchmark_scene_conf.py
    python scripts/benchmark_scene_conf.py --max-frames 200
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

PERSON_CONF    = 0.25
IMGSZ          = 640
TRACKER        = "bytetrack.yaml"
TEMPORAL_WIN   = 20
INSIDE_FRAC_THR      = 0.20  # frac benchmark sonucu: optimal değer
SCENE_PPE_INFER_EVERY = 2    # her 2 frame'de bir PPE inference → ~30fps
PIPELINE_MAX_WIDTH    = 1280  # yüksek çözünürlüklü videoları yeniden boyutlandır
WARMUP_F       = 30
H_MIN_KNOWN    = 2
V_MIN_KNOWN    = 2
M_MIN_KNOWN    = 1

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

        if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
            scale = PIPELINE_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * scale)))

        p_res = person_model.track(
            frame, classes=list(p_ids), tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_AUTO_DEVICE, verbose=False,
        )[0]

        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        # Scene: tam kare PPE tespiti (her frame)
        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        _do_ppe = (frame_idx % SCENE_PPE_INFER_EVERY == 0)
        if _do_ppe:
            h_dets = _scene_dets(helmet_model, frame, h_ids, helmet_conf)
            v_dets = _scene_dets(vest_model,   frame, v_ids, vest_conf)
            m_dets = _scene_dets(mask_model,   frame, m_ids, mask_conf)
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


# ── Yazdırma ─────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None: return "  -  "
    return f"{v:6.1f}"


def print_section(title: str, rows: list[dict], conf_key: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = (f"{'conf':>6}  {'video':<20}  {'fps':>6}  {'H-know%':>8}  "
              f"{'V-know%':>8}  {'M-know%':>8}  {'H-viol%':>8}  {'V-viol%':>8}  {'M-viol%':>8}")
    print(header)
    print("-" * len(header))
    prev_conf = None
    for row in rows:
        if prev_conf is not None and row[conf_key] != prev_conf:
            print()
        print(f"{row[conf_key]:>6.2f}  {row['video']:<20}  "
              f"{_fmt(row['fps'])}  "
              f"{_fmt(row['helmet_known_rate'])}  "
              f"{_fmt(row['vest_known_rate'])}  "
              f"{_fmt(row['mask_known_rate'])}  "
              f"{_fmt(row['helmet_violation_rate'])}  "
              f"{_fmt(row['vest_violation_rate'])}  "
              f"{_fmt(row['mask_violation_rate'])}")
        prev_conf = row[conf_key]
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
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
    print(f"  Hazır. {len(test_videos)} video, INSIDE_FRAC_THR={INSIDE_FRAC_THR} (sabit)\n")

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
    print_section("HELMET CONF TARAMASI (SCENE)", h_rows, "helmet_conf")
    print_section("VEST CONF TARAMASI (SCENE)",   v_rows, "vest_conf")
    print_section("MASK CONF TARAMASI (SCENE)",   m_rows, "mask_conf")

    # ── CSV kaydet ───────────────────────────────────────────────────────────
    out_dir = ROOT / "runs" / "benchmarks" / "scene_conf"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir / f"scene_conf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSonuçlar kaydedildi: {csv_path}")


if __name__ == "__main__":
    main()
