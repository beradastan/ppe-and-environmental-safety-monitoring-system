# -*- coding: utf-8 -*-
"""
run_sweeps.py — 4 parametre taraması (basitleştirilmiş çıktı)
=============================================================
1. Kırpma PPE_INFER_EVERY taraması  → test_results/sweep_ppe_infer_every.csv
2. Kırpma temporal_window taraması  → test_results/sweep_crop_temporal.csv
3. Sahne INSIDE_FRAC_THR taraması   → test_results/sweep_inside_frac.csv
4. Sahne temporal_window taraması   → test_results/sweep_scene_temporal.csv

CSV sütunları: param_value, helmet_tanima, vest_tanima, mask_tanima, fps
Tanıma oranı = temporal voting'in "unknown" dışında karar ürettiği oran (%)
Metrikler tüm test videolarının ortalamasıdır.
"""
from __future__ import annotations

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

# ── Ortam ────────────────────────────────────────────────────────────────────
_DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
_HALF              = _DEVICE == "cuda"
PIPELINE_MAX_WIDTH = 1920
WARMUP_F           = 30

# ── Model yolları ─────────────────────────────────────────────────────────────
PERSON_MODEL = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"

CROP_HELMET = ROOT / "models/bera/crophelmet_agent_final_best.pt"
CROP_VEST   = ROOT / "models/bera/cropvest_agent_final_best.pt"
CROP_MASK   = ROOT / "models/bera/cropmask_agent_final_best.pt"

SCENE_HELMET = ROOT / "models/vinayak_trained_byBera/helmet_agent_final_best.pt"
SCENE_VEST   = ROOT / "models/vinayak_trained_byBera/vest_agent_final_best.pt"
SCENE_MASK   = ROOT / "models/mask_agent_scene_200ep_yolov8m_best.pt"

# ── Üretim sabitleri ──────────────────────────────────────────────────────────
PERSON_CONF     = 0.25
TRACKER         = "bytetrack.yaml"
IMGSZ           = 640
MIN_CROP_PX     = 40

# Kırpma modu (crop)
CROP_HELMET_CONF = 0.20
CROP_VEST_CONF   = 0.30
CROP_MASK_CONF   = 0.25
CROP_SKIP        = 4      # PPE_INFER_EVERY üretim değeri
CROP_WINDOW      = 20     # temporal_window üretim değeri
CROP_MIN_KNOWN   = 3

# Sahne modu (scene)
SCENE_HELMET_CONF = 0.25
SCENE_VEST_CONF   = 0.30
SCENE_MASK_CONF   = 0.05
INSIDE_FRAC_THR   = 0.20   # üretim değeri
SCENE_SKIP        = 2      # SCENE_PPE_INFER_EVERY üretim değeri
SCENE_WINDOW      = 30     # temporal_window üretim değeri
SCENE_H_MIN_KNOWN = 2
SCENE_V_MIN_KNOWN = 2
SCENE_M_MIN_KNOWN = 1

HELMET_CLASSES = ["Hardhat",     "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",        "NO-Mask"]

# ── Kullanılacak test videoları ───────────────────────────────────────────────
SWEEP_VIDEOS = [
    "nohat_test",
    "novest_test",
    "noppe_test",
    "mask_test",
]

MAX_FRAMES = 300

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _class_ids(model: YOLO, names: list[str]) -> list[int]:
    n2id = {n: i for i, n in model.names.items()}
    return [n2id[n] for n in names if n in n2id]


def _vote(q: deque, min_known: int) -> str:
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


def _best_det(result, allowed_ids: list[int], min_conf: float) -> str | None:
    if result.boxes is None:
        return None
    best_lbl, best_conf = None, -1.0
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid in allowed_ids and conf >= min_conf and conf > best_conf:
            best_lbl  = result.names[cid]
            best_conf = conf
    return best_lbl


def _crop_region(frame, x1, y1, x2, y2, ppe: str):
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
    crop = frame[cy1:cy2, cx1:cx2]
    h, w = crop.shape[:2]
    return crop if h >= MIN_CROP_PX and w >= MIN_CROP_PX else None


def _scene_dets(model: YOLO, frame, allowed_ids: list[int], min_conf: float) -> list[tuple]:
    res = model.predict(frame, imgsz=IMGSZ, conf=min_conf,
                        device=_DEVICE, half=_HALF, verbose=False)[0]
    if not res.boxes:
        return []
    return [
        (str(model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist())
        for b in res.boxes if int(b.cls[0]) in allowed_ids
    ]


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


def _best_scene(dets: list[tuple], person_box: list, frac_thr: float) -> str:
    best_lbl, best_conf = None, -1.0
    for lbl, c, bbox in dets:
        if _inside_frac(bbox, person_box) >= frac_thr and c > best_conf:
            best_lbl  = lbl
            best_conf = c
    return best_lbl if best_lbl else "unknown"


def _resize(frame):
    if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
        s = PIPELINE_MAX_WIDTH / frame.shape[1]
        return cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * s)))
    return frame


# ── Kırpma modu tek video ─────────────────────────────────────────────────────

def run_crop_video(video_path: Path, models: dict,
                   skip: int, window: int,
                   h_conf: float, v_conf: float, m_conf: float) -> dict:
    """
    Kırpma modunda tek video çalıştır.
    Dönüş: {h_known, h_total, v_known, v_total, m_known, m_total, frames, elapsed}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    person_model = models["person"]
    helmet_model = models["helmet"]
    vest_model   = models["vest"]
    mask_model   = models["mask"]

    p_ids = _class_ids(person_model, ["person", "Person"])
    if not p_ids:
        p_ids = [i for i, n in person_model.names.items() if n.lower() == "person"]
    h_ids = _class_ids(helmet_model, HELMET_CLASSES)
    v_ids = _class_ids(vest_model,   VEST_CLASSES)
    m_ids = _class_ids(mask_model,   MASK_CLASSES)

    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    h_known = h_total = v_known = v_total = m_known = m_total = 0
    frame_idx = 0
    t0 = None

    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = _resize(frame)
        _do_ppe = (frame_idx % skip == 0)

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_DEVICE, half=_HALF, verbose=False,
        )[0]

        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            if frame_idx > WARMUP_F and t0 is None:
                t0 = time.perf_counter()
            elif frame_idx > WARMUP_F and t0 is not None:
                pass
            continue

        track_ids = [int(t) for t in boxes.id]
        xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]

        if _do_ppe:
            h_batch, v_batch, m_batch = [], [], []
            for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
                hc = _crop_region(frame, x1, y1, x2, y2, "helmet")
                vc = _crop_region(frame, x1, y1, x2, y2, "vest")
                mc = _crop_region(frame, x1, y1, x2, y2, "mask")
                if hc is not None: h_batch.append((tid, hc))
                if vc is not None: v_batch.append((tid, vc))
                if mc is not None: m_batch.append((tid, mc))

            if h_batch:
                h_res = helmet_model.predict([b[1] for b in h_batch], imgsz=IMGSZ,
                                             conf=h_conf, device=_DEVICE, half=_HALF, verbose=False)
                for (tid, _), res in zip(h_batch, h_res):
                    h_deqs[tid].append(_best_det(res, h_ids, h_conf) or "unknown")

            if v_batch:
                v_res = vest_model.predict([b[1] for b in v_batch], imgsz=IMGSZ,
                                           conf=v_conf, device=_DEVICE, half=_HALF, verbose=False)
                for (tid, _), res in zip(v_batch, v_res):
                    v_deqs[tid].append(_best_det(res, v_ids, v_conf) or "unknown")

            if m_batch:
                m_res = mask_model.predict([b[1] for b in m_batch], imgsz=IMGSZ,
                                           conf=m_conf, device=_DEVICE, half=_HALF, verbose=False)
                for (tid, _), res in zip(m_batch, m_res):
                    m_deqs[tid].append(_best_det(res, m_ids, m_conf) or "unknown")

        if frame_idx <= WARMUP_F:
            continue
        if t0 is None:
            t0 = time.perf_counter()

        for tid in track_ids:
            if _vote(h_deqs[tid], CROP_MIN_KNOWN) != "unknown":
                h_known += 1
            if _vote(v_deqs[tid], CROP_MIN_KNOWN) != "unknown":
                v_known += 1
            if _vote(m_deqs[tid], CROP_MIN_KNOWN) != "unknown":
                m_known += 1
            h_total += 1
            v_total += 1
            m_total += 1

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    cap.release()
    measured = max(frame_idx - WARMUP_F, 1)
    return {
        "h_known": h_known, "h_total": h_total,
        "v_known": v_known, "v_total": v_total,
        "m_known": m_known, "m_total": m_total,
        "fps": measured / elapsed if elapsed > 0 else 0.0,
    }


# ── Sahne modu tek video ──────────────────────────────────────────────────────

def run_scene_video(video_path: Path, models: dict,
                    skip: int, window: int, frac_thr: float,
                    h_conf: float, v_conf: float, m_conf: float) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    person_model = models["person"]
    helmet_model = models["helmet"]
    vest_model   = models["vest"]
    mask_model   = models["mask"]

    p_ids = [i for i, n in person_model.names.items() if n.lower() == "person"]
    h_ids = _class_ids(helmet_model, HELMET_CLASSES)
    v_ids = _class_ids(vest_model,   VEST_CLASSES)
    m_ids = _class_ids(mask_model,   MASK_CLASSES)

    h_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    v_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
    m_deqs: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    h_known = h_total = v_known = v_total = m_known = m_total = 0
    frame_idx = 0
    t0 = None

    h_dets_cache: list[tuple] = []
    v_dets_cache: list[tuple] = []
    m_dets_cache: list[tuple] = []

    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = _resize(frame)
        _do_ppe = (frame_idx % skip == 0)

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=_DEVICE, half=_HALF, verbose=False,
        )[0]

        if _do_ppe:
            h_dets_cache = _scene_dets(helmet_model, frame, h_ids, h_conf)
            v_dets_cache = _scene_dets(vest_model,   frame, v_ids, v_conf)
            m_dets_cache = _scene_dets(mask_model,   frame, m_ids, m_conf)

        boxes = p_res.boxes
        if boxes is not None and boxes.id is not None:
            track_ids = [int(t) for t in boxes.id]
            xyxys     = [list(map(int, b.tolist())) for b in boxes.xyxy]
            for tid, (x1, y1, x2, y2) in zip(track_ids, xyxys):
                pb = [x1, y1, x2, y2]
                h_deqs[tid].append(_best_scene(h_dets_cache, pb, frac_thr))
                v_deqs[tid].append(_best_scene(v_dets_cache, pb, frac_thr))
                m_deqs[tid].append(_best_scene(m_dets_cache, pb, frac_thr))
        else:
            track_ids = []

        if frame_idx <= WARMUP_F:
            continue
        if t0 is None:
            t0 = time.perf_counter()

        for tid in track_ids:
            if _vote(h_deqs[tid], SCENE_H_MIN_KNOWN) != "unknown":
                h_known += 1
            if _vote(v_deqs[tid], SCENE_V_MIN_KNOWN) != "unknown":
                v_known += 1
            if _vote(m_deqs[tid], SCENE_M_MIN_KNOWN) != "unknown":
                m_known += 1
            h_total += 1
            v_total += 1
            m_total += 1

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    cap.release()
    measured = max(frame_idx - WARMUP_F, 1)
    return {
        "h_known": h_known, "h_total": h_total,
        "v_known": v_known, "v_total": v_total,
        "m_known": m_known, "m_total": m_total,
        "fps": measured / elapsed if elapsed > 0 else 0.0,
    }


# ── Sonuç birleştirme ─────────────────────────────────────────────────────────

def _aggregate(results: list[dict]) -> dict:
    """Tüm video sonuçlarını topla, oran ve ortalama FPS hesapla."""
    H_known = H_total = V_known = V_total = M_known = M_total = 0
    fps_sum = 0.0
    n = 0
    for r in results:
        if not r:
            continue
        H_known += r["h_known"]; H_total += r["h_total"]
        V_known += r["v_known"]; V_total += r["v_total"]
        M_known += r["m_known"]; M_total += r["m_total"]
        fps_sum += r["fps"]
        n += 1
    def pct(a, b): return round(100.0 * a / b, 1) if b > 0 else 0.0
    return {
        "helmet_tanima": pct(H_known, H_total),
        "vest_tanima":   pct(V_known, V_total),
        "mask_tanima":   pct(M_known, M_total),
        "fps":           round(fps_sum / n, 1) if n > 0 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["param_value", "helmet_tanima", "vest_tanima", "mask_tanima", "fps"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


def _load_models(paths: dict[str, Path], compile_person: bool = True) -> dict:
    models = {k: YOLO(str(p)) for k, p in paths.items()}
    if _DEVICE == "cuda" and compile_person:
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            models["person"].model = torch.compile(models["person"].model, mode="reduce-overhead")
            print("  torch.compile(person) etkin")
        except Exception as e:
            print(f"  torch.compile atlandı: {e}")
    return models


def _find_videos(test_dir: Path, stems: list[str]) -> list[Path]:
    found = []
    for stem in stems:
        p = test_dir / f"{stem}.mp4"
        if p.exists():
            found.append(p)
        else:
            print(f"  [uyarı] video bulunamadı: {p}")
    return found


# ── Tarama 1: Kırpma PPE_INFER_EVERY ─────────────────────────────────────────

def sweep_ppe_infer_every(models: dict, videos: list[Path]) -> list[dict]:
    skip_values = [1, 2, 3, 4, 6, 8, 10]
    rows = []
    for skip in skip_values:
        print(f"  skip={skip}", end="", flush=True)
        results = [
            run_crop_video(v, models, skip=skip, window=CROP_WINDOW,
                           h_conf=CROP_HELMET_CONF, v_conf=CROP_VEST_CONF, m_conf=CROP_MASK_CONF)
            for v in videos
        ]
        agg = _aggregate(results)
        row = {"param_value": skip, **agg}
        rows.append(row)
        print(f"  fps={agg['fps']}  H={agg['helmet_tanima']}%  V={agg['vest_tanima']}%  M={agg['mask_tanima']}%")
    return rows


# ── Tarama 2: Kırpma temporal_window ─────────────────────────────────────────

def sweep_crop_temporal(models: dict, videos: list[Path]) -> list[dict]:
    windows = [5, 10, 15, 20, 30, 50]
    rows = []
    for win in windows:
        print(f"  window={win}", end="", flush=True)
        results = [
            run_crop_video(v, models, skip=CROP_SKIP, window=win,
                           h_conf=CROP_HELMET_CONF, v_conf=CROP_VEST_CONF, m_conf=CROP_MASK_CONF)
            for v in videos
        ]
        agg = _aggregate(results)
        row = {"param_value": win, **agg}
        rows.append(row)
        print(f"  fps={agg['fps']}  H={agg['helmet_tanima']}%  V={agg['vest_tanima']}%  M={agg['mask_tanima']}%")
    return rows


# ── Tarama 3: Sahne INSIDE_FRAC_THR ──────────────────────────────────────────

def sweep_inside_frac(models: dict, videos: list[Path]) -> list[dict]:
    fracs = [0.10, 0.20, 0.30, 0.40, 0.60]
    rows = []
    for frac in fracs:
        print(f"  frac={frac:.2f}", end="", flush=True)
        results = [
            run_scene_video(v, models, skip=SCENE_SKIP, window=SCENE_WINDOW,
                            frac_thr=frac,
                            h_conf=SCENE_HELMET_CONF, v_conf=SCENE_VEST_CONF, m_conf=SCENE_MASK_CONF)
            for v in videos
        ]
        agg = _aggregate(results)
        row = {"param_value": frac, **agg}
        rows.append(row)
        print(f"  fps={agg['fps']}  H={agg['helmet_tanima']}%  V={agg['vest_tanima']}%  M={agg['mask_tanima']}%")
    return rows


# ── Tarama 4: Sahne temporal_window ──────────────────────────────────────────

def sweep_scene_temporal(models: dict, videos: list[Path]) -> list[dict]:
    windows = [5, 10, 20, 30, 50]
    rows = []
    for win in windows:
        print(f"  window={win}", end="", flush=True)
        results = [
            run_scene_video(v, models, skip=SCENE_SKIP, window=win,
                            frac_thr=INSIDE_FRAC_THR,
                            h_conf=SCENE_HELMET_CONF, v_conf=SCENE_VEST_CONF, m_conf=SCENE_MASK_CONF)
            for v in videos
        ]
        agg = _aggregate(results)
        row = {"param_value": win, **agg}
        rows.append(row)
        print(f"  fps={agg['fps']}  H={agg['helmet_tanima']}%  V={agg['vest_tanima']}%  M={agg['mask_tanima']}%")
    return rows


# ── Ana akış ──────────────────────────────────────────────────────────────────

def main():
    test_dir  = ROOT / "test"
    out_dir   = ROOT / "test_results"
    videos    = _find_videos(test_dir, SWEEP_VIDEOS)

    if not videos:
        sys.exit("Hiçbir test videosu bulunamadı.")

    print(f"\n{'='*60}")
    print(f"Tarama videoları ({len(videos)}): {[v.stem for v in videos]}")
    print(f"Max frame / run: {MAX_FRAMES}")
    print(f"{'='*60}\n")

    # ── Kırpma modeli yükleme ─────────────────────────────────────────────────
    print("Kırpma modelleri yükleniyor...")
    crop_models = _load_models({
        "person": PERSON_MODEL,
        "helmet": CROP_HELMET,
        "vest":   CROP_VEST,
        "mask":   CROP_MASK,
    })

    print("\n[1/4] PPE_INFER_EVERY taraması (kırpma modu)")
    rows1 = sweep_ppe_infer_every(crop_models, videos)
    _write_csv(out_dir / "sweep_ppe_infer_every.csv", rows1)

    print("\n[2/4] Crop temporal_window taraması (kırpma modu)")
    rows2 = sweep_crop_temporal(crop_models, videos)
    _write_csv(out_dir / "sweep_crop_temporal.csv", rows2)

    # Kırpma modellerini serbest bırak
    del crop_models
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── Sahne modeli yükleme ──────────────────────────────────────────────────
    print("\nSahne modelleri yükleniyor...")
    scene_models = _load_models({
        "person": PERSON_MODEL,
        "helmet": SCENE_HELMET,
        "vest":   SCENE_VEST,
        "mask":   SCENE_MASK,
    })

    print("\n[3/4] INSIDE_FRAC_THR taraması (sahne modu)")
    rows3 = sweep_inside_frac(scene_models, videos)
    _write_csv(out_dir / "sweep_inside_frac.csv", rows3)

    print("\n[4/4] Scene temporal_window taraması (sahne modu)")
    rows4 = sweep_scene_temporal(scene_models, videos)
    _write_csv(out_dir / "sweep_scene_temporal.csv", rows4)

    del scene_models
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Tüm taramalar tamamlandı.")
    print(f"CSV'ler: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
