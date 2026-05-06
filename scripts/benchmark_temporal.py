# -*- coding: utf-8 -*-
"""
benchmark_temporal.py — temporal_window (voting penceresi) optimizasyonu
=========================================================================
Farklı temporal window boyutlarında karar kalitesini ölçer.
PPE_INFER_EVERY=4 ve üretim conf değerleri sabit tutulur.

Metrikler:
  known_rate     — "unknown" olmayan temporal oy oranı (%)
  violation_rate — ground truth ihlalinin doğru tespiti (%)
  first_known_f  — ilk kararlı kararın geldiği ortalama frame numarası

Kullanım:
    python scripts/benchmark_temporal.py
    python scripts/benchmark_temporal.py --max-frames 300
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
HELMET_MODEL = ROOT / "models/bera/crophelmet_agent_final_best.pt"
VEST_MODEL   = ROOT / "models/bera/cropvest_agent_final_best.pt"
MASK_MODEL   = ROOT / "models/bera/cropmask_agent_final_best.pt"

PERSON_CONF    = 0.25
HELMET_CONF    = 0.20
VEST_CONF      = 0.30
MASK_CONF      = 0.25
IMGSZ          = 640
MASK_IMGSZ     = 640
TRACKER        = "bytetrack.yaml"
MIN_KNOWN      = 3
WARMUP_F       = 30
MIN_CROP_PX    = 40
PPE_INFER_EVERY = 4

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


def _class_ids(model, names):
    n2id = {n: i for i, n in model.names.items()}
    return [n2id[n] for n in names if n in n2id]

def _crop_ok(crop):
    if crop is None or crop.size == 0: return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX

def crop_ppe(frame, x1, y1, x2, y2, ppe):
    fh, fw = frame.shape[:2]
    pw, ph = x2-x1, y2-y1
    if ppe == "helmet":
        return frame[max(0,y1-int(ph*.15)):min(fh,y1+int(ph*.40)),
                     max(0,x1-int(pw*.10)):min(fw,x2+int(pw*.10))]
    elif ppe == "vest":
        return frame[max(0,y1+int(ph*.10)):min(fh,y1+int(ph*.90)),
                     max(0,x1-int(pw*.15)):min(fw,x2+int(pw*.15))]
    else:
        return frame[max(0,y1-int(ph*.10)):min(fh,y1+int(ph*.45)),
                     max(0,x1-int(pw*.15)):min(fw,x2+int(pw*.15))]

def vote(q: deque, min_known: int) -> str:
    if not q: return "unknown"
    top = Counter(q).most_common(1)[0][0]
    if top != "unknown": return top
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known: return "unknown"
    top_l, top_c = Counter(known).most_common(1)[0]
    return top_l if top_c / len(known) >= 0.5 else "unknown"

def _best_det(result, allowed_ids, min_conf):
    if result.boxes is None: return None
    best_label, best_conf = None, -1.0
    for box in result.boxes:
        cid = int(box.cls[0]); conf = float(box.conf[0])
        if cid in allowed_ids and conf >= min_conf and conf > best_conf:
            best_label = result.names[cid]; best_conf = conf
    return best_label


def run_one(video_path: Path, models: dict, window: int, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return {}

    p_model = models["person"]; h_model = models["helmet"]
    v_model = models["vest"];   m_model = models["mask"]

    _pcls = next(n for n in p_model.names.values() if n.lower() == "person")
    p_ids = _class_ids(p_model, [_pcls])
    h_ids = _class_ids(h_model, HELMET_CLASSES)
    v_ids = _class_ids(v_model, VEST_CLASSES)
    m_ids = _class_ids(m_model, MASK_CLASSES)

    h_deqs = defaultdict(lambda: deque(maxlen=window))
    v_deqs = defaultdict(lambda: deque(maxlen=window))
    m_deqs = defaultdict(lambda: deque(maxlen=window))

    h_known=h_total=v_known=v_total=m_known=m_total=0
    h_viol=v_viol=m_viol=0
    h_first: dict[int,int] = {}
    v_first: dict[int,int] = {}
    m_first: dict[int,int] = {}

    gt = GROUND_TRUTH.get(video_path.stem, {})
    h_exp = gt.get("helmet"); v_exp = gt.get("vest"); m_exp = gt.get("mask")

    frame_idx = 0
    t0 = time.perf_counter()

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        _do_ppe = (frame_idx % PPE_INFER_EVERY == 0)

        p_res = p_model.track(frame, classes=p_ids, tracker=TRACKER, persist=True,
                              imgsz=IMGSZ, conf=PERSON_CONF, device=_AUTO_DEVICE, verbose=False)[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None: continue
        track_ids = [int(t) for t in boxes.id]
        xyxys = [list(map(int, b.tolist())) for b in boxes.xyxy]

        if _do_ppe:
            h_batch, v_batch, m_batch = [], [], []
            for tid, (x1,y1,x2,y2) in zip(track_ids, xyxys):
                hc=crop_ppe(frame,x1,y1,x2,y2,"helmet")
                vc=crop_ppe(frame,x1,y1,x2,y2,"vest")
                mc=crop_ppe(frame,x1,y1,x2,y2,"mask")
                if _crop_ok(hc): h_batch.append((tid,hc))
                if _crop_ok(vc): v_batch.append((tid,vc))
                if _crop_ok(mc): m_batch.append((tid,mc))

            if h_batch:
                for (tid,_),res in zip(h_batch, h_model.predict([b[1] for b in h_batch],
                        imgsz=IMGSZ, conf=HELMET_CONF, device=_AUTO_DEVICE, verbose=False)):
                    h_deqs[tid].append(_best_det(res, h_ids, HELMET_CONF) or "unknown")
            if v_batch:
                for (tid,_),res in zip(v_batch, v_model.predict([b[1] for b in v_batch],
                        imgsz=IMGSZ, conf=VEST_CONF, device=_AUTO_DEVICE, verbose=False)):
                    v_deqs[tid].append(_best_det(res, v_ids, VEST_CONF) or "unknown")
            if m_batch:
                for (tid,_),res in zip(m_batch, m_model.predict([b[1] for b in m_batch],
                        imgsz=MASK_IMGSZ, conf=MASK_CONF, device=_AUTO_DEVICE, verbose=False)):
                    m_deqs[tid].append(_best_det(res, m_ids, MASK_CONF) or "unknown")

        if frame_idx <= WARMUP_F: continue

        for tid in track_ids:
            hv=vote(h_deqs[tid], MIN_KNOWN)
            vv=vote(v_deqs[tid], MIN_KNOWN)
            mv=vote(m_deqs[tid], MIN_KNOWN)
            h_total+=1; v_total+=1; m_total+=1
            if hv!="unknown":
                h_known+=1
                if h_exp and hv==h_exp: h_viol+=1
                if tid not in h_first: h_first[tid]=frame_idx
            if vv!="unknown":
                v_known+=1
                if v_exp and vv==v_exp: v_viol+=1
                if tid not in v_first: v_first[tid]=frame_idx
            if mv!="unknown":
                m_known+=1
                if m_exp and mv==m_exp: m_viol+=1
                if tid not in m_first: m_first[tid]=frame_idx

    elapsed = time.perf_counter()-t0
    cap.release()

    def _r(n,d): return round(100.*n/d,1) if d>0 else 0.
    def _first(d): return round(sum(d.values())/len(d),1) if d else None

    return {
        "fps":                   round(frame_idx/elapsed,1) if elapsed>0 else 0.,
        "helmet_known_rate":     _r(h_known,h_total),
        "vest_known_rate":       _r(v_known,v_total),
        "mask_known_rate":       _r(m_known,m_total),
        "helmet_violation_rate": _r(h_viol,h_known) if h_exp else None,
        "vest_violation_rate":   _r(v_viol,v_known) if v_exp else None,
        "mask_violation_rate":   _r(m_viol,m_known) if m_exp else None,
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
    test_videos = [v for stem in ["nohat_test","novest_test","noppe_test","mask_test"]
                   for v in [test_dir/f"{stem}.mp4"] if v.exists()]
    if not test_videos:
        sys.exit(f"Test videoları bulunamadı: {test_dir}")

    print("Modeller yükleniyor...")
    models = {"person": YOLO(str(PERSON_MODEL)), "helmet": YOLO(str(HELMET_MODEL)),
              "vest": YOLO(str(VEST_MODEL)), "mask": YOLO(str(MASK_MODEL))}
    print(f"  Hazır. {len(test_videos)} video, window değerleri: {args.window_values}\n")

    all_rows = []
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

    # Özet tablo
    hdr = f"{'win':>4}  {'video':<20}  {'fps':>6}  {'H-know%':>8}  {'V-know%':>8}  {'M-know%':>8}  {'H-viol%':>8}  {'V-viol%':>8}  {'M-viol%':>8}  {'H-1st':>6}  {'V-1st':>6}  {'M-1st':>6}"
    print(f"\n{hdr}")
    print("-"*len(hdr))
    prev_win = None
    for row in all_rows:
        if prev_win is not None and row["window"] != prev_win: print()
        def f(v): return f"{v:6.1f}" if v is not None and not isinstance(v,str) else "   -  "
        print(f"{row['window']:>4}  {row['video']:<20}  "
              f"{f(row['fps'])}  {f(row['helmet_known_rate'])}  {f(row['vest_known_rate'])}  "
              f"{f(row['mask_known_rate'])}  {f(row['helmet_violation_rate'])}  "
              f"{f(row['vest_violation_rate'])}  {f(row['mask_violation_rate'])}  "
              f"{f(row['helmet_first_known'])}  {f(row['vest_first_known'])}  {f(row['mask_first_known'])}")
        prev_win = row["window"]

    out_dir = ROOT/"runs"/"benchmarks"/"temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    csv_path = out_dir/f"temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSonuçlar kaydedildi: {csv_path}")

if __name__ == "__main__":
    main()
