# -*- coding: utf-8 -*-
"""
scripts/compare_modes.py
=========================
Crop-based ve scene-based modlari ayni test videolari uzerinde karsilastirir.
FPS, known_rate ve violation_rate yan yana raporlanir.

Kullanim:
    python scripts/compare_modes.py
    python scripts/compare_modes.py --max-frames 300
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import yaml

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

_CROP_CFG  = _CFG.get("models", {}).get("crop", {})
_SCENE_CFG = _CFG.get("models", {}).get("scene", {})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSON_MODEL   = ROOT / _CFG["models"].get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")
CROP_HELMET    = ROOT / _CROP_CFG.get("helmet_model",  "models/bera/crophelmet_agent_final_best.pt")
CROP_VEST      = ROOT / _CROP_CFG.get("vest_model",    "models/bera/cropvest_agent_final_best.pt")
CROP_MASK      = ROOT / _CROP_CFG.get("mask_model",    "models/bera/cropmask_agent_final_best.pt")
SCENE_HELMET   = ROOT / _SCENE_CFG.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
SCENE_VEST     = ROOT / _SCENE_CFG.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
SCENE_MASK     = ROOT / _SCENE_CFG.get("mask_model",   "models/mask_agent_scene_200ep_yolov8m_best.pt")

PIPELINE_MAX_WIDTH = 1280
TRACKER      = "bytetrack.yaml"
IMGSZ        = 640
WARMUP_F     = 30
PERSON_CONF  = 0.25

CROP_HELMET_CONF  = 0.20
CROP_VEST_CONF    = 0.30
CROP_MASK_CONF    = 0.25
CROP_PPE_EVERY    = 4
CROP_TEMPORAL     = 20

SCENE_HELMET_CONF = 0.25
SCENE_VEST_CONF   = 0.30
SCENE_MASK_CONF   = 0.05
SCENE_PPE_EVERY   = 2
SCENE_TEMPORAL    = 30
SCENE_FRAC_THR    = 0.20

GROUND_TRUTH = {
    "nohat_test":  {"helmet": "NO-Hardhat", "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest", "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat", "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}


def vote(q, min_known=2):
    if not q:
        return "unknown"
    top, _ = Counter(q).most_common(1)[0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    tl, tc = Counter(known).most_common(1)[0]
    return tl if tc / len(known) >= 0.5 else "unknown"


def inside_frac(ppe_box, person_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area  = (bx2 - bx1) * (by2 - by1)
    return inter / area if area > 0 else 0.0


def crop_region(frame, box, region):
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    if region == "head":
        return frame[y1:y1 + int(h * 0.40), x1:x2]
    if region == "torso":
        return frame[y1 + int(h * 0.25):y1 + int(h * 0.75), x1:x2]
    return frame[y1:y2, x1:x2]


def resize_frame(frame):
    if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
        s = PIPELINE_MAX_WIDTH / frame.shape[1]
        return cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * s)))
    return frame


def empty_counts():
    return {ppe: {"known": 0, "total": 0, "viol": 0} for ppe in ("helmet", "vest", "mask")}


def run_crop(video_path, models, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "video acilamadi"}

    pm = models["person"]
    hm = models["c_helmet"]
    vm = models["c_vest"]
    mm = models["c_mask"]
    p_ids = {i for i, n in pm.names.items() if n.lower() == "person"}
    h_ids = {i for i, n in hm.names.items() if n in ["Hardhat", "NO-Hardhat"]}
    v_ids = {i for i, n in vm.names.items() if n in ["Safety Vest", "NO-Safety Vest"]}
    m_ids = {i for i, n in mm.names.items() if n in ["Mask", "NO-Mask"]}

    states   = defaultdict(lambda: {k: deque(maxlen=CROP_TEMPORAL) for k in ("helmet", "vest", "mask")})
    counts   = empty_counts()
    viol_det = {"helmet": False, "vest": False, "mask": False}
    fi, t0   = 0, None

    while fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        frame = resize_frame(frame)

        p_res = pm.track(frame, classes=list(p_ids), tracker=TRACKER, persist=True,
                         imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False)[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        do_ppe = (fi % CROP_PPE_EVERY == 0) and fi > WARMUP_F

        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            if do_ppe:
                def best_crop(model, crop, ids, conf):
                    if crop.size == 0:
                        return "unknown"
                    res = model.predict(crop, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
                    if not res.boxes:
                        return "unknown"
                    b   = max(res.boxes, key=lambda b: float(b.conf[0]))
                    cls = int(b.cls[0])
                    return model.names[cls] if cls in ids else "unknown"

                states[tid]["helmet"].append(best_crop(hm, crop_region(frame, box.tolist(), "head"),  h_ids, CROP_HELMET_CONF))
                states[tid]["vest"].append(  best_crop(vm, crop_region(frame, box.tolist(), "torso"), v_ids, CROP_VEST_CONF))
                states[tid]["mask"].append(  best_crop(mm, crop_region(frame, box.tolist(), "head"),  m_ids, CROP_MASK_CONF))

            if fi == WARMUP_F + 1 and t0 is None:
                t0 = time.perf_counter()

            if fi > WARMUP_F:
                for ppe in ("helmet", "vest", "mask"):
                    v = vote(states[tid][ppe])
                    counts[ppe]["total"] += 1
                    if v != "unknown":
                        counts[ppe]["known"] += 1
                        if v.startswith("NO-"):
                            counts[ppe]["viol"] += 1
                            viol_det[ppe] = True

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    cap.release()
    return {"fps": round(max(fi - WARMUP_F, 1) / elapsed, 1) if elapsed > 0 else 0.0,
            "frames": fi, "viol_det": viol_det, "counts": counts}


def run_scene(video_path, models, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "video acilamadi"}

    pm = models["person"]
    hm = models["s_helmet"]
    vm = models["s_vest"]
    mm = models["s_mask"]
    p_ids = {i for i, n in pm.names.items() if n.lower() == "person"}
    h_ids = {i for i, n in hm.names.items() if n in ["Hardhat", "NO-Hardhat"]}
    v_ids = {i for i, n in vm.names.items() if n in ["Safety Vest", "NO-Safety Vest"]}
    m_ids = {i for i, n in mm.names.items() if n in ["Mask", "NO-Mask"]}

    states   = defaultdict(lambda: {k: deque(maxlen=SCENE_TEMPORAL) for k in ("helmet", "vest", "mask")})
    counts   = empty_counts()
    viol_det = {"helmet": False, "vest": False, "mask": False}
    fi, t0   = 0, None

    while fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        frame = resize_frame(frame)

        p_res = pm.track(frame, classes=list(p_ids), tracker=TRACKER, persist=True,
                         imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False)[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        do_ppe = (fi % SCENE_PPE_EVERY == 0)
        if do_ppe:
            def scene_dets(model, ids, conf):
                res = model.predict(frame, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
                if not res.boxes:
                    return []
                return [(model.names[int(b.cls[0])], b.xyxy[0].tolist())
                        for b in res.boxes if int(b.cls[0]) in ids]

            h_dets = scene_dets(hm, h_ids, SCENE_HELMET_CONF)
            v_dets = scene_dets(vm, v_ids, SCENE_VEST_CONF)
            m_dets = scene_dets(mm, m_ids, SCENE_MASK_CONF)
        else:
            h_dets = v_dets = m_dets = []

        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            pb  = list(map(int, box.tolist()))

            def best_scene(dets, pb):
                result = None
                for lbl, bbox in dets:
                    if inside_frac(bbox, pb) >= SCENE_FRAC_THR:
                        result = lbl
                return result or "unknown"

            if do_ppe:
                states[tid]["helmet"].append(best_scene(h_dets, pb))
                states[tid]["vest"].append(  best_scene(v_dets, pb))
                states[tid]["mask"].append(  best_scene(m_dets, pb))

            if fi == WARMUP_F + 1 and t0 is None:
                t0 = time.perf_counter()

            if fi > WARMUP_F:
                for ppe in ("helmet", "vest", "mask"):
                    v = vote(states[tid][ppe])
                    counts[ppe]["total"] += 1
                    if v != "unknown":
                        counts[ppe]["known"] += 1
                        if v.startswith("NO-"):
                            counts[ppe]["viol"] += 1
                            viol_det[ppe] = True

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    cap.release()
    return {"fps": round(max(fi - WARMUP_F, 1) / elapsed, 1) if elapsed > 0 else 0.0,
            "frames": fi, "viol_det": viol_det, "counts": counts}


def known_rate(counts, ppe):
    t = counts[ppe]["total"]
    return round(counts[ppe]["known"] / t * 100, 1) if t else 0.0


def viol_rate(counts, ppe):
    k = counts[ppe]["known"]
    return round(counts[ppe]["viol"] / k * 100, 1) if k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    test_dir    = ROOT / "test"
    test_videos = sorted([p for p in test_dir.glob("*.mp4") if p.stem in GROUND_TRUTH])
    if not test_videos:
        sys.exit("Test videolari bulunamadi.")

    print("Modeller yukleniyor...")
    models = {
        "person":   YOLO(str(PERSON_MODEL)),
        "c_helmet": YOLO(str(CROP_HELMET)),
        "c_vest":   YOLO(str(CROP_VEST)),
        "c_mask":   YOLO(str(CROP_MASK)),
        "s_helmet": YOLO(str(SCENE_HELMET)),
        "s_vest":   YOLO(str(SCENE_VEST)),
        "s_mask":   YOLO(str(SCENE_MASK)),
    }
    print(f"  Hazir. Device={DEVICE}\n")

    rows = []
    for vp in test_videos:
        gt = GROUND_TRUTH[vp.stem]
        print(f"{'='*62}")
        print(f"  {vp.name}   GT={list(gt.keys())}")
        print(f"{'─'*62}")

        print("  Crop modu calisiyor...")
        cr = run_crop(vp, models, args.max_frames)
        print("  Scene modu calisiyor...")
        sr = run_scene(vp, models, args.max_frames)

        if "error" in cr or "error" in sr:
            print(f"  HATA: {cr.get('error', '')} {sr.get('error', '')}")
            continue

        print(f"\n  {'Metrik':<28} {'CROP':>8} {'SCENE':>8}")
        print(f"  {'─'*46}")
        print(f"  {'FPS':<28} {cr['fps']:>8.1f} {sr['fps']:>8.1f}")

        for ppe in ("helmet", "vest", "mask"):
            ckr = known_rate(cr["counts"], ppe)
            skr = known_rate(sr["counts"], ppe)
            cvr = viol_rate(cr["counts"], ppe)
            svr = viol_rate(sr["counts"], ppe)
            cd  = cr["viol_det"].get(ppe, False)
            sd  = sr["viol_det"].get(ppe, False)

            if ppe in gt:
                sym_c = "[+]" if cd else "[-]"
                sym_s = "[+]" if sd else "[-]"
            else:
                sym_c = sym_s = " · "

            print(f"  {ppe + ' known_rate':<28} {ckr:>7.1f}% {skr:>7.1f}%")
            print(f"  {ppe + ' viol_rate':<28} {cvr:>7.1f}% {svr:>7.1f}%")
            print(f"  {ppe + ' ihlal tespit':<28} {sym_c:>8} {sym_s:>8}")

            rows.append({
                "video": vp.stem,
                "ppe": ppe,
                "in_gt": int(ppe in gt),
                "crop_fps": cr["fps"],    "scene_fps": sr["fps"],
                "crop_known_rate": ckr,   "scene_known_rate": skr,
                "crop_viol_rate": cvr,    "scene_viol_rate": svr,
                "crop_detected": int(cd), "scene_detected": int(sd),
            })
        print()

    if not rows:
        print("Sonuc yok.")
        return

    out_dir = ROOT / "runs" / "benchmarks" / "compare_modes"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"compare_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'='*62}")
    print(f"  CSV kaydedildi: {csv_path.relative_to(ROOT)}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
