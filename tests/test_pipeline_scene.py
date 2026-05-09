# -*- coding: utf-8 -*-
"""
tests/test_pipeline_scene.py — Scene modu pipeline testi
=========================================================
nohat_test.mp4, novest_test.mp4, noppe_test.mp4 üzerinde scene modunu çalıştırır.
Her video için beklenen ihlallerin tespit edilip edilmediğini kontrol eder.

Kullanım:
    python tests/test_pipeline_scene.py
    python tests/test_pipeline_scene.py --max-frames 200
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque, defaultdict
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import yaml

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

_SCENE_CFG = _CFG.get("models", {}).get("scene", {})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSON_MODEL_PATH = ROOT / _CFG["models"].get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")
HELMET_MODEL_PATH = ROOT / _SCENE_CFG.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
VEST_MODEL_PATH   = ROOT / _SCENE_CFG.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
MASK_MODEL_PATH   = ROOT / _SCENE_CFG.get("mask_model",   "models/mask_agent_scene_200ep_yolov8m_best.pt")

HELMET_CONF = 0.25
VEST_CONF   = 0.30
MASK_CONF   = 0.05
PERSON_CONF = 0.25
INSIDE_FRAC_THR = 0.20
TEMPORAL_WIN = 30
PIPELINE_MAX_WIDTH = 1280
PPE_INFER_EVERY = 2
TRACKER = "bytetrack.yaml"
IMGSZ = 640
WARMUP_F = 30

GROUND_TRUTH = {
    "nohat_test":  {"helmet": "NO-Hardhat", "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest", "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat", "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def vote(q: deque, min_known: int = 2) -> str:
    if not q:
        return "unknown"
    top, cnt = Counter(q).most_common(1)[0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    tl, tc = Counter(known).most_common(1)[0]
    return tl if tc / len(known) >= 0.5 else "unknown"


def inside_frac(ppe_box, person_box) -> float:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area  = (bx2 - bx1) * (by2 - by1)
    return inter / area if area > 0 else 0.0


def scene_dets(model, frame, allowed_ids, conf):
    res = model.predict(frame, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
    if not res.boxes:
        return []
    return [(model.names[int(b.cls[0])], b.xyxy[0].tolist())
            for b in res.boxes if int(b.cls[0]) in allowed_ids]


def best_scene(dets, person_box):
    best = None
    for lbl, bbox in dets:
        if inside_frac(bbox, person_box) >= INSIDE_FRAC_THR:
            best = lbl
    return best or "unknown"


def run_video(video_path: Path, models: dict, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "video açılamadı"}

    pm = models["person"]
    hm = models["helmet"]
    vm = models["vest"]
    mm = models["mask"]

    p_ids = {i for i, n in pm.names.items() if n.lower() == "person"}
    h_ids = {i for i, n in hm.names.items() if n in ["Hardhat", "NO-Hardhat"]}
    v_ids = {i for i, n in vm.names.items() if n in ["Safety Vest", "NO-Safety Vest"]}
    m_ids = {i for i, n in mm.names.items() if n in ["Mask", "NO-Mask"]}

    states = defaultdict(lambda: {
        "helmet": deque(maxlen=TEMPORAL_WIN),
        "vest":   deque(maxlen=TEMPORAL_WIN),
        "mask":   deque(maxlen=TEMPORAL_WIN),
    })

    violation_detected = {"helmet": False, "vest": False, "mask": False}
    fi = 0
    t0 = None  # warmup sonrası başlar

    while fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1

        if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
            scale = PIPELINE_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * scale)))

        p_res = pm.track(frame, classes=list(p_ids), tracker=TRACKER, persist=True,
                         imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False)[0]
        boxes = p_res.boxes
        if boxes is None or boxes.id is None:
            continue

        do_ppe = (fi % PPE_INFER_EVERY == 0)
        if do_ppe:
            h_dets = scene_dets(hm, frame, h_ids, HELMET_CONF)
            v_dets = scene_dets(vm, frame, v_ids, VEST_CONF)
            m_dets = scene_dets(mm, frame, m_ids, MASK_CONF)
        else:
            h_dets = v_dets = m_dets = []

        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            pb = list(map(int, box.tolist()))
            if do_ppe:
                states[tid]["helmet"].append(best_scene(h_dets, pb))
                states[tid]["vest"].append(best_scene(v_dets, pb))
                states[tid]["mask"].append(best_scene(m_dets, pb))

            if fi == WARMUP_F + 1 and t0 is None:
                t0 = time.perf_counter()

            if fi > WARMUP_F:
                if vote(states[tid]["helmet"]) == "NO-Hardhat":
                    violation_detected["helmet"] = True
                if vote(states[tid]["vest"], min_known=2) == "NO-Safety Vest":
                    violation_detected["vest"] = True
                if vote(states[tid]["mask"], min_known=1) == "NO-Mask":
                    violation_detected["mask"] = True

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    post_warmup_frames = max(fi - WARMUP_F, 1)
    cap.release()
    return {"fps": round(post_warmup_frames / elapsed, 1) if elapsed > 0 else 0.0,
            "frames": fi, "violation_detected": violation_detected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    test_dir = ROOT / "test"
    test_videos = [p for p in test_dir.glob("*.mp4") if p.stem in GROUND_TRUTH]
    if not test_videos:
        sys.exit(f"Ground truth videoları bulunamadı: {list(GROUND_TRUTH.keys())}")

    print("Modeller yükleniyor (scene modu)...")
    models = {
        "person": YOLO(str(PERSON_MODEL_PATH)),
        "helmet": YOLO(str(HELMET_MODEL_PATH)),
        "vest":   YOLO(str(VEST_MODEL_PATH)),
        "mask":   YOLO(str(MASK_MODEL_PATH)),
    }
    print(f"  Hazır. Device={DEVICE}\n")

    all_pass = True
    for vp in sorted(test_videos):
        gt = GROUND_TRUTH[vp.stem]
        print(f"{'─'*55}")
        print(f"  Video : {vp.name}")
        print(f"  GT    : {gt}")
        m = run_video(vp, models, args.max_frames)
        if "error" in m:
            print(f"  {FAIL}  HATA: {m['error']}")
            all_pass = False
            continue
        print(f"  FPS   : {m['fps']}  ({m['frames']} frame)")
        det = m["violation_detected"]
        for ppe, expected_label in gt.items():
            detected = det.get(ppe, False)
            sym = PASS if detected else FAIL
            print(f"  {sym}  {ppe} ihlali {'tespit edildi' if detected else 'TESPİT EDİLEMEDİ'}")
            if not detected:
                all_pass = False

    print(f"\n{'='*55}")
    print(f"  Scene modu testi: {'BAŞARILI' if all_pass else 'BAŞARISIZ — yukarıya bakın'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
