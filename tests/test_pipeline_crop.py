# -*- coding: utf-8 -*-
"""
tests/test_pipeline_crop.py — Crop modu pipeline testi
=======================================================
nohat_test.mp4, novest_test.mp4, noppe_test.mp4 üzerinde crop modunu çalıştırır.
Her video için beklenen ihlallerin tespit edilip edilmediğini kontrol eder.

Kullanım:
    python tests/test_pipeline_crop.py
    python tests/test_pipeline_crop.py --max-frames 200
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

_CROP_CFG   = _CFG.get("models", {}).get("crop", {})
_SCENE_CFG  = _CFG.get("models", {}).get("scene", {})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSON_MODEL_PATH = ROOT / _CFG["models"].get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")
HELMET_MODEL_PATH = ROOT / _CROP_CFG.get("helmet_model", "models/bera/crophelmet_agent_final_best.pt")
VEST_MODEL_PATH   = ROOT / _CROP_CFG.get("vest_model",   "models/bera/cropvest_agent_final_best.pt")
MASK_MODEL_PATH   = ROOT / _CROP_CFG.get("mask_model",   "models/bera/cropmask_agent_final_best.pt")

HELMET_CONF = 0.15
VEST_CONF   = 0.35
MASK_CONF   = 0.10
PERSON_CONF = 0.25
TEMPORAL_WIN = 20
PIPELINE_MAX_WIDTH = 1280
TRACKER = "config/bytetrack.yaml"
IMGSZ = 640
WARMUP_F = 30
PPE_INFER_EVERY = 4

GROUND_TRUTH = {
    "nohat_test":  {"helmet": "NO-Hardhat", "mask": "NO-Mask"},
    "novest_test": {"vest":   "NO-Safety Vest", "mask": "NO-Mask"},
    "noppe_test":  {"helmet": "NO-Hardhat", "vest": "NO-Safety Vest", "mask": "NO-Mask"},
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


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


def crop_region(frame, box, region: str):
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    if region == "head":
        return frame[y1:y1 + int(h * 0.40), x1:x2]
    elif region == "torso":
        return frame[y1 + int(h * 0.25):y1 + int(h * 0.75), x1:x2]
    return frame[y1:y2, x1:x2]


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

        do_ppe = (fi % PPE_INFER_EVERY == 0) and fi > WARMUP_F

        for box, tid in zip(boxes.xyxy, boxes.id):
            tid = int(tid)
            if do_ppe:
                hcrop = crop_region(frame, box.tolist(), "head")
                vcrop = crop_region(frame, box.tolist(), "torso")
                mcrop = crop_region(frame, box.tolist(), "head")

                def best(model, crop, ids, conf):
                    if crop.size == 0:
                        return "unknown"
                    res = model.predict(crop, imgsz=IMGSZ, conf=conf, device=DEVICE, verbose=False)[0]
                    if not res.boxes:
                        return "unknown"
                    best_box = max(res.boxes, key=lambda b: float(b.conf[0]))
                    cls = int(best_box.cls[0])
                    return model.names[cls] if cls in ids else "unknown"

                states[tid]["helmet"].append(best(hm, hcrop, h_ids, HELMET_CONF))
                states[tid]["vest"].append(best(vm, vcrop, v_ids, VEST_CONF))
                states[tid]["mask"].append(best(mm, mcrop, m_ids, MASK_CONF))

            if fi == WARMUP_F + 1 and t0 is None:
                t0 = time.perf_counter()

            if fi > WARMUP_F:
                hv = vote(states[tid]["helmet"])
                vv = vote(states[tid]["vest"])
                mv = vote(states[tid]["mask"])
                if hv == "NO-Hardhat":
                    violation_detected["helmet"] = True
                if vv == "NO-Safety Vest":
                    violation_detected["vest"] = True
                if mv == "NO-Mask":
                    violation_detected["mask"] = True

    elapsed = time.perf_counter() - (t0 or time.perf_counter())
    post_warmup_frames = max(fi - WARMUP_F, 1)
    cap.release()
    return {
        "fps": round(post_warmup_frames / elapsed, 1) if elapsed > 0 else 0.0,
        "frames": fi,
        "violation_detected": violation_detected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    test_dir = ROOT / "test"
    test_videos = [p for p in test_dir.glob("*.mp4") if p.stem in GROUND_TRUTH]
    if not test_videos:
        sys.exit(f"Ground truth videoları bulunamadı: {list(GROUND_TRUTH.keys())}")

    print("Modeller yükleniyor (crop modu)...")
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
            ok = detected
            sym = PASS if ok else FAIL
            print(f"  {sym}  {ppe} ihlali {'tespit edildi' if detected else 'TESPİT EDİLEMEDİ'}")
            if not ok:
                all_pass = False

    print(f"\n{'='*55}")
    print(f"  Crop modu testi: {'BAŞARILI' if all_pass else 'BAŞARISIZ — yukarıya bakın'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
