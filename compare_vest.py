# -*- coding: utf-8 -*-
"""
compare_vest.py
===============
vest_agent_final_best.pt vs cropvest_agent_final_best.pt
Crop-based pipeline ile her iki modeli test videolarında karşılaştırır.

Ground truth (kişi başı yelek ihlali sayısı):
  novest  : 1 ihlal / 2 kişi
  noppe   : 4 ihlal / 4 kişi
  nohat   : 0 ihlal / 5 kişi
  intel   : 1 ihlal / 3 kişi
  karam   : 1 ihlal / 1 kişi
"""
from __future__ import annotations

import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2, torch
from collections import defaultdict, deque
from ultralytics import YOLO

# ---------------------------------------------------------------------------
PERSON_MODEL = "models/person_agent_scene_vinayakstyle_best.pt"
OLD_VEST     = "models/bera/vest_agent_final_best.pt"
NEW_VEST     = "models/bera/cropvest_agent_final_best.pt"

PERSON_CONF  = 0.25
VEST_CONF    = 0.30
VEST_PAD     = 0.60
IMGSZ        = 640
SAMPLE_EVERY = 3
TEMPORAL_WIN = 20
MIN_CROP_PX  = 40

VIDEOS = [
    ("novest", "test/novest_test.mp4",  {"n_main": 2, "gt_violations": 1}),
    ("noppe",  "test/noppe_test.mp4",   {"n_main": 4, "gt_violations": 4}),
    ("nohat",  "test/nohat_test.mp4",   {"n_main": 5, "gt_violations": 0}),
    ("intel",  "test/intel_safety_full.mp4", {"n_main": 3, "gt_violations": 1}),
    ("karam",  "test/github_hardhat.mp4",    {"n_main": 1, "gt_violations": 1}),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------

def class_ids(model, names):
    n2id = {n: cid for cid, n in model.names.items()}
    return [n2id[n] for n in names]

def crop_pad(frame, x1, y1, x2, y2, pad):
    fh, fw = frame.shape[:2]
    pw = (x2 - x1) * pad; ph = (y2 - y1) * pad
    cx1 = max(0, int(x1 - pw)); cy1 = max(0, int(y1 - ph))
    cx2 = min(fw, int(x2 + pw)); cy2 = min(fh, int(y2 + ph))
    return frame[cy1:cy2, cx1:cx2]

def run_video(video_path, person_model, vest_model, p_ids, v_ids):
    """Her track için temporal voting ile yelek tespiti yapar.
    Dönüş: {track_id: 'ok'|'violation'|'unknown'}"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    votes  = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SAMPLE_EVERY != 0:
            continue

        fh, fw = frame.shape[:2]
        res = person_model.track(
            frame, classes=p_ids, tracker="bytetrack.yaml",
            persist=True, imgsz=IMGSZ, conf=PERSON_CONF,
            device=DEVICE, verbose=False,
        )[0]

        if res.boxes is None or res.boxes.id is None:
            continue

        for box, tid in zip(res.boxes, res.boxes.id):
            track_id = int(tid)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = crop_pad(frame, x1, y1, x2, y2, VEST_PAD)
            if crop.shape[0] < MIN_CROP_PX or crop.shape[1] < MIN_CROP_PX:
                continue

            vres = vest_model.predict(
                crop, classes=v_ids, imgsz=IMGSZ,
                conf=VEST_CONF, device=DEVICE, verbose=False,
            )[0]

            label = None
            best_conf = 0.0
            if vres.boxes:
                for vb in vres.boxes:
                    c = float(vb.conf[0])
                    if c > best_conf:
                        best_conf = c
                        label = vest_model.names[int(vb.cls[0])]

            if label:
                votes[track_id].append(label)

    cap.release()

    results = {}
    for tid, v in votes.items():
        if not v:
            results[tid] = "unknown"
            continue
        ok_ct  = sum(1 for x in v if x == "Safety Vest")
        no_ct  = sum(1 for x in v if x == "NO-Safety Vest")
        if ok_ct + no_ct == 0:
            results[tid] = "unknown"
        elif no_ct > ok_ct:
            results[tid] = "violation"
        else:
            results[tid] = "ok"
    return results


def count_violations(results):
    return sum(1 for v in results.values() if v == "violation")


def main():
    print("Modeller yukleniyor...")
    person_model = YOLO(PERSON_MODEL).to(DEVICE)
    old_model    = YOLO(OLD_VEST).to(DEVICE)
    new_model    = YOLO(NEW_VEST).to(DEVICE)

    p_ids = class_ids(person_model,
                      [n for n in person_model.names.values() if n.lower() == "person"])
    v_ids = class_ids(old_model, ["Safety Vest", "NO-Safety Vest"])

    print(f"Device: {DEVICE}\n")

    hdr = f"{'Video':<8}  {'GT':>4}  {'Eski':>6}  {'Yeni':>6}  {'Fark':>6}  Sonuc"
    print(hdr)
    print("-" * len(hdr))

    old_correct = new_correct = 0

    for name, path, info in VIDEOS:
        if not os.path.exists(path):
            print(f"{name:<8}  (dosya yok: {path})")
            continue

        gt = info["gt_violations"]

        # Eski model
        old_res = run_video(path, person_model, old_model, p_ids, v_ids)
        old_viol = count_violations(old_res)

        # Yeni model — tracker'ı sıfırla
        person_model = YOLO(PERSON_MODEL).to(DEVICE)
        new_res = run_video(path, person_model, new_model, p_ids, v_ids)
        new_viol = count_violations(new_res)
        person_model = YOLO(PERSON_MODEL).to(DEVICE)  # sonraki video icin sifirla

        diff = new_viol - old_viol
        old_ok = abs(old_viol - gt) <= 1
        new_ok = abs(new_viol - gt) <= 1
        if old_ok: old_correct += 1
        if new_ok: new_correct += 1

        sonuc = ""
        if new_ok and not old_ok: sonuc = "<< YENI DAHA IYI"
        elif old_ok and not new_ok: sonuc = "!! YENI DAHA KOTU"
        elif new_ok and old_ok:    sonuc = "ikisi de dogru"

        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{name:<8}  {gt:>4}  {old_viol:>6}  {new_viol:>6}  {diff_str:>6}  {sonuc}")

    print("-" * len(hdr))
    print(f"{'Dogru':>20}  {old_correct:>6}  {new_correct:>6}  ({len(VIDEOS)} video)")


if __name__ == "__main__":
    main()
