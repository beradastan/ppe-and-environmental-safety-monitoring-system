# -*- coding: utf-8 -*-
"""
compare_crop_vs_scene.py
========================
Crop-based PPE pipeline karsilastirmasi:
  - CROP      : klasik crop → doğrudan atama
  - CROP+ASSOC: crop → geometrik doğrulama → atama

Her test videosu için her iki pipeline çalışır; temporal vote sonuçları
ground truth ile karsilastirilir.

Kullanim:
    python scripts/compare_crop_vs_scene.py
"""
from __future__ import annotations

import sys
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model yolları ve eşikler
# ---------------------------------------------------------------------------
HELMET_MODEL_PATH = "models/bera/crophelmet_agent_final_best.pt"
VEST_MODEL_PATH   = "models/bera/vest_agent_final_best.pt"
MASK_MODEL_PATH   = "models/bera/cropmask_agent_final_best.pt"
PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"

HELMET_CONF  = 0.15
VEST_CONF    = 0.30
MASK_CONF    = 0.10
MASK_IMGSZ   = 1280
PERSON_CONF  = 0.25
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"
TEMPORAL_WIN     = 20
MIN_CROP_PX      = 40
MIN_TRACK_FRAMES = 10

# PPE tipine göre geometrik eşikler (run_live_video.py ile senkron)
PPE_MIN_SELF_SCORE: dict[str, float] = {
    "helmet": 0.20,
    "vest":   0.20,
    "mask":   0.15,
}
PPE_MAX_NEIGHBOR_RATIO: dict[str, float] = {
    "helmet": 0.80,
    "vest":   0.75,
    "mask":   0.90,
}
MIN_HEAD_PX  = 30
MIN_TORSO_PX = 50

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask", "NO-Mask"]

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "novest_test": {
        "person_count": 2,
        "expected_violations": {"no_mask": 2, "no_vest": 1},
        "notes": "2 kisi: ikisi de maske takmıyor, biri yelek takmıyor",
    },
    "nohat_test": {
        "person_count": None,
        "expected_violations": {"no_mask": "all", "no_helmet": 2},
        "notes": "hepsi maske takmıyor, 2 kişi kask takmıyor",
    },
    "noppe_test": {
        "person_count": 4,
        "expected_violations": {"no_helmet": 4, "no_vest": 4, "no_mask": 4},
        "notes": "4 kisi: hiçbiri hiçbir PPE takmıyor",
    },
    "mask_test": {
        "person_count": 1,
        "expected_violations": {"no_helmet": 1},
        "notes": "1 kisi: yelek+maske var, kask yok",
    },
}

# ---------------------------------------------------------------------------
# Paylaşılan yardımcılar
# ---------------------------------------------------------------------------

def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    return [name_to_id[n] for n in names if n in name_to_id]


def vote(q: deque, min_known: int = 3, ratio_threshold: float = 0.5) -> str:
    if not q:
        return "unknown"
    result = Counter(q).most_common(1)[0][0]
    if result != "unknown":
        return result
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    top_label, top_count = Counter(known).most_common(1)[0]
    if top_count / len(known) >= ratio_threshold:
        return top_label
    return "unknown"


def violations_from_votes(hvote, vvote, mvote) -> list[str]:
    viols = []
    if hvote == "NO-Hardhat":     viols.append("no_helmet")
    if vvote == "NO-Safety Vest": viols.append("no_vest")
    if mvote == "NO-Mask":        viols.append("no_mask")
    return viols


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
    else:
        cx1 = max(0, x1 - int(pw * 0.15))
        cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.45))
    return frame[cy1:cy2, cx1:cx2], cx1, cy1


def _crop_ok(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX


def crop_to_frame(bbox, ox, oy, fh, fw):
    dx1, dy1, dx2, dy2 = bbox
    return [min(fw-1, int(ox+dx1)), min(fh-1, int(oy+dy1)),
            min(fw-1, int(ox+dx2)), min(fh-1, int(oy+dy2))]


def best_det_label(model, result, allowed_ids, min_conf) -> tuple[str, float]:
    best_label, best_conf = "unknown", 0.0
    if result.boxes is None:
        return best_label, best_conf
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        if conf > best_conf:
            best_label = str(model.names[cid])
            best_conf  = conf
    return best_label, best_conf


def _new_states():
    return defaultdict(lambda: {
        "hardhat":     deque(maxlen=TEMPORAL_WIN),
        "vest":        deque(maxlen=TEMPORAL_WIN),
        "mask":        deque(maxlen=TEMPORAL_WIN),
        "frame_count": 0,
    })


def _finalize(states, id_map, id_next,
              decision_counts=None) -> tuple[dict[int, dict], dict]:
    """
    Döner: (results, metrics)
    metrics: raw_track_count, mature_track_count, unknown_rate, decision_counts
    """
    def did(raw):
        if raw not in id_map:
            id_map[raw] = id_next[0]; id_next[0] += 1
        return id_map[raw]

    raw_count = len(states)
    mature_count = 0
    unknowns = {"helmet": 0, "vest": 0, "mask": 0}
    results = {}

    for track_id, st in states.items():
        if st["frame_count"] < MIN_TRACK_FRAMES:
            continue
        mature_count += 1
        hvote = vote(st["hardhat"], min_known=3)
        vvote = vote(st["vest"],    min_known=2)
        mvote = vote(st["mask"],    min_known=1)
        results[did(track_id)] = {
            "helmet": hvote, "vest": vvote, "mask": mvote,
            "violations": violations_from_votes(hvote, vvote, mvote),
        }
        if hvote == "unknown": unknowns["helmet"] += 1
        if vvote == "unknown": unknowns["vest"]   += 1
        if mvote == "unknown": unknowns["mask"]   += 1

    n = max(1, mature_count)
    metrics = {
        "raw_track_count":    raw_count,
        "mature_track_count": mature_count,
        "unknown_rate": {k: round(v / n, 2) for k, v in unknowns.items()},
        "decision_counts": decision_counts or {},
    }
    return results, metrics


# ---------------------------------------------------------------------------
# Geometric association helpers
# ---------------------------------------------------------------------------

def _containment(inner: list, outer: list) -> float:
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def anatomical_region(x1, y1, x2, y2, ppe_type: str) -> list:
    pw, ph = x2 - x1, y2 - y1
    if ppe_type == "helmet":
        return [x1 + int(pw*0.05), y1 - int(ph*0.10),
                x2 - int(pw*0.05), y1 + int(ph*0.35)]
    elif ppe_type == "mask":
        return [x1 + int(pw*0.15), y1,
                x2 - int(pw*0.15), y1 + int(ph*0.28)]
    else:
        return [x1, y1 + int(ph*0.15), x2, y1 + int(ph*0.85)]


def _collect_dets(model, result, allowed_ids, min_conf) -> list[dict]:
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        dets.append({"label": str(model.names[cid]), "conf": conf,
                     "bbox": box.xyxy[0].tolist()})
    return dets


def validate_ppe(ppe_bbox_frame, label, target_tid, all_persons_frame, ppe_type) -> str:
    target = next((p for p in all_persons_frame if p["tid"] == target_tid), None)
    if target is None:
        return "unknown"
    min_self = PPE_MIN_SELF_SCORE.get(ppe_type, 0.20)
    max_nb_r = PPE_MAX_NEIGHBOR_RATIO.get(ppe_type, 0.80)
    own_region = anatomical_region(*target["box"], ppe_type)
    own_score  = _containment(ppe_bbox_frame, own_region)
    if own_score < min_self:
        return "unknown"
    for p in all_persons_frame:
        if p["tid"] == target_tid:
            continue
        nb_score = _containment(ppe_bbox_frame, anatomical_region(*p["box"], ppe_type))
        if nb_score > own_score * max_nb_r:
            return "unknown"
    return label


def neighbor_overlap_score(crop_box, all_persons_frame, target_tid) -> float:
    max_score = 0.0
    for p in all_persons_frame:
        if p["tid"] == target_tid:
            continue
        score = _containment(p["box"], crop_box)
        if score > max_score:
            max_score = score
    return max_score


def crop_has_neighbor(crop_box, all_persons_frame, target_tid, thresh=0.25) -> bool:
    return neighbor_overlap_score(crop_box, all_persons_frame, target_tid) > thresh


def is_region_too_small(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> bool:
    region = anatomical_region(x1, y1, x2, y2, ppe_type)
    w = region[2] - region[0]
    if ppe_type in ("helmet", "mask"):
        return w < MIN_HEAD_PX
    return w < MIN_TORSO_PX


# ---------------------------------------------------------------------------
# CROP pipeline (klasik — doğrudan atama)
# ---------------------------------------------------------------------------

def run_crop_pipeline(video_path, person_model, helmet_model, vest_model, mask_model,
                      h_ids, v_ids, m_ids) -> dict[int, dict]:
    cap    = cv2.VideoCapture(str(video_path))
    states = _new_states()
    id_map, id_next = {}, [1]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p_result = person_model.track(
            frame, classes=class_ids(person_model, ["person"]),
            tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False,
        )[0]
        boxes = p_result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id = int(tid)
            states[track_id]["frame_count"] += 1
            x1, y1, x2, y2 = map(int, box.tolist())

            for ppe, model, ids, conf_thr, key, isz in [
                ("helmet", helmet_model, h_ids, HELMET_CONF, "hardhat", IMGSZ),
                ("vest",   vest_model,   v_ids, VEST_CONF,   "vest",    IMGSZ),
                ("mask",   mask_model,   m_ids, MASK_CONF,   "mask",    MASK_IMGSZ),
            ]:
                crop, _, _ = crop_ppe(frame, x1, y1, x2, y2, ppe)
                label = "unknown"
                if _crop_ok(crop) and not is_region_too_small(x1, y1, x2, y2, ppe):
                    res = model.predict(crop, classes=ids, imgsz=isz,
                                        conf=conf_thr, device=DEVICE, verbose=False)[0]
                    label, _ = best_det_label(model, res, ids, conf_thr)
                states[track_id][key].append(label)

    cap.release()
    return _finalize(states, id_map, id_next)  # (results, metrics)


# ---------------------------------------------------------------------------
# CROP+ASSOC pipeline (geometrik doğrulamalı)
# ---------------------------------------------------------------------------

def run_assoc_pipeline(video_path, person_model, helmet_model, vest_model, mask_model,
                       h_ids, v_ids, m_ids):
    cap    = cv2.VideoCapture(str(video_path))
    states = _new_states()
    id_map, id_next = {}, [1]
    dcounts: dict[str, dict[str, int]] = {
        ppe: {"accepted": 0, "rejected": 0, "ambiguous": 0, "no_det": 0, "no_target": 0}
        for ppe in ("helmet", "vest", "mask")
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        p_result = person_model.track(
            frame, classes=class_ids(person_model, ["person"]),
            tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False,
        )[0]
        boxes = p_result.boxes
        if boxes is None or boxes.id is None:
            continue

        all_persons_frame = [
            {"tid": int(tid), "box": list(map(int, box.tolist()))}
            for box, tid in zip(boxes.xyxy, boxes.id)
        ]

        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id = int(tid)
            states[track_id]["frame_count"] += 1
            x1, y1, x2, y2 = map(int, box.tolist())

            for ppe, model, ids, conf_thr, key, isz in [
                ("helmet", helmet_model, h_ids, HELMET_CONF, "hardhat", IMGSZ),
                ("vest",   vest_model,   v_ids, VEST_CONF,   "vest",    IMGSZ),
                ("mask",   mask_model,   m_ids, MASK_CONF,   "mask",    MASK_IMGSZ),
            ]:
                crop, ox, oy = crop_ppe(frame, x1, y1, x2, y2, ppe)
                label = "unknown"
                if _crop_ok(crop) and not is_region_too_small(x1, y1, x2, y2, ppe):
                    res  = model.predict(crop, classes=ids, imgsz=isz,
                                         conf=conf_thr, device=DEVICE, verbose=False)[0]
                    dets = _collect_dets(model, res, ids, conf_thr)
                    if dets:
                        best   = max(dets, key=lambda d: d["conf"])
                        bbox_f = crop_to_frame(best["bbox"], ox, oy, fh, fw)
                        label  = validate_ppe(bbox_f, best["label"], track_id,
                                              all_persons_frame, ppe)
                        dcounts[ppe]["accepted" if label != "unknown" else "rejected"] += 1
                    else:
                        neighbor_overlap_score(
                            [ox, oy, ox + crop.shape[1], oy + crop.shape[0]],
                            all_persons_frame, track_id)
                        dcounts[ppe]["no_det"] += 1
                states[track_id][key].append(label)

    cap.release()
    return _finalize(states, id_map, id_next, dcounts)


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------

def summarize(results: dict[int, dict]) -> dict:
    counts = {"no_helmet": 0, "no_vest": 0, "no_mask": 0}
    for p in results.values():
        for v in p["violations"]:
            counts[v] = counts.get(v, 0) + 1
    return {"person_count": len(results), "violations": counts}


def print_result(pipeline: str, results: dict[int, dict], gt: dict, metrics: dict):
    summary = summarize(results)
    raw  = metrics.get("raw_track_count", "?")
    mat  = metrics.get("mature_track_count", "?")
    urate = metrics.get("unknown_rate", {})
    print(f"\n  [{pipeline.upper()}]  kişi={summary['person_count']}"
          f"  raw_track={raw}  mature={mat}  ihlaller={summary['violations']}")
    print(f"    unknown_rate → helmet={urate.get('helmet','?')}  "
          f"vest={urate.get('vest','?')}  mask={urate.get('mask','?')}")
    dc = metrics.get("decision_counts", {})
    if dc:
        for ppe, cnts in dc.items():
            print(f"    [{ppe}] accepted={cnts.get('accepted',0)}  "
                  f"rejected={cnts.get('rejected',0)}  "
                  f"no_det={cnts.get('no_det',0)}")
    for pid, p in sorted(results.items()):
        viols = p["violations"] or ["temiz"]
        print(f"    ID{pid}: helmet={p['helmet'][:14]}  vest={p['vest'][:14]}  mask={p['mask'][:8]}  → {viols}")
    gt_viols = gt.get("expected_violations", {})
    mismatches = []
    for viol, expected in gt_viols.items():
        got = summary["violations"].get(viol, 0)
        if expected == "all":
            if got < summary["person_count"]:
                mismatches.append(f"{viol}: {got}/{summary['person_count']} (beklenen: hepsi)")
        elif got != expected:
            mismatches.append(f"{viol}: {got} (beklenen: {expected})")
    if mismatches:
        print(f"    ⚠  Uyumsuzluk: {', '.join(mismatches)}")
    else:
        print(f"    ✓  Ground truth ile uyumlu")


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("Modeller yukleniyor...")
    helmet_model = YOLO(HELMET_MODEL_PATH)
    vest_model   = YOLO(VEST_MODEL_PATH)
    mask_model   = YOLO(MASK_MODEL_PATH)
    h_ids = class_ids(helmet_model, HELMET_CLASSES)
    v_ids = class_ids(vest_model,   VEST_CLASSES)
    m_ids = class_ids(mask_model,   MASK_CLASSES)
    print("Modeller hazır.\n")

    test_dir = Path("test")
    videos = {
        "novest_test": test_dir / "novest_test.mp4",
        "nohat_test":  test_dir / "nohat_test.mp4",
        "noppe_test":  test_dir / "noppe_test.mp4",
        "mask_test":   test_dir / "mask_test.mp4",
    }

    for name, video_path in videos.items():
        if not video_path.exists():
            print(f"[ATLA] {video_path} bulunamadi.")
            continue
        gt = GROUND_TRUTH.get(name, {})
        print(f"\n{'='*60}")
        print(f"VIDEO: {name}  —  GT: {gt.get('notes', '?')}")
        print(f"{'='*60}")

        # Her pipeline için ayrı person_model (tracker state'ini sıfırla)
        pm_crop  = YOLO(PERSON_MODEL_PATH)
        pm_assoc = YOLO(PERSON_MODEL_PATH)

        print("  [CROP]       çalışıyor...")
        crop_res,  crop_metrics  = run_crop_pipeline(video_path, pm_crop,
                                                     helmet_model, vest_model, mask_model,
                                                     h_ids, v_ids, m_ids)
        print("  [CROP+ASSOC] çalışıyor...")
        assoc_res, assoc_metrics = run_assoc_pipeline(video_path, pm_assoc,
                                                      helmet_model, vest_model, mask_model,
                                                      h_ids, v_ids, m_ids)

        print_result("crop",       crop_res,  gt, crop_metrics)
        print_result("crop+assoc", assoc_res, gt, assoc_metrics)

    print(f"\n{'='*60}")
    print("Karsilastirma tamamlandi.")


if __name__ == "__main__":
    main()
