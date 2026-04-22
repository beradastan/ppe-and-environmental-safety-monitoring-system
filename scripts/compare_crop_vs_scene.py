# -*- coding: utf-8 -*-
"""
compare_crop_vs_scene.py
========================
Crop-based vs Scene-based PPE pipeline karsilastirmasi.

Her test videosu için her iki pipeline çalışır; son frame'deki
temporal vote'lara göre per-person PPE durumu raporlanır.
Ground truth ile karsilastirilarak her pipeline'in dogruluğu hesaplanir.

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
MASK_CONF    = 0.15
PERSON_CONF  = 0.25
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"
TEMPORAL_WIN     = 20
MIN_CROP_PX      = 40
MIN_TRACK_FRAMES = 10

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask", "NO-Mask"]

# ---------------------------------------------------------------------------
# Ground truth: video adı → beklenen ihlaller (set)
# Her kişi için hangi PPE'ler eksik olmalı
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "novest_test": {
        "person_count": 2,
        "expected_violations": {"no_mask": 2, "no_vest": 1},   # kaç kişide bu ihlal var
        "notes": "2 kisi: ikisi de maske takmıyor, biri yelek takmıyor",
    },
    "nohat_test": {
        "person_count": None,  # "birden fazla" — sayı belli değil
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
    if hvote == "NO-Hardhat":    viols.append("no_helmet")
    if vvote == "NO-Safety Vest": viols.append("no_vest")
    if mvote == "NO-Mask":        viols.append("no_mask")
    return viols


# ---------------------------------------------------------------------------
# CROP-BASED pipeline
# ---------------------------------------------------------------------------

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
        cx1 = max(0, x1 - int(pw * 0.10))
        cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.10))
        cy2 = min(fh, y1 + int(ph * 0.40))
    crop = frame[cy1:cy2, cx1:cx2]
    return crop


def _crop_ok(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX


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


def run_crop_pipeline(video_path: Path, person_model, helmet_model, vest_model, mask_model,
                      h_ids, v_ids, m_ids) -> dict[int, dict]:
    """Returns {display_id: {helmet, vest, mask, violations}}"""
    cap = cv2.VideoCapture(str(video_path))
    states = defaultdict(lambda: {
        "hardhat":     deque(maxlen=TEMPORAL_WIN),
        "vest":        deque(maxlen=TEMPORAL_WIN),
        "mask":        deque(maxlen=TEMPORAL_WIN),
        "frame_count": 0,
    })
    id_map, id_next = {}, [1]

    def did(raw):
        if raw not in id_map:
            id_map[raw] = id_next[0]; id_next[0] += 1
        return id_map[raw]

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

            hcrop = crop_ppe(frame, x1, y1, x2, y2, "helmet")
            hlabel = "unknown"
            if _crop_ok(hcrop):
                hres = helmet_model.predict(hcrop, classes=h_ids, imgsz=IMGSZ,
                                            conf=HELMET_CONF, device=DEVICE, verbose=False)[0]
                hlabel, _ = best_det_label(helmet_model, hres, h_ids, HELMET_CONF)
            states[track_id]["hardhat"].append(hlabel)

            vcrop = crop_ppe(frame, x1, y1, x2, y2, "vest")
            vlabel = "unknown"
            if _crop_ok(vcrop):
                vres = vest_model.predict(vcrop, classes=v_ids, imgsz=IMGSZ,
                                          conf=VEST_CONF, device=DEVICE, verbose=False)[0]
                vlabel, _ = best_det_label(vest_model, vres, v_ids, VEST_CONF)
            states[track_id]["vest"].append(vlabel)

            mcrop = crop_ppe(frame, x1, y1, x2, y2, "mask")
            mlabel = "unknown"
            if _crop_ok(mcrop):
                mres = mask_model.predict(mcrop, classes=m_ids, imgsz=IMGSZ,
                                          conf=MASK_CONF, device=DEVICE, verbose=False)[0]
                mlabel, _ = best_det_label(mask_model, mres, m_ids, MASK_CONF)
            states[track_id]["mask"].append(mlabel)

    cap.release()

    results = {}
    for track_id, st in states.items():
        if st["frame_count"] < MIN_TRACK_FRAMES:
            continue  # ghost track — atla
        hvote = vote(st["hardhat"], min_known=3)
        vvote = vote(st["vest"],    min_known=2)
        mvote = vote(st["mask"],    min_known=2)
        results[did(track_id)] = {
            "helmet": hvote, "vest": vvote, "mask": mvote,
            "violations": violations_from_votes(hvote, vvote, mvote),
        }
    return results


# ---------------------------------------------------------------------------
# SCENE-BASED pipeline
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def scene_dets(model, result, allowed_ids, min_conf) -> list[dict]:
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        dets.append({"label": str(model.names[cid]), "conf": conf, "bbox": box.xyxy[0].tolist()})
    return dets


def match_to_person(dets, px1, py1, px2, py2, iou_thresh=0.05) -> tuple[str, float]:
    person_box = [px1, py1, px2, py2]
    best_label, best_conf = "unknown", 0.0
    for d in dets:
        if _iou(d["bbox"], person_box) >= iou_thresh and d["conf"] > best_conf:
            best_label = d["label"]
            best_conf  = d["conf"]
    return best_label, best_conf


def run_scene_pipeline(video_path: Path, person_model, helmet_model, vest_model, mask_model,
                       h_ids, v_ids, m_ids) -> dict[int, dict]:
    cap = cv2.VideoCapture(str(video_path))
    states = defaultdict(lambda: {
        "hardhat":     deque(maxlen=TEMPORAL_WIN),
        "vest":        deque(maxlen=TEMPORAL_WIN),
        "mask":        deque(maxlen=TEMPORAL_WIN),
        "frame_count": 0,
    })
    id_map, id_next = {}, [1]

    def did(raw):
        if raw not in id_map:
            id_map[raw] = id_next[0]; id_next[0] += 1
        return id_map[raw]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p_result = person_model.track(
            frame, classes=class_ids(person_model, ["person"]),
            tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=DEVICE, verbose=False,
        )[0]

        hres_scene = helmet_model.predict(frame, classes=h_ids, imgsz=IMGSZ,
                                          conf=HELMET_CONF, device=DEVICE, verbose=False)[0]
        vres_scene = vest_model.predict(frame, classes=v_ids, imgsz=IMGSZ,
                                        conf=VEST_CONF, device=DEVICE, verbose=False)[0]
        mres_scene = mask_model.predict(frame, classes=m_ids, imgsz=IMGSZ,
                                        conf=MASK_CONF, device=DEVICE, verbose=False)[0]
        all_helmet = scene_dets(helmet_model, hres_scene, h_ids, HELMET_CONF)
        all_vest   = scene_dets(vest_model,   vres_scene, v_ids, VEST_CONF)
        all_mask   = scene_dets(mask_model,   mres_scene, m_ids, MASK_CONF)

        boxes = p_result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box, tid in zip(boxes.xyxy, boxes.id):
            track_id = int(tid)
            states[track_id]["frame_count"] += 1
            x1, y1, x2, y2 = map(int, box.tolist())
            hlabel, _ = match_to_person(all_helmet, x1, y1, x2, y2)
            vlabel, _ = match_to_person(all_vest,   x1, y1, x2, y2)
            mlabel, _ = match_to_person(all_mask,   x1, y1, x2, y2)
            states[track_id]["hardhat"].append(hlabel)
            states[track_id]["vest"].append(vlabel)
            states[track_id]["mask"].append(mlabel)

    cap.release()

    results = {}
    for track_id, st in states.items():
        if st["frame_count"] < MIN_TRACK_FRAMES:
            continue  # ghost track — atla
        hvote = vote(st["hardhat"], min_known=3)
        vvote = vote(st["vest"],    min_known=2)
        mvote = vote(st["mask"],    min_known=2)
        results[did(track_id)] = {
            "helmet": hvote, "vest": vvote, "mask": mvote,
            "violations": violations_from_votes(hvote, vvote, mvote),
        }
    return results


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------

def summarize(results: dict[int, dict]) -> dict:
    """Sonuçları özetle: kişi sayısı + ihlal sayıları."""
    counts = {"no_helmet": 0, "no_vest": 0, "no_mask": 0}
    for p in results.values():
        for v in p["violations"]:
            counts[v] = counts.get(v, 0) + 1
    return {"person_count": len(results), "violations": counts}


def print_result(name: str, pipeline: str, results: dict[int, dict], gt: dict):
    summary = summarize(results)
    print(f"\n  [{pipeline.upper()}]  kişi={summary['person_count']}  ihlaller={summary['violations']}")
    for pid, p in sorted(results.items()):
        viols = p["violations"] or ["temiz"]
        print(f"    ID{pid}: helmet={p['helmet'][:12]}  vest={p['vest'][:12]}  mask={p['mask'][:12]}  → {viols}")
    # Basit uyum kontrolü
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
        print(f"    ⚠ Uyumsuzluk: {', '.join(mismatches)}")
    else:
        print(f"    ✓ Ground truth ile uyumlu")


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("Modeller yukleniyor...")
    person_model = YOLO(PERSON_MODEL_PATH)
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

        # Trackers'ı sıfırla: her pipeline için person_model yeniden yüklenir
        # (bytetrack state'ini temizlemek için)
        pm_crop  = YOLO(PERSON_MODEL_PATH)
        pm_scene = YOLO(PERSON_MODEL_PATH)

        print("  [CROP]  çalışıyor...")
        crop_res  = run_crop_pipeline(video_path, pm_crop,  helmet_model, vest_model, mask_model, h_ids, v_ids, m_ids)
        print("  [SCENE] çalışıyor...")
        scene_res = run_scene_pipeline(video_path, pm_scene, helmet_model, vest_model, mask_model, h_ids, v_ids, m_ids)

        print_result(name, "crop",  crop_res,  gt)
        print_result(name, "scene", scene_res, gt)

    print(f"\n{'='*60}")
    print("Karsilastirma tamamlandi.")


if __name__ == "__main__":
    main()
