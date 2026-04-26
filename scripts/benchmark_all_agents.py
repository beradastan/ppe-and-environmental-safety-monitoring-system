# -*- coding: utf-8 -*-
"""
Unified PPE Agent Benchmark
============================
Helmet / Vest / Mask için tüm mevcut ajanları karşılaştırır.

Kullanım:
    python scripts/benchmark_all_agents.py                      # tüm videolar, tüm ajanlar
    python scripts/benchmark_all_agents.py --ppe helmet         # sadece helmet
    python scripts/benchmark_all_agents.py --video nohat_test   # tek video stem
    python scripts/benchmark_all_agents.py --max-frames 300

Çıktı: runs/benchmarks/all_agents/<timestamp>/summary.csv + summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model kataloğu
# ---------------------------------------------------------------------------

HELMET_MODELS: list[dict] = [
    {"id": "bera_crophelmet",  "path": "models/bera/crophelmet_agent_final_best.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "vt_crophelmet",    "path": "models/vinayak_trained_byBera/crophelmet_agent_final_best.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "vt_helmet_final",  "path": "models/vinayak_trained_byBera/helmet_agent_final_best.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "vt_helmet_keren",  "path": "models/vinayak_trained_byBera/helmet_agent_keremberke_ft_best.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "vt_helmet_libre",  "path": "models/vinayak_trained_byBera/helmet_agent_libreyolo_ft_best.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "hansung",          "path": "models/hansung_yolov8_ppe.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "hexmon",           "path": "models/hexmon_yolo_ppe.pt",
     "classes": ["Hardhat", "NO-Hardhat"]},
    {"id": "amirt",            "path": "models/amirt_yolov8_ppe.pt",
     "classes": ["helmet", "no-helmet"],
     "present_cls": "helmet", "missing_cls": "no-helmet"},
]

VEST_MODELS: list[dict] = [
    {"id": "bera_vest",   "path": "models/bera/vest_agent_final_best.pt",
     "classes": ["Safety Vest", "NO-Safety Vest"]},
    {"id": "vt_vest",     "path": "models/vinayak_trained_byBera/vest_agent_final_best.pt",
     "classes": ["Safety Vest", "NO-Safety Vest"]},
    {"id": "hansung",     "path": "models/hansung_yolov8_ppe.pt",
     "classes": ["Safety Vest", "NO-Safety Vest"]},
    {"id": "hexmon",      "path": "models/hexmon_yolo_ppe.pt",
     "classes": ["Safety Vest", "NO-Safety Vest"]},
    {"id": "amirt",       "path": "models/amirt_yolov8_ppe.pt",
     "classes": ["vest", "no-vest"],
     "present_cls": "vest", "missing_cls": "no-vest"},
]

MASK_MODELS: list[dict] = [
    {"id": "bera_cropmask", "path": "models/bera/cropmask_agent_final_best.pt",
     "classes": ["Mask", "NO-Mask"]},
    {"id": "hansung",       "path": "models/hansung_yolov8_ppe.pt",
     "classes": ["Mask", "NO-Mask"]},
    {"id": "hexmon",        "path": "models/hexmon_yolo_ppe.pt",
     "classes": ["Mask", "NO-Mask"]},
]

PERSON_MODEL = "models/person_agent_scene_vinayakstyle_best.pt"
TRACKER      = "bytetrack.yaml"
PERSON_CONF  = 0.25
IMGSZ        = 640
TEMPORAL_WIN = 20

# conf eşikleri (production değerleri)
CONF: dict[str, float] = {"helmet": 0.15, "vest": 0.30, "mask": 0.10}
IMGSZ_PPE: dict[str, int] = {"helmet": 640, "vest": 640, "mask": 1280}

# Anatomik crop oranları (run_live_video.py ile aynı)
CROP_PARAMS: dict[str, dict] = {
    "helmet": {"x_pad": 0.10, "y_top": 0.15, "y_bot_frac": 0.40},
    "vest":   {"x_pad": 0.15, "y_top_frac": 0.10, "y_bot_frac": 0.90},
    "mask":   {"x_pad": 0.15, "y_top": 0.10, "y_bot_frac": 0.45},
}

TEST_VIDEOS: list[dict] = [
    {"stem": "nohat_test",  "gt_note": "hepsi no_mask, 2 kisi no_helmet"},
    {"stem": "novest_test", "gt_note": "2 kisi no_mask, 1 kisi no_vest"},
    {"stem": "mask_test",   "gt_note": "1 kisi: vest+mask ok, helmet yok"},
    {"stem": "noppe_test",  "gt_note": "tum PPE eksik"},
    {"stem": "general_test","gt_note": "karisik"},
]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def class_ids(model: YOLO, names: list[str]) -> list[int]:
    n2id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in n2id]
    if missing:
        raise ValueError(f"Eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [n2id[n] for n in names]


def crop_ppe(frame, x1: int, y1: int, x2: int, y2: int, ppe: str):
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    p = CROP_PARAMS[ppe]
    if ppe == "helmet":
        cx1 = max(0, x1 - int(pw * p["x_pad"]))
        cy1 = max(0, y1 - int(ph * p["y_top"]))
        cx2 = min(fw, x2 + int(pw * p["x_pad"]))
        cy2 = min(fh, y1 + int(ph * p["y_bot_frac"]))
    elif ppe == "vest":
        cx1 = max(0, x1 - int(pw * p["x_pad"]))
        cy1 = max(0, y1 + int(ph * p["y_top_frac"]))
        cx2 = min(fw, x2 + int(pw * p["x_pad"]))
        cy2 = min(fh, y1 + int(ph * p["y_bot_frac"]))
    else:  # mask
        cx1 = max(0, x1 - int(pw * p["x_pad"]))
        cy1 = max(0, y1 - int(ph * p["y_top"]))
        cx2 = min(fw, x2 + int(pw * p["x_pad"]))
        cy2 = min(fh, y1 + int(ph * p["y_bot_frac"]))
    crop = frame[cy1:cy2, cx1:cx2]
    return crop, cx1, cy1


def _containment(inner: list, outer: list) -> float:
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def anatomical_region(x1, y1, x2, y2, ppe_type: str) -> list:
    pw, ph = x2 - x1, y2 - y1
    if ppe_type == "helmet":
        return [x1 + int(pw * 0.05), y1 - int(ph * 0.10),
                x2 - int(pw * 0.05), y1 + int(ph * 0.35)]
    elif ppe_type == "mask":
        return [x1 + int(pw * 0.15), y1,
                x2 - int(pw * 0.15), y1 + int(ph * 0.28)]
    else:
        return [x1, y1 + int(ph * 0.15), x2, y1 + int(ph * 0.85)]


MIN_SELF = {"helmet": 0.20, "vest": 0.20, "mask": 0.15}
MAX_NB_R = {"helmet": 0.80, "vest": 0.75, "mask": 0.90}


def validate_geo(ppe_bbox_f: list, label: str, tid: int,
                 all_persons: list[dict], ppe_type: str) -> str:
    target = next((p for p in all_persons if p["tid"] == tid), None)
    if target is None:
        return "unknown"
    own_region = anatomical_region(*target["box"], ppe_type)
    own_score  = _containment(ppe_bbox_f, own_region)
    if own_score < MIN_SELF[ppe_type]:
        return "unknown"
    for p in all_persons:
        if p["tid"] == tid:
            continue
        nb_score = _containment(ppe_bbox_f, anatomical_region(*p["box"], ppe_type))
        if nb_score > own_score * MAX_NB_R[ppe_type]:
            return "unknown"
    return label


def vote(q: deque) -> str:
    if not q:
        return "unknown"
    top = Counter(q).most_common(1)[0][0]
    if top != "unknown":
        return top
    known = [v for v in q if v != "unknown"]
    if len(known) < 3:
        return "unknown"
    t, c = Counter(known).most_common(1)[0]
    return t if c / len(known) >= 0.5 else "unknown"


def mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Tek model + tek video benchmark
# ---------------------------------------------------------------------------

def run_one(
    video_path: Path,
    ppe_type: str,
    model_cfg: dict,
    person_model: YOLO,
    device: str,
    max_frames: int,
) -> dict[str, Any]:
    model_path = resolve(model_cfg["path"])
    ppe_model  = YOLO(str(model_path))
    ppe_ids    = class_ids(ppe_model, model_cfg["classes"])

    # normalise class names → Hardhat/NO-Hardhat tarzına
    present_cls = model_cfg.get("present_cls", model_cfg["classes"][0])
    missing_cls = model_cfg.get("missing_cls", model_cfg["classes"][1])

    person_cls = next((n for n in person_model.names.values() if n.lower() == "person"), "person")
    p_ids    = class_ids(person_model, [person_cls])
    conf     = CONF[ppe_type]
    imgsz_pp = IMGSZ_PPE[ppe_type]

    states: dict[int, deque] = defaultdict(lambda: deque(maxlen=TEMPORAL_WIN))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"video acilamadi: {video_path}"}

    frames = 0
    conf_vals: list[float] = []
    counts: Counter = Counter()
    label_changes = 0
    observations  = 0
    last_labels: dict[int, str] = {}
    track_ids: set[int] = set()
    active_counts: list[int] = []
    t0 = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        fh, fw = frame.shape[:2]

        p_res = person_model.track(
            frame, classes=p_ids, tracker=TRACKER, persist=True,
            imgsz=IMGSZ, conf=PERSON_CONF, device=device, verbose=False,
        )[0]

        boxes = p_res.boxes
        all_persons: list[dict] = []
        if boxes is not None and boxes.id is not None:
            all_persons = [
                {"tid": int(tid), "box": list(map(int, box.tolist()))}
                for box, tid in zip(boxes.xyxy, boxes.id)
            ]

        if boxes is not None and boxes.id is not None:
            active_counts.append(len(boxes.id))
            for box, tid in zip(boxes.xyxy, boxes.id):
                track_id = int(tid)
                track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.tolist())

                crop, ox, oy = crop_ppe(frame, x1, y1, x2, y2, ppe_type)
                if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                    states[track_id].append("unknown")
                    counts["unknown"] += 1
                    observations += 1
                    continue

                res = ppe_model.predict(
                    crop, classes=ppe_ids, imgsz=imgsz_pp,
                    conf=conf, device=device, verbose=False,
                )[0]

                best_label, best_conf, best_bbox = "unknown", 0.0, None
                if res.boxes:
                    for b in res.boxes:
                        if int(b.cls[0]) in ppe_ids and float(b.conf[0]) > best_conf:
                            best_conf  = float(b.conf[0])
                            best_label = str(ppe_model.names[int(b.cls[0])])
                            best_bbox  = b.xyxy[0].tolist()

                if best_bbox is not None:
                    bbox_f = [
                        min(fw-1, int(ox + best_bbox[0])), min(fh-1, int(oy + best_bbox[1])),
                        min(fw-1, int(ox + best_bbox[2])), min(fh-1, int(oy + best_bbox[3])),
                    ]
                    best_label = validate_geo(bbox_f, best_label, track_id, all_persons, ppe_type)

                # normalize label → canonical
                if best_label == present_cls:
                    best_label = model_cfg["classes"][0]
                elif best_label == missing_cls:
                    best_label = model_cfg["classes"][1]

                states[track_id].append(best_label)
                counts[best_label] += 1
                if best_conf > 0:
                    conf_vals.append(best_conf)

                if track_id in last_labels:
                    label_changes += int(last_labels[track_id] != best_label)
                last_labels[track_id] = best_label
                observations += 1
        else:
            active_counts.append(0)

        if frames >= max_frames:
            break

    cap.release()
    elapsed = time.perf_counter() - t0

    # final voted labels per track
    voted: dict[int, str] = {tid: vote(q) for tid, q in states.items()}
    violation_label = model_cfg["classes"][1]   # NO-Hardhat / NO-Safety Vest / NO-Mask
    violations = sum(1 for v in voted.values() if v == violation_label)
    unknowns   = sum(1 for v in voted.values() if v == "unknown")
    total_tracks = len(voted)

    unknown_rate  = counts["unknown"] / observations if observations else 1.0
    known_rate    = 1.0 - unknown_rate
    stability     = 1.0 - (label_changes / max(1, observations - 1))
    avg_active    = mean([float(v) for v in active_counts])

    return {
        "ppe_type":        ppe_type,
        "model_id":        model_cfg["id"],
        "video":           video_path.stem,
        "frames":          frames,
        "fps":             round(frames / elapsed, 2) if elapsed else 0.0,
        "tracks_total":    total_tracks,
        "tracks_violation": violations,
        "tracks_unknown":  unknowns,
        "known_rate":      round(known_rate, 4),
        "unknown_rate":    round(unknown_rate, 4),
        "avg_conf":        round(mean(conf_vals), 4),
        "stability":       round(max(0.0, stability), 4),
        "avg_active":      round(avg_active, 2),
        "label_counts":    dict(counts),
        "voted_labels":    {str(k): v for k, v in voted.items()},
    }


def score(r: dict) -> float:
    return round(
        0.40 * r["known_rate"]
        + 0.30 * r["stability"]
        + 0.20 * r["avg_conf"],
        4,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ppe", choices=["helmet", "vest", "mask", "all"], default="all")
    p.add_argument("--video", help="Video stem (nohat_test vb.), belirtilmezse tümü")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--output-dir", default="runs/benchmarks/all_agents")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA yok, cpu'ya geciyor.")
        device = "cpu"

    person_path = resolve(PERSON_MODEL)
    if not person_path.exists():
        sys.exit(f"Person model bulunamadi: {person_path}")
    print(f"Person model yukleniyor: {person_path.name}")
    person_model = YOLO(str(person_path))

    # PPE type seçimi
    ppe_map: dict[str, list[dict]] = {
        "helmet": HELMET_MODELS,
        "vest":   VEST_MODELS,
        "mask":   MASK_MODELS,
    }
    selected_ppe = list(ppe_map.keys()) if args.ppe == "all" else [args.ppe]

    # Video seçimi
    videos = TEST_VIDEOS
    if args.video:
        videos = [v for v in TEST_VIDEOS if v["stem"] == args.video]
        if not videos:
            sys.exit(f"Bilinmeyen video: {args.video}")

    all_results: list[dict] = []

    for ppe_type in selected_ppe:
        models_cfg = ppe_map[ppe_type]
        print(f"\n{'='*65}")
        print(f"  PPE: {ppe_type.upper()}  —  {len(models_cfg)} model, {len(videos)} video")
        print(f"{'='*65}")

        for video_meta in videos:
            video_path = resolve("test/" + video_meta["stem"] + ".mp4")
            if not video_path.exists():
                print(f"  [SKIP] {video_path.name} bulunamadi")
                continue

            print(f"\n  Video: {video_path.name}  (GT: {video_meta['gt_note']})")
            for mcfg in models_cfg:
                mpath = resolve(mcfg["path"])
                if not mpath.exists():
                    print(f"    [SKIP] {mcfg['id']} — model dosyasi yok")
                    continue
                print(f"    [{mcfg['id']}]  ", end="", flush=True)
                r = run_one(video_path, ppe_type, mcfg, person_model, device, args.max_frames)
                if "error" in r:
                    print(r["error"])
                    continue
                r["score"] = score(r)
                all_results.append(r)
                print(
                    f"fps={r['fps']:5.1f}  known={r['known_rate']*100:5.1f}%  "
                    f"conf={r['avg_conf']:.3f}  stab={r['stability']:.3f}  "
                    f"score={r['score']:.4f}  "
                    f"viols={r['tracks_violation']}/{r['tracks_total']}  "
                    f"unk={r['tracks_unknown']}"
                )

    if not all_results:
        sys.exit("Hic sonuc uretilmedi.")

    # ---------------------------------------------------------------------------
    # Özet ranking — her PPE tipi için en iyi model
    # ---------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  SONUÇ: PPE tipine göre sıralama (ortalama score)")
    print(f"{'='*65}")

    best_per_ppe: dict[str, str] = {}
    for ppe_type in selected_ppe:
        rows = [r for r in all_results if r["ppe_type"] == ppe_type]
        if not rows:
            continue
        # model bazında ortalama score
        by_model: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_model[r["model_id"]].append(r["score"])
        ranked = sorted(by_model.items(), key=lambda kv: mean(kv[1]), reverse=True)
        print(f"\n  {ppe_type.upper()}")
        for i, (mid, scores) in enumerate(ranked):
            avg = mean(scores)
            marker = " ← EN İYİ" if i == 0 else ""
            print(f"    {i+1}. {mid:<22}  avg_score={avg:.4f}{marker}")
        best_per_ppe[ppe_type] = ranked[0][0]

    print(f"\n{'='*65}")
    print("  ÖNERİLEN MODEL AYARLARI:")
    print(f"{'='*65}")
    for ppe_type, model_id in best_per_ppe.items():
        cfg = next(m for m in ppe_map[ppe_type] if m["id"] == model_id)
        print(f"  {ppe_type.upper():8s}  {model_id:<22}  {cfg['path']}")

    # ---------------------------------------------------------------------------
    # Dosya çıktısı
    # ---------------------------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = resolve(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    best_path = out_dir / "best_models.json"

    fieldnames = [
        "score", "ppe_type", "model_id", "video", "fps", "frames",
        "tracks_total", "tracks_violation", "tracks_unknown",
        "known_rate", "unknown_rate", "avg_conf", "stability", "avg_active",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(all_results, key=lambda r: (-r["score"], r["ppe_type"])))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_per_ppe, f, indent=2, ensure_ascii=False)

    print(f"\n  Raporlar: {out_dir}")
    print(f"  CSV   : {csv_path.name}")
    print(f"  JSON  : {json_path.name}")
    print(f"  Best  : {best_path.name}")


if __name__ == "__main__":
    main()
