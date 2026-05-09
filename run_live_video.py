# -*- coding: utf-8 -*-
"""
run_live_video.py
=================
Birleşik PPE pipeline — crop-based veya scene-based modda çalışır.

Kullanim:
    python run_live_video.py --mode crop   # crop-based (varsayilan)
    python run_live_video.py --mode scene  # scene-based
    python run_live_video.py --mode crop --display
    python run_live_video.py --mode scene --video test/nohat_test.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import os
import cv2
import yaml
import requests
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    _TORCH_CUDA = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Yardımcı: backend URL + kamera bildirimi
# ---------------------------------------------------------------------------

def _build_alarm_text(event_type: str, persons: list[dict]) -> str:
    if event_type == "fire_detected":
        return "Yangın/duman tespit edildi! Acil müdahale gerekiyor."
    violators = []
    for p in persons:
        viols = p.get("violations", [])
        if not viols:
            continue
        names = []
        if "no_helmet" in viols: names.append("baret")
        if "no_vest"   in viols: names.append("yelek")
        if "no_mask"   in viols: names.append("maske")
        violators.append(f"#{p['track_id']} ({', '.join(names)} eksik)")
    ppe_str = ", ".join(violators) if violators else "PPE ihlali"
    if event_type == "multi_hazard":
        return f"Yangın + PPE ihlali tespit edildi — {ppe_str}."
    return f"PPE ihlali: {ppe_str}."


def _get_backend_url() -> str:
    try:
        with open("config.yaml", encoding="utf-8") as f:
            _b = (yaml.safe_load(f) or {}).get("backend", {})
        return f"http://localhost:{_b.get('port', 5050)}"
    except Exception:
        return "http://localhost:5050"


def _notify_camera_status(status: str, camera_id: str | None = None, zone: str | None = None) -> None:
    try:
        requests.post(
            f"{_get_backend_url()}/api/pipeline/camera-status",
            json={"status": status, "camera_id": camera_id or "", "zone": zone or ""},
            timeout=2,
        )
        print(f"  [KAMERA] Durum: {status}")
    except Exception:
        pass


def _close_event(event_id: str, repeat_count: int | None = None, duration_sec: float | None = None) -> None:
    try:
        body: dict = {}
        if repeat_count is not None:
            body["repeat_count"] = repeat_count
        if duration_sec is not None:
            body["duration_sec"] = duration_sec
        requests.patch(
            f"{_get_backend_url()}/api/events/{event_id}/close",
            json=body,
            timeout=5,
        )
        print(f"  [CLOSE] {event_id} kapatildi.")
    except Exception as e:
        print(f"  [CLOSE] Hata: {e}")


# ---------------------------------------------------------------------------
# LLM entegrasyonu (her iki modda da etkin, config.yaml llm.enabled ile kontrol)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Sabitler — ortak
# ---------------------------------------------------------------------------

def _load_model_cfg() -> dict:
    try:
        with open("config.yaml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("models", {})
    except Exception:
        return {}

_MODEL_CFG = _load_model_cfg()
_DEV_CFG = _MODEL_CFG.get("device", "auto")
_DEVICE  = ("cuda" if _TORCH_CUDA else "cpu") if _DEV_CFG == "auto" else _DEV_CFG

FIRE_MODEL_PATH   = _MODEL_CFG.get("fire_model",   "models/bera/fire_smoke_other_agent_final_best.pt")
PERSON_MODEL_PATH = _MODEL_CFG.get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")

# Crop-based varsayılan model yolları (config.yaml models.crop'tan okunur)
_CROP_CFG = _MODEL_CFG.get("crop", {})
CROP_HELMET_MODEL_PATH = _CROP_CFG.get("helmet_model", "models/bera/crophelmet_agent_final_best.pt")
CROP_VEST_MODEL_PATH   = _CROP_CFG.get("vest_model",   "models/bera/cropvest_agent_final_best.pt")
CROP_MASK_MODEL_PATH   = _CROP_CFG.get("mask_model",   "models/bera/cropmask_agent_final_best.pt")
CROP_MASK_IMGSZ        = int(_CROP_CFG.get("mask_imgsz", 640))

# Scene-based varsayılan model yolları (config.yaml models.scene'den okunur)
_SCENE_CFG = _MODEL_CFG.get("scene", {})
SCENE_HELMET_MODEL_PATH = _SCENE_CFG.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
SCENE_VEST_MODEL_PATH   = _SCENE_CFG.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
SCENE_MASK_MODEL_PATH   = _SCENE_CFG.get("mask_model",   "models/mask_agent_scene_200ep_yolov8m_best.pt")

# Confidence eşikleri — crop modu
CROP_HELMET_CONF = 0.15
CROP_VEST_CONF   = 0.35
CROP_MASK_CONF   = 0.10

# Confidence eşikleri — scene modu (benchmark_scene_conf.py + skip=2 ile optimize edildi)
SCENE_HELMET_CONF = 0.25
SCENE_VEST_CONF   = 0.30
SCENE_MASK_CONF   = 0.05

FIRE_CONF    = 0.75
PERSON_CONF  = 0.25
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"
TEMPORAL_WIN = 20
STATES_CLEANUP_EVERY = 300

# Crop-specific
HELMET_PAD  = 0.80
VEST_PAD    = 0.60
MASK_PAD    = 0.80
PPE_INFER_EVERY  = 4
FIRE_INFER_EVERY = 5
MIN_CROP_PX      = 40
MIN_TRACK_FRAMES = 10
PPE_MIN_SELF_SCORE: dict[str, float] = {"helmet": 0.20, "vest": 0.20, "mask": 0.15}
PPE_MAX_NEIGHBOR_RATIO: dict[str, float] = {"helmet": 0.80, "vest": 0.75, "mask": 0.90}
MIN_HEAD_PX  = 30
MIN_TORSO_PX = 50
TELEMETRY    = False

# Scene-specific (benchmark_scene_frac.py + benchmark_scene_temporal.py ile optimize edildi)
INSIDE_FRAC_THR      = 0.20
SCENE_TEMPORAL_WIN   = 30
SCENE_PPE_INFER_EVERY = 2   # PPE modellerini her 2 frame'de bir çalıştır → ~30fps

# Çözünürlük sınırlama — yüksek çözünürlüklü kaynaklarda CPU/GPU transfer yükünü azaltır.
# Frame bu genişliği aşarsa orantılı olarak küçültülür. None = devre dışı.
PIPELINE_MAX_WIDTH = 1280

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask", "NO-Mask"]

COLOR_OK      = (0, 200, 0)
COLOR_WARN    = (0, 100, 255)
COLOR_DANGER  = (0, 0, 230)
COLOR_UNKNOWN = (0, 200, 255)
COLOR_FIRE    = (0, 60, 255)

# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------

def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [name_to_id[n] for n in names]


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


def compliance_color(hvote: str, vvote: str, mvote: str) -> tuple[tuple[int, int, int], list[str]]:
    h_miss = hvote == "NO-Hardhat"
    v_miss = vvote == "NO-Safety Vest"
    m_miss = mvote == "NO-Mask"
    h_ok   = hvote == "Hardhat"
    v_ok   = vvote == "Safety Vest"
    m_ok   = mvote == "Mask"
    viols  = []
    if h_miss: viols.append("no_helmet")
    if v_miss: viols.append("no_vest")
    if m_miss: viols.append("no_mask")
    if h_ok and v_ok and m_ok:
        return COLOR_OK, []
    if viols:
        return COLOR_DANGER if len(viols) >= 2 else COLOR_WARN, viols
    return COLOR_UNKNOWN, viols


def draw_box(frame, x1, y1, x2, y2, text: str, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    scale, thick, pad = 0.60, 2, 6
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bx1 = max(0, x1)
    by2 = max(th + bl + pad * 2, y1)
    by1 = by2 - th - bl - pad * 2
    bx2 = min(frame.shape[1] - 1, bx1 + tw + pad * 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    fg  = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad - bl),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def draw_ppe_box(frame, x1, y1, x2, y2, label: str, color, tag: str):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    scale, thick, pad = 0.45, 1, 4
    text = f"{tag} {label}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bx1 = max(0, x1)
    by2 = min(frame.shape[0] - 1, y2)
    by1 = max(0, by2 - th - bl - pad * 2)
    bx2 = min(frame.shape[1] - 1, bx1 + tw + pad * 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    fg  = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad - bl),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def draw_hud(frame, event_id, status, repeat, viols_per_person):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 200) if status in ("new", "active") else (0, 200, 0)
    cv2.putText(frame, f"EVENT: {event_id or 'N/A'} [{status.upper()}]",
                (10, 30), font, 0.7, color, 2)
    cv2.putText(frame, f"Repeat: {repeat}  Active violations: {len(viols_per_person)}",
                (10, 58), font, 0.6, color, 2)


# ---------------------------------------------------------------------------
# Crop-tabanlı detection yardımcıları
# ---------------------------------------------------------------------------

def crop_ppe(frame, x1, y1, x2, y2, ppe: str):
    fh, fw = frame.shape[:2]
    pw = x2 - x1
    ph = y2 - y1
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
    return frame[cy1:cy2, cx1:cx2], cx1, cy1


def _crop_ok(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX


def crop_to_frame(bbox, ox: int, oy: int, fh: int, fw: int):
    dx1, dy1, dx2, dy2 = bbox
    return (
        min(fw - 1, int(ox + dx1)),
        min(fh - 1, int(oy + dy1)),
        min(fw - 1, int(ox + dx2)),
        min(fh - 1, int(oy + dy2)),
    )


def _containment(inner: list, outer: list) -> float:
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def anatomical_region(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> list:
    pw, ph = x2 - x1, y2 - y1
    if ppe_type == "helmet":
        return [x1 + int(pw * 0.05), y1 - int(ph * 0.10),
                x2 - int(pw * 0.05), y1 + int(ph * 0.35)]
    elif ppe_type == "mask":
        return [x1 + int(pw * 0.15), y1,
                x2 - int(pw * 0.15), y1 + int(ph * 0.28)]
    else:  # vest
        return [x1, y1 + int(ph * 0.15), x2, y1 + int(ph * 0.85)]


def _collect_dets(model: YOLO, result, allowed_ids: list[int], min_conf: float) -> list[dict]:
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


def _validate_ppe_scored(
    ppe_bbox_frame: list,
    label: str,
    target_tid: int,
    all_persons_frame: list[dict],
    ppe_type: str,
) -> tuple[str, float, float, str]:
    target = next((p for p in all_persons_frame if p["tid"] == target_tid), None)
    if target is None:
        return "unknown", 0.0, 0.0, "no_target"
    min_self = PPE_MIN_SELF_SCORE.get(ppe_type, 0.20)
    max_nb_r = PPE_MAX_NEIGHBOR_RATIO.get(ppe_type, 0.80)
    own_region = anatomical_region(*target["box"], ppe_type)
    own_score  = _containment(ppe_bbox_frame, own_region)
    if own_score < min_self:
        return "unknown", own_score, 0.0, "rejected"
    max_nb = 0.0
    for p in all_persons_frame:
        if p["tid"] == target_tid:
            continue
        nb_score = _containment(ppe_bbox_frame, anatomical_region(*p["box"], ppe_type))
        if nb_score > max_nb:
            max_nb = nb_score
        if nb_score > own_score * max_nb_r:
            return "unknown", own_score, nb_score, "ambiguous"
    return label, own_score, max_nb, "accepted"


def neighbor_overlap_score(
    crop_box: list,
    all_persons_frame: list[dict],
    target_tid: int,
) -> float:
    max_score = 0.0
    for p in all_persons_frame:
        if p["tid"] == target_tid:
            continue
        score = _containment(p["box"], crop_box)
        if score > max_score:
            max_score = score
    return max_score


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def _global_assign_ppe(
    candidates: list[dict],
    iou_thresh: float = 0.40,
) -> dict[int, dict]:
    sorted_c = sorted(
        (c for c in candidates if c["reason"] == "accepted"),
        key=lambda c: c["conf"] * c["own_score"],
        reverse=True,
    )
    used: list[list] = []
    result: dict[int, dict] = {}
    for cand in sorted_c:
        bbox = cand["bbox_f"]
        if not any(_iou(bbox, u) > iou_thresh for u in used):
            used.append(bbox)
            result[cand["tid"]] = cand
    return result


def is_region_too_small(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> bool:
    region = anatomical_region(x1, y1, x2, y2, ppe_type)
    w = region[2] - region[0]
    if ppe_type in ("helmet", "mask"):
        return w < MIN_HEAD_PX
    return w < MIN_TORSO_PX


@dataclass
class PPEDecision:
    frame_idx:    int
    stable_pid:   int
    ppe_type:     str
    raw_label:    str
    conf:         float
    own_score:    float
    neighbor_pen: float
    ambiguous:    bool
    accepted:     bool
    reason:       str


def _log_ppe_decision(d: PPEDecision) -> None:
    print(
        f"  [TEL] f={d.frame_idx:5d} pid={d.stable_pid:3d} {d.ppe_type:6s}"
        f" raw={d.raw_label:18s} conf={d.conf:.2f}"
        f" own={d.own_score:.2f} nbpen={d.neighbor_pen:.2f}"
        f" {'ACCEPT' if d.accepted else 'REJECT':6s} ({d.reason})"
    )


# ---------------------------------------------------------------------------
# Scene-tabanlı detection yardımcıları
# ---------------------------------------------------------------------------

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


def _scene_dets(
    model: YOLO,
    frame,
    allowed_ids: set[int],
    min_conf: float,
    device: str,
) -> list[tuple[str, float, list]]:
    res = model.predict(frame, imgsz=IMGSZ, conf=min_conf, device=device, verbose=False)[0]
    if not res.boxes:
        return []
    return [
        (str(model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist())
        for b in res.boxes
        if int(b.cls[0]) in allowed_ids
    ]


def _best_scene(
    dets: list[tuple[str, float, list]],
    person_box: list,
) -> tuple[str, float, list | None]:
    best: tuple[str, float, list] | None = None
    for lbl, c, bbox in dets:
        if _inside_frac(bbox, person_box) >= INSIDE_FRAC_THR:
            if best is None or c > best[1]:
                best = (lbl, c, bbox)
    return best if best else ("unknown", 0.0, None)


# ---------------------------------------------------------------------------
# Event kayıt
# ---------------------------------------------------------------------------

def save_event(
    event_info:       dict,
    frame,
    results_dir:      Path,
    persons_snapshot: list[dict],
    fire_conf:        float = 0.0,
    smoke_detected:   bool  = False,
    smoke_conf:       float = 0.0,
    camera_id:        str | None = None,
    zone:             str | None = None,
) -> None:
    event_id     = event_info["event_id"]
    event_status = event_info["event_status"]
    base_sig     = event_info.get("signature", {})

    event_dir = results_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    json_path = event_dir / f"{event_id}_new.json"
    img_path  = event_dir / f"{event_id}_new.jpg"

    has_ppe  = base_sig.get("helmet_violation") or base_sig.get("vest_violation") or base_sig.get("mask_violation")
    has_fire = base_sig.get("fire_detected", False)
    if has_ppe and has_fire:
        event_type = "multi_hazard"
    elif has_fire:
        event_type = "fire_detected"
    else:
        event_type = "ppe_violation"

    dur_map = {p["track_id"]: p.get("duration_sec", 0.0)
               for p in event_info.get("person_violations", [])}

    persons_detail = [
        {
            "track_id":      p["track_id"],
            "helmet_status": p.get("helmet_status", "unknown"),
            "vest_status":   p.get("vest_status",   "unknown"),
            "mask_status":   p.get("mask_status",   "unknown"),
            "violations":    p.get("violations",    []),
            "helmet_conf":   p.get("helmet_conf",   0.0),
            "vest_conf":     p.get("vest_conf",     0.0),
            "mask_conf":     p.get("mask_conf",     0.0),
            "duration_sec":  dur_map.get(p["track_id"], 0.0),
        }
        for p in persons_snapshot
    ]

    scene = {
        "fire_detected":  has_fire,
        "fire_conf":      round(fire_conf, 2),
        "smoke_detected": smoke_detected,
        "smoke_conf":     round(smoke_conf, 2),
    }

    alarm_text   = _build_alarm_text(event_type, persons_detail)
    enriched_sig = {
        **base_sig,
        "helmet_missing_ids": [p["track_id"] for p in persons_detail if "no_helmet" in p.get("violations", [])],
        "vest_missing_ids":   [p["track_id"] for p in persons_detail if "no_vest"   in p.get("violations", [])],
        "mask_missing_ids":   [p["track_id"] for p in persons_detail if "no_mask"   in p.get("violations", [])],
    }

    payload = {
        "event_id":      event_id,
        "event_type":    event_type,
        "event_status":  event_status,
        "timestamp":     datetime.now().isoformat(),
        "duration_sec":  event_info.get("duration_sec", 0.0),
        "repeat_count":  event_info.get("repeat_count", 1),
        "total_persons": len(persons_detail),
        "change_reason": event_info.get("change_reason", ""),
        "persons":       persons_detail,
        "scene":         scene,
        "signature":     enriched_sig,
        "alarm_text":    alarm_text,
        "llm_report":    None,
        "camera_id":     camera_id,
        "zone":          zone,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cv2.imwrite(str(img_path), frame)
    print(f"  [KAYIT] {event_id}/new  alarm: {alarm_text}")

    try:
        import winsound
        winsound.Beep(1000, 400)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sonuçlar temizliği
# ---------------------------------------------------------------------------

def _cleanup_old_results(results_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("evt_")],
        key=lambda d: d.name,
    )
    to_delete = dirs[:-keep] if len(dirs) > keep else []
    for d in to_delete:
        try:
            import shutil
            shutil.rmtree(d)
            print(f"  [TEMIZLIK] Silindi: {d.name}")
        except Exception as e:
            print(f"  [TEMIZLIK] Hata ({d.name}): {e}")


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def run(args):
    device = args.device
    mode   = args.mode

    print(f"Modeller yukleniyor... (mod: {mode})")
    person_model = YOLO(PERSON_MODEL_PATH)
    fire_model   = YOLO(FIRE_MODEL_PATH)

    if mode == "crop":
        helmet_model = YOLO(CROP_HELMET_MODEL_PATH)
        vest_model   = YOLO(CROP_VEST_MODEL_PATH)
        mask_model   = YOLO(CROP_MASK_MODEL_PATH)
        print(f"  Crop modelleri: helmet={CROP_HELMET_MODEL_PATH}")
    else:  # scene
        helmet_model = YOLO(SCENE_HELMET_MODEL_PATH)
        vest_model   = YOLO(SCENE_VEST_MODEL_PATH)
        mask_model   = YOLO(SCENE_MASK_MODEL_PATH)
        print(f"  Scene modelleri: helmet={SCENE_HELMET_MODEL_PATH}")

    _person_cls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = class_ids(person_model, [_person_cls])
    h_ids = class_ids(helmet_model, HELMET_CLASSES)
    v_ids = class_ids(vest_model,   VEST_CLASSES)
    m_ids = class_ids(mask_model,   MASK_CLASSES)
    h_ids_set = set(h_ids)
    v_ids_set = set(v_ids)
    m_ids_set = set(m_ids)

    from event_manager import PersonEventManager
    from tracking_identity import TrackReattacher
    _full_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    _ppe_cfg  = _full_cfg.get("ppe_pipeline", {})
    _em_cfg   = _full_cfg.get("event_manager", {})

    _use_fire           = bool( _ppe_cfg.get("use_fire",            True))
    _fire_min_area      = float(_ppe_cfg.get("fire_min_area_ratio", 0.01))
    _fire_growth_window = int(  _ppe_cfg.get("fire_growth_window",  10))
    _fire_growth_factor = float(_ppe_cfg.get("fire_growth_factor",  1.5))
    _fire_area_history:  deque = deque(maxlen=_fire_growth_window * 2)
    _smoke_area_history: deque = deque(maxlen=_fire_growth_window * 2)

    event_manager = PersonEventManager(
        new_confirm_sec      = float(_em_cfg.get("new_confirm_sec",      3.0)),
        resolved_confirm_sec = float(_em_cfg.get("resolved_confirm_sec", 5.0)),
        fire_confirm_frames  = int(  _em_cfg.get("fire_confirm_frames",  20)),
        fire_clear_frames    = int(  _em_cfg.get("fire_clear_frames",    10)),
        check_helmet=_ppe_cfg.get("use_helmet", True),
        check_vest=_ppe_cfg.get("use_vest", True),
        check_mask=_ppe_cfg.get("use_mask", False),
    )
    reattacher = TrackReattacher()

    _temporal_win = SCENE_TEMPORAL_WIN if mode == "scene" else TEMPORAL_WIN
    states = defaultdict(lambda: {
        "hardhat":     deque(maxlen=_temporal_win),
        "vest":        deque(maxlen=_temporal_win),
        "mask":        deque(maxlen=_temporal_win),
        "frame_count": 0,
    })

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    try:
        _keep = int(_full_cfg.get("results_keep_events", 50))
    except Exception:
        _keep = 50
    _cleanup_old_results(results_dir, _keep)

    existing = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("evt_")]
    start_counter = max((int(d.name.split("_")[1]) for d in existing), default=0)
    event_manager._counter = start_counter
    if start_counter:
        print(f"  Mevcut {start_counter} event bulundu, sayac {start_counter}'den devam ediyor.")

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(str(source) if args.video else int(source))
    if not cap.isOpened():
        sys.exit(f"Kaynak acilamadi: {source}")

    display = args.display
    if display:
        cv2.namedWindow("Factory Safety", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Factory Safety", 1280, 720)

    frame_idx         = 0
    event_count       = 0
    seen_stable_pids: set[int] = set()
    _fire_raw         = False
    _fire_conf_max    = 0.0
    _smoke_raw        = False
    _smoke_conf_max   = 0.0
    _max_smoke_area   = 0.0

    _cam_freeze_frames = int(_ppe_cfg.get("cam_freeze_frames", 60))
    _cam_dark_frames   = int(_ppe_cfg.get("cam_dark_frames",   60))
    _cam_freeze_diff   = float(_ppe_cfg.get("cam_freeze_diff",  0.002))
    _cam_dark_thresh   = float(_ppe_cfg.get("cam_dark_thresh",  0.03))
    _prev_gray:    "cv2.Mat | None" = None
    _freeze_cnt    = 0
    _dark_cnt      = 0
    _cam_status    = "online"

    print("Basladi." + (" ESC = cikis." if display else " Ctrl+C = cikis.") + f" [{mode.upper()} modu]\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if _cam_status != "offline":
                    _notify_camera_status("offline", args.camera_id, args.zone)
                    _cam_status = "offline"
                break
            frame_idx += 1

            # --- Çözünürlük sınırlama ---
            if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
                scale = PIPELINE_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * scale)))

            # --- Kamera donma / karartı ---
            _gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _brightness = cv2.mean(_gray)[0] / 255.0
            if _brightness < _cam_dark_thresh:
                _dark_cnt   += 1
                _freeze_cnt  = 0
            else:
                _dark_cnt = 0
                if _prev_gray is not None:
                    _diff = cv2.absdiff(_gray, _prev_gray)
                    if cv2.mean(_diff)[0] / 255.0 < _cam_freeze_diff:
                        _freeze_cnt += 1
                    else:
                        _freeze_cnt = 0
            _prev_gray = _gray
            _new_status = (
                "dark"   if _dark_cnt   >= _cam_dark_frames   else
                "frozen" if _freeze_cnt >= _cam_freeze_frames  else
                "online"
            )
            if _new_status != _cam_status:
                _notify_camera_status(_new_status, args.camera_id, args.zone)
                _cam_status = _new_status

            # --- Bellek temizliği ---
            if frame_idx % STATES_CLEANUP_EVERY == 0 and seen_stable_pids:
                stale = set(states.keys()) - seen_stable_pids
                for pid in stale:
                    del states[pid]
                if stale:
                    print(f"  [CLEANUP] {len(stale)} kayip stable_pid temizlendi. Kalan: {len(states)}")
                seen_stable_pids.clear()

            # --- Person tracking (her iki modda ortak) ---
            draw_frame = frame.copy()
            fh, fw = frame.shape[:2]

            p_result = person_model.track(
                frame, classes=p_ids, tracker=TRACKER, persist=True,
                imgsz=IMGSZ, conf=PERSON_CONF, device=device, verbose=False,
            )[0]

            persons_with_ppe: list[dict] = []
            boxes = p_result.boxes
            all_persons_frame: list[dict] = []
            stable_map: dict[int, int] = {}

            if boxes is not None and boxes.id is not None:
                all_persons_frame = [
                    {"tid": int(tid), "box": list(map(int, box.tolist()))}
                    for box, tid in zip(boxes.xyxy, boxes.id)
                ]
                stable_map = reattacher.update(all_persons_frame)

            # ════════════════════════════════════════════════════════════════
            # PPE Detection — CROP modu
            # ════════════════════════════════════════════════════════════════
            if mode == "crop" and boxes is not None and boxes.id is not None:
                _do_ppe = (frame_idx % PPE_INFER_EVERY == 0)

                h_cands: list[dict] = []
                v_cands: list[dict] = []
                m_cands: list[dict] = []
                person_coords: dict[int, tuple] = {}

                h_batch: list[tuple] = []
                v_batch: list[tuple] = []
                m_batch: list[tuple] = []

                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id   = int(tid)
                    stable_pid = stable_map.get(track_id, track_id)
                    seen_stable_pids.add(stable_pid)
                    states[stable_pid]["frame_count"] += 1
                    x1, y1, x2, y2 = map(int, box.tolist())
                    person_coords[stable_pid] = (x1, y1, x2, y2)

                    if _do_ppe:
                        hcrop, hox, hoy = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                        if _crop_ok(hcrop) and not is_region_too_small(x1, y1, x2, y2, "helmet"):
                            h_batch.append((stable_pid, track_id, hcrop, hox, hoy))

                        vcrop, vox, voy = crop_ppe(frame, x1, y1, x2, y2, "vest")
                        if _crop_ok(vcrop):
                            v_batch.append((stable_pid, track_id, vcrop, vox, voy))

                        mcrop, mox, moy = crop_ppe(frame, x1, y1, x2, y2, "mask")
                        if _crop_ok(mcrop) and not is_region_too_small(x1, y1, x2, y2, "mask"):
                            m_batch.append((stable_pid, track_id, mcrop, mox, moy))

                if h_batch:
                    h_results = helmet_model.predict(
                        [b[2] for b in h_batch], classes=h_ids, imgsz=IMGSZ,
                        conf=CROP_HELMET_CONF, device=device, verbose=False,
                    )
                    for (stable_pid, track_id, hcrop, hox, hoy), hres in zip(h_batch, h_results):
                        hdets = _collect_dets(helmet_model, hres, h_ids, CROP_HELMET_CONF)
                        if hdets:
                            best = max(hdets, key=lambda d: d["conf"])
                            hbbox_f = list(crop_to_frame(best["bbox"], hox, hoy, fh, fw))
                            vlbl, h_own, h_nb, h_reason = _validate_ppe_scored(
                                hbbox_f, best["label"], track_id, all_persons_frame, "helmet")
                            h_cands.append({
                                "tid": stable_pid, "bbox_f": hbbox_f,
                                "label": vlbl, "raw_label": best["label"],
                                "conf": best["conf"], "own_score": h_own,
                                "neighbor_pen": h_nb, "reason": h_reason,
                            })
                        elif TELEMETRY:
                            h_nb_crop = neighbor_overlap_score(
                                [hox, hoy, hox + hcrop.shape[1], hoy + hcrop.shape[0]],
                                all_persons_frame, track_id)
                            if h_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="helmet",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=h_nb_crop, ambiguous=(h_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                if v_batch:
                    v_results = vest_model.predict(
                        [b[2] for b in v_batch], classes=v_ids, imgsz=IMGSZ,
                        conf=CROP_VEST_CONF, device=device, verbose=False,
                    )
                    for (stable_pid, track_id, vcrop, vox, voy), vres in zip(v_batch, v_results):
                        vdets = _collect_dets(vest_model, vres, v_ids, CROP_VEST_CONF)
                        if vdets:
                            best = max(vdets, key=lambda d: d["conf"])
                            vbbox_f = list(crop_to_frame(best["bbox"], vox, voy, fh, fw))
                            vlbl, v_own, v_nb, v_reason = _validate_ppe_scored(
                                vbbox_f, best["label"], track_id, all_persons_frame, "vest")
                            v_cands.append({
                                "tid": stable_pid, "bbox_f": vbbox_f,
                                "label": vlbl, "raw_label": best["label"],
                                "conf": best["conf"], "own_score": v_own,
                                "neighbor_pen": v_nb, "reason": v_reason,
                            })
                        elif TELEMETRY:
                            v_nb_crop = neighbor_overlap_score(
                                [vox, voy, vox + vcrop.shape[1], voy + vcrop.shape[0]],
                                all_persons_frame, track_id)
                            if v_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="vest",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=v_nb_crop, ambiguous=(v_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                if m_batch:
                    m_results = mask_model.predict(
                        [b[2] for b in m_batch], classes=m_ids, imgsz=CROP_MASK_IMGSZ,
                        conf=CROP_MASK_CONF, device=device, verbose=False,
                    )
                    for (stable_pid, track_id, mcrop, mox, moy), mres in zip(m_batch, m_results):
                        mdets = _collect_dets(mask_model, mres, m_ids, CROP_MASK_CONF)
                        if mdets:
                            best = max(mdets, key=lambda d: d["conf"])
                            mbbox_f = list(crop_to_frame(best["bbox"], mox, moy, fh, fw))
                            vlbl, m_own, m_nb, m_reason = _validate_ppe_scored(
                                mbbox_f, best["label"], track_id, all_persons_frame, "mask")
                            m_cands.append({
                                "tid": stable_pid, "bbox_f": mbbox_f,
                                "label": vlbl, "raw_label": best["label"],
                                "conf": best["conf"], "own_score": m_own,
                                "neighbor_pen": m_nb, "reason": m_reason,
                            })
                        elif TELEMETRY:
                            m_nb_crop = neighbor_overlap_score(
                                [mox, moy, mox + mcrop.shape[1], moy + mcrop.shape[0]],
                                all_persons_frame, track_id)
                            if m_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="mask",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=m_nb_crop, ambiguous=(m_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                h_assigned = _global_assign_ppe(h_cands)
                v_assigned = _global_assign_ppe(v_cands)
                m_assigned = _global_assign_ppe(m_cands)

                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id   = int(tid)
                    stable_pid = stable_map.get(track_id, track_id)
                    x1, y1, x2, y2 = person_coords[stable_pid]

                    hcand  = h_assigned.get(stable_pid)
                    hlabel = hcand["label"]  if hcand else "unknown"
                    hconf  = hcand["conf"]   if hcand else 0.0
                    hbbox  = hcand["bbox_f"] if hcand else None
                    if _do_ppe:
                        if TELEMETRY and hcand:
                            _log_ppe_decision(PPEDecision(
                                frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="helmet",
                                raw_label=hcand["raw_label"], conf=hcand["conf"],
                                own_score=hcand["own_score"], neighbor_pen=hcand["neighbor_pen"],
                                ambiguous=False, accepted=True, reason=hcand["reason"],
                            ))
                        states[stable_pid]["hardhat"].append(hlabel)
                    hvote = vote(states[stable_pid]["hardhat"], min_known=3)

                    vcand  = v_assigned.get(stable_pid)
                    vlabel = vcand["label"]  if vcand else "unknown"
                    vconf  = vcand["conf"]   if vcand else 0.0
                    vbbox  = vcand["bbox_f"] if vcand else None
                    if _do_ppe:
                        if TELEMETRY and vcand:
                            _log_ppe_decision(PPEDecision(
                                frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="vest",
                                raw_label=vcand["raw_label"], conf=vcand["conf"],
                                own_score=vcand["own_score"], neighbor_pen=vcand["neighbor_pen"],
                                ambiguous=False, accepted=True, reason=vcand["reason"],
                            ))
                        states[stable_pid]["vest"].append(vlabel)
                    vvote = vote(states[stable_pid]["vest"], min_known=2)

                    mcand  = m_assigned.get(stable_pid)
                    mlabel = mcand["label"]  if mcand else "unknown"
                    mconf  = mcand["conf"]   if mcand else 0.0
                    mbbox  = mcand["bbox_f"] if mcand else None
                    if _do_ppe:
                        if TELEMETRY and mcand:
                            _log_ppe_decision(PPEDecision(
                                frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="mask",
                                raw_label=mcand["raw_label"], conf=mcand["conf"],
                                own_score=mcand["own_score"], neighbor_pen=mcand["neighbor_pen"],
                                ambiguous=False, accepted=True, reason=mcand["reason"],
                            ))
                        states[stable_pid]["mask"].append(mlabel)
                    mvote = vote(states[stable_pid]["mask"], min_known=1)

                    if states[stable_pid]["frame_count"] < MIN_TRACK_FRAMES:
                        draw_box(draw_frame, x1, y1, x2, y2, "...", COLOR_UNKNOWN)
                        continue

                    color, viols = compliance_color(hvote, vvote, mvote)
                    persons_with_ppe.append({
                        "track_id":      stable_pid,
                        "violations":    viols,
                        "helmet_status": "ok" if hvote == "Hardhat" else "violation" if hvote == "NO-Hardhat" else "unknown",
                        "vest_status":   "ok" if vvote == "Safety Vest" else "violation" if vvote == "NO-Safety Vest" else "unknown",
                        "mask_status":   "ok" if mvote == "Mask" else "violation" if mvote == "NO-Mask" else "unknown",
                        "helmet_conf":   round(hconf, 2),
                        "vest_conf":     round(vconf, 2),
                        "mask_conf":     round(mconf, 2),
                    })
                    draw_box(draw_frame, x1, y1, x2, y2, f"ID{stable_pid}", color)
                    ppe_items = [
                        (hbbox, hvote, hconf, "H",
                         COLOR_OK if hvote == "Hardhat" else COLOR_DANGER if hvote == "NO-Hardhat" else COLOR_UNKNOWN),
                        (vbbox, vvote, vconf, "V",
                         COLOR_OK if vvote == "Safety Vest" else COLOR_DANGER if vvote == "NO-Safety Vest" else COLOR_UNKNOWN),
                        (mbbox, mvote, mconf, "M",
                         COLOR_OK if mvote == "Mask" else COLOR_WARN if mvote == "NO-Mask" else COLOR_UNKNOWN),
                    ]
                    for bbox, vote_label, conf, tag, c in ppe_items:
                        if bbox is None or vote_label == "unknown":
                            continue
                        draw_ppe_box(draw_frame, bbox[0], bbox[1], bbox[2], bbox[3],
                                     f"{vote_label[:8]}({conf:.2f})", c, tag)

            # ════════════════════════════════════════════════════════════════
            # PPE Detection — SCENE modu
            # ════════════════════════════════════════════════════════════════
            elif mode == "scene" and boxes is not None and boxes.id is not None:
                _do_scene_ppe = (frame_idx % SCENE_PPE_INFER_EVERY == 0)

                if _do_scene_ppe:
                    scene_helmet_dets = _scene_dets(helmet_model, frame, h_ids_set, SCENE_HELMET_CONF, device)
                    scene_vest_dets   = _scene_dets(vest_model,   frame, v_ids_set, SCENE_VEST_CONF,   device)
                    scene_mask_dets   = _scene_dets(mask_model,   frame, m_ids_set, SCENE_MASK_CONF,   device)
                else:
                    scene_helmet_dets = []
                    scene_vest_dets   = []
                    scene_mask_dets   = []

                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id   = int(tid)
                    stable_pid = stable_map.get(track_id, track_id)
                    seen_stable_pids.add(stable_pid)
                    states[stable_pid]["frame_count"] += 1
                    x1, y1, x2, y2 = map(int, box.tolist())
                    person_box = [x1, y1, x2, y2]

                    if _do_scene_ppe:
                        hlabel, hconf, hbbox = _best_scene(scene_helmet_dets, person_box)
                        vlabel, vconf, vbbox = _best_scene(scene_vest_dets,   person_box)
                        mlabel, mconf, mbbox = _best_scene(scene_mask_dets,   person_box)
                        states[stable_pid]["hardhat"].append(hlabel)
                        states[stable_pid]["vest"].append(vlabel)
                        states[stable_pid]["mask"].append(mlabel)
                    else:
                        hconf = vconf = mconf = 0.0
                        hbbox = vbbox = mbbox = None

                    hvote = vote(states[stable_pid]["hardhat"], min_known=2)
                    vvote = vote(states[stable_pid]["vest"],    min_known=2)
                    mvote = vote(states[stable_pid]["mask"],    min_known=1)

                    color, viols = compliance_color(hvote, vvote, mvote)
                    persons_with_ppe.append({
                        "track_id":      stable_pid,
                        "violations":    viols,
                        "helmet_status": "ok" if hvote == "Hardhat" else "violation" if hvote == "NO-Hardhat" else "unknown",
                        "vest_status":   "ok" if vvote == "Safety Vest" else "violation" if vvote == "NO-Safety Vest" else "unknown",
                        "mask_status":   "ok" if mvote == "Mask" else "violation" if mvote == "NO-Mask" else "unknown",
                        "helmet_conf":   round(hconf, 2),
                        "vest_conf":     round(vconf, 2),
                        "mask_conf":     round(mconf, 2),
                    })
                    draw_box(draw_frame, x1, y1, x2, y2, f"ID{stable_pid}", color)
                    ppe_items = [
                        (hbbox, hvote, hconf, "H",
                         COLOR_OK if hvote == "Hardhat" else COLOR_DANGER if hvote == "NO-Hardhat" else COLOR_UNKNOWN),
                        (vbbox, vvote, vconf, "V",
                         COLOR_OK if vvote == "Safety Vest" else COLOR_DANGER if vvote == "NO-Safety Vest" else COLOR_UNKNOWN),
                        (mbbox, mvote, mconf, "M",
                         COLOR_OK if mvote == "Mask" else COLOR_WARN if mvote == "NO-Mask" else COLOR_UNKNOWN),
                    ]
                    for bbox, vote_label, conf, tag, c in ppe_items:
                        if bbox is None or vote_label == "unknown":
                            continue
                        bx1, by1, bx2, by2 = [int(v) for v in bbox]
                        draw_ppe_box(draw_frame, bx1, by1, bx2, by2,
                                     f"{vote_label[:8]}({conf:.2f})", c, tag)

            # --- Fire / smoke detection (throttled, her iki modda aynı) ---
            if _use_fire and frame_idx % FIRE_INFER_EVERY == 0:
                fire_res = fire_model.predict(
                    frame, imgsz=IMGSZ, conf=FIRE_CONF, device=device, verbose=False,
                )[0]
                _fire_raw       = False
                _fire_conf_max  = 0.0
                _smoke_raw      = False
                _smoke_conf_max = 0.0
                _frame_area     = frame.shape[0] * frame.shape[1] or 1
                _max_fire_area  = 0.0
                _max_smoke_area = 0.0

                if fire_res.boxes:
                    for box in fire_res.boxes:
                        cid  = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = fire_model.names[cid]
                        if name == "fire":
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            area_ratio = ((x2 - x1) * (y2 - y1)) / _frame_area
                            _max_fire_area = max(_max_fire_area, area_ratio)
                            _fire_conf_max = max(_fire_conf_max, conf)
                        elif name == "smoke":
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            area_ratio = ((x2 - x1) * (y2 - y1)) / _frame_area
                            _max_smoke_area = max(_max_smoke_area, area_ratio)
                            _smoke_conf_max = max(_smoke_conf_max, conf)

                # Fire filtresi: alan veya büyüme
                _fire_area_history.append(_max_fire_area)
                _is_large   = _max_fire_area >= _fire_min_area
                _is_growing = False
                if _max_fire_area > 0 and len(_fire_area_history) >= _fire_growth_window:
                    half      = _fire_growth_window // 2
                    hist      = list(_fire_area_history)
                    older_avg = sum(hist[-_fire_growth_window:-half]) / half
                    newer_avg = sum(hist[-half:]) / half
                    _is_growing = older_avg > 0 and (newer_avg / older_avg) >= _fire_growth_factor
                if _fire_conf_max > 0 and (_is_large or _is_growing):
                    _fire_raw = True

                # Smoke filtresi: aynı alan ve büyüme kriterleri
                _smoke_area_history.append(_max_smoke_area)
                _smoke_is_large   = _max_smoke_area >= _fire_min_area
                _smoke_is_growing = False
                if _max_smoke_area > 0 and len(_smoke_area_history) >= _fire_growth_window:
                    half      = _fire_growth_window // 2
                    hist      = list(_smoke_area_history)
                    older_avg = sum(hist[-_fire_growth_window:-half]) / half
                    newer_avg = sum(hist[-half:]) / half
                    _smoke_is_growing = older_avg > 0 and (newer_avg / older_avg) >= _fire_growth_factor
                if _smoke_conf_max > 0 and (_smoke_is_large or _smoke_is_growing):
                    _smoke_raw = True

            if _fire_raw or _smoke_raw:
                label = "FIRE" if _fire_raw else "SMOKE"
                cv2.putText(draw_frame, f"{label} DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FIRE, 3)

            # --- Event state machine (ortak) ---
            event_info = event_manager.process_frame(persons_with_ppe, _fire_raw or _smoke_raw)

            if event_info["should_save"] and event_info.get("event_id"):
                event_count += 1
                save_event(
                    event_info, draw_frame, results_dir, persons_with_ppe,
                    fire_conf=_fire_conf_max,
                    smoke_detected=_smoke_raw,
                    smoke_conf=_smoke_conf_max,
                    camera_id=args.camera_id,
                    zone=args.zone,
                )

            if event_info["event_status"] == "closed" and event_info.get("event_id"):
                threading.Thread(
                    target=_close_event,
                    args=(event_info["event_id"], event_info.get("repeat_count"), event_info.get("duration_sec")),
                    daemon=True,
                ).start()

            if display:
                draw_hud(
                    draw_frame,
                    event_info.get("event_id"),
                    event_info["event_status"],
                    event_info.get("repeat_count", 0),
                    event_info.get("person_violations", []),
                )
                cv2.imshow("Factory Safety", draw_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frame_idx % 60 == 0:
                print(f"  Frame {frame_idx} | Events: {event_count} | Status: {event_info['event_status']}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        _notify_camera_status("online", args.camera_id, args.zone)
        print(f"\nToplam frame: {frame_idx} | Kaydedilen event: {event_count}")
        print("Sonuclar: results/")
        if event_manager._active is not None:
            ev = event_manager._active
            _close_event(ev.event_id, ev.repeat_count, ev.duration_sec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Factory Safety — birleşik PPE pipeline")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video",     help="Video dosyasi yolu")
    src.add_argument("--camera",    default=0, help="Kamera indeksi (varsayilan: 0)")
    parser.add_argument("--mode",       default="crop", choices=["crop", "scene"],
                        help="Detection modu: crop-based veya scene-based (varsayilan: crop)")
    parser.add_argument("--device",     default=_DEVICE, help="cuda device veya cpu")
    parser.add_argument("--display",    action="store_true", help="OpenCV penceresi goster")
    parser.add_argument("--camera-id",  default=None, help="Kamera kimligi (orn: cam_01)")
    parser.add_argument("--zone",       default=None, help="Kamera bolgesi (orn: Uretim Hatti A)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
