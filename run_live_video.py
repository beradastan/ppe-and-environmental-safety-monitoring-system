# -*- coding: utf-8 -*-
"""
run_live_video.py
=================
Crop-tabanlı PPE pipeline — nihai modeller:
  Helmet : crophelmet_agent_final_best.pt  conf=0.20  (üst %55 crop)
  Vest   : vest_agent_final_best.pt        conf=0.30  (orta %15-85 crop)
  Mask   : cropmask_agent_final_best.pt    conf=0.25  (baş %35 crop)

Kullanim:
    python run_live_video.py                          # headless (arka plan)
    python run_live_video.py --display                # pencereli
    python run_live_video.py --video test/nohat_test.mp4
    python run_live_video.py --camera 1 --display
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
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
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# LLM entegrasyonu
# ---------------------------------------------------------------------------

def _build_alarm_text(event_type: str, persons: list[dict]) -> str:
    """Template-based kısa alarm metni — konsol/bildirim için."""
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


def _load_llm_cfg() -> dict:
    try:
        with open("config.yaml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("llm", {})
    except Exception:
        return {}


_LLM_SYSTEM = (
    "You are a factory safety AI. Write exactly 1 line in Turkish using ONLY the given data. "
    "Do NOT mention time, seconds, risk, danger, recommendations, or what people ARE wearing. "
    "Copy the Output pattern from the matching example below.\n\n"
    "Input: Olay tipi: Yangin/duman\nSahne: yangin tespit edildi\n"
    "Output: Sahada yangın tespit edildi.\n\n"
    "Input: Olay tipi: Yangin/duman\nSahne: duman tespit edildi\n"
    "Output: Sahada duman tespit edildi.\n\n"
    "Input: Olay tipi: KKD ihlali\nKisi #2: baret=YOK\n"
    "Output: Kişi #2 baret takmıyor.\n\n"
    "Input: Olay tipi: KKD ihlali\nKisi #9: yelek=YOK\n"
    "Output: Kişi #9 yelek takmıyor.\n\n"
    "Input: Olay tipi: KKD ihlali\nKisi #5: baret=YOK, yelek=YOK, maske=YOK\n"
    "Output: Kişi #5 baret, yelek ve maske takmıyor.\n\n"
    "Input: Olay tipi: KKD ihlali (2 kisi)\nKisi #1: baret=YOK, yelek=YOK\nKisi #3: yelek=YOK\n"
    "Output: 2 kişi ihlal yapıyor; kişi #1 baret ve yelek takmıyor, kişi #3 yelek takmıyor.\n\n"
    "Input: Olay tipi: KKD ihlali (3 kisi)\nKisi #1: baret=YOK, yelek=YOK\nKisi #4: yelek=YOK\nKisi #6: baret=YOK\n"
    "Output: 3 kişi ihlal yapıyor; kişi #1 baret ve yelek takmıyor, kişi #4 yelek takmıyor, kişi #6 baret takmıyor.\n\n"
    "Input: Olay tipi: Coklu tehlike\nSahne: yangin tespit edildi\nKisi #4: baret=YOK\n"
    "Output: Sahada yangın tespit edildi; kişi #4 baret takmıyor.\n\n"
    "Input: Olay tipi: Coklu tehlike\nSahne: duman tespit edildi\nKisi #3: yelek=YOK\n"
    "Output: Sahada duman tespit edildi; kişi #3 yelek takmıyor.\n\n"
    "Input: Olay tipi: Coklu tehlike (2 kisi)\nSahne: duman tespit edildi\nKisi #2: yelek=YOK\nKisi #4: baret=YOK\n"
    "Output: Sahada duman tespit edildi; 2 kişi ihlal yapıyor; kişi #2 yelek takmıyor, kişi #4 baret takmıyor.\n\n"
    "Now write only the Output line for the given input. Nothing else."
)


def _build_llm_prompt(payload: dict) -> str:
    event_type = payload.get("event_type", "ppe_violation")
    persons    = payload.get("persons", [])
    scene      = payload.get("scene", {})

    STATUS_TR = {"ok": "var", "violation": "YOK"}

    person_lines = []
    for p in persons:
        tid   = p["track_id"]
        parts = []
        for field, label in [("helmet_status", "baret"), ("vest_status", "yelek"), ("mask_status", "maske")]:
            st = p.get(field, "unknown")
            if st in STATUS_TR:
                parts.append(f"{label}={STATUS_TR[st]}")
        if parts:
            person_lines.append(f"Kisi #{tid}: {', '.join(parts)}")

    violator_count = sum(1 for line in person_lines if "YOK" in line)

    scene_parts = []
    if scene.get("fire_detected"):
        scene_parts.append("yangin tespit edildi")
    if scene.get("smoke_detected"):
        scene_parts.append("duman tespit edildi")
    scene_str = "Sahne: " + ", ".join(scene_parts) + "\n" if scene_parts else ""

    if event_type == "fire_detected":
        type_tr = "Yangin/duman"
    elif event_type == "ppe_violation":
        type_tr = f"KKD ihlali ({violator_count} kisi)" if violator_count > 0 else "KKD ihlali"
    else:  # multi_hazard — always include count so model echoes it
        type_tr = f"Coklu tehlike ({violator_count} kisi)"

    persons_section = "\n".join(person_lines) + "\n" if person_lines else ""
    return (
        f"Input: Olay tipi: {type_tr}\n"
        f"{scene_str}"
        f"{persons_section}"
        "Output:"
    )


def _call_ollama(prompt: str, cfg: dict, system=None):
    body = {
        "model":       cfg.get("model", "mistral"),
        "prompt":      prompt,
        "temperature": cfg.get("temperature", 0.1),
        "stream":      False,
    }
    if system:
        body["system"] = system
    try:
        resp = requests.post(
            f"{cfg.get('base_url', 'http://localhost:11434')}/api/generate",
            json=body,
            timeout=cfg.get("timeout", 120),
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            raw = raw.removeprefix("Output:").strip(" \"'\n")
            first_line = raw.split("\n")[0].strip().strip("\"'")
            if first_line.endswith("."):
                return first_line
            if "." in first_line:
                return first_line[:first_line.rindex(".") + 1]
            return first_line or None
    except Exception as e:
        print(f"  [LLM] Hata: {e}")
    return None


def _llm_report_async(payload: dict, json_path: Path, cfg: dict) -> None:
    prompt = _build_llm_prompt(payload)
    report = _call_ollama(prompt, cfg, system=_LLM_SYSTEM)

    if report:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            data["llm_report"] = report
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [LLM] Rapor yazildi: {json_path.name} — {report}")
        except Exception as e:
            print(f"  [LLM] JSON yazma hatasi: {e}")
    else:
        print("  [LLM] Yanit alinamadi.")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

HELMET_MODEL_PATH = "models/bera/crophelmet_agent_final_best.pt"
VEST_MODEL_PATH   = "models/bera/vest_agent_final_best.pt"
MASK_MODEL_PATH   = "models/bera/cropmask_agent_final_best.pt"
FIRE_MODEL_PATH   = "models/bera/fire_smoke_other_agent_final_best.pt"
PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"

HELMET_PAD   = 0.80
VEST_PAD     = 0.60
MASK_PAD     = 0.80
HELMET_CONF  = 0.15
VEST_CONF    = 0.30
MASK_CONF    = 0.10
MASK_IMGSZ   = 1280  # maske için yüksek çözünürlük: küçük yüz bölgelerinde recall artışı
FIRE_CONF    = 0.75  # yükseltildi: kum/yelek false positive bastırma (eski: 0.50)
PERSON_CONF  = 0.25
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"   # proje kökündeki özel config (track_buffer=60)
TEMPORAL_WIN      = 20   # artırıldı: daha kararlı oy (eski: 10)
STATES_CLEANUP_EVERY = 300  # her N frame'de kayıp track temizliği

MIN_CROP_PX      = 40   # crop bu boyutun altındaysa model çağrılmaz (kenar kişi)
MIN_TRACK_FRAMES     = 10    # bu kadar frame görülmemiş track'lar ghost — event'e dahil edilmez
# PPE tipine göre geometrik eşikler
# mask daha hoşgörülü: yüz bölgesi küçük, biraz komşu üstüne gelse yine de reddedilmemeli
# vest daha sıkı: büyük nesne, komşu üstüne gelmesi nadiren meşru
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
MIN_HEAD_PX  = 30   # kafa/maske anatomik bölgesi bu genişliğin altındaysa çıkarım yapma
MIN_TORSO_PX = 50   # gövde/yelek bölgesi için minimum genişlik
TELEMETRY    = False  # PPE karar telemetrisi (True → her kararda konsola log yazar)

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
MASK_CLASSES   = ["Mask", "NO-Mask"]

# Ekran renkleri (BGR)
COLOR_OK      = (0, 200, 0)
COLOR_WARN    = (0, 100, 255)
COLOR_DANGER  = (0, 0, 230)
COLOR_UNKNOWN = (0, 200, 255)
COLOR_FIRE    = (0, 60, 255)

# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def class_ids(model: YOLO, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [name_to_id[n] for n in names]


def crop_pad(frame, x1, y1, x2, y2, pad: float):
    h, w = frame.shape[:2]
    pw = int(max(1, x2 - x1) * pad)
    ph = int(max(1, y2 - y1) * pad)
    return frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]


def _crop_ok(crop) -> bool:
    """Crop geçerliyse True — çok küçük veya boş croplar atlanır."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX


def crop_ppe(frame, x1, y1, x2, y2, ppe: str):
    """
    Her PPE tipi için optimize edilmiş bölge crop'u.
    Dar padding → komşu kişi kirliliği önlenir.
    Döndürür: (crop_img, origin_x, origin_y)
    """
    fh, fw = frame.shape[:2]
    pw = x2 - x1
    ph = y2 - y1

    if ppe == "helmet":
        # Kişinin üst %40'ı + yanlara %10 + yukarı %15 (elde tutulan bareti dışarıda bırakmak için daraltıldı)
        cx1 = max(0, x1 - int(pw * 0.10))
        cy1 = max(0, y1 - int(ph * 0.15))
        cx2 = min(fw, x2 + int(pw * 0.10))
        cy2 = min(fh, y1 + int(ph * 0.40))
    elif ppe == "vest":
        # Kişinin %10-%90 yükseklik aralığı + yanlara %15
        cx1 = max(0, x1 - int(pw * 0.15))
        cy1 = max(0, y1 + int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.90))
    else:  # mask
        # Sadece baş bölgesi: üst %45 + yanlara %15 + yukarı %10
        cx1 = max(0, x1 - int(pw * 0.15))
        cy1 = max(0, y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.45))

    crop = frame[cy1:cy2, cx1:cx2]
    return crop, cx1, cy1


def best_det(model: YOLO, result, allowed_ids: list[int], min_conf: float):
    """En yüksek confidence tespiti döndür: (label, conf, bbox_in_crop | None)"""
    best = None  # (label, conf, bbox)
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        label = str(model.names[cid])
        bbox  = box.xyxy[0].tolist()
        if best is None or conf > best[1]:
            best = (label, conf, bbox)
    return best if best else ("unknown", 0.0, None)


def crop_to_frame(bbox, ox: int, oy: int, fh: int, fw: int):
    """Crop içi koordinatları frame koordinatlarına çevir (origin-tabanlı)."""
    dx1, dy1, dx2, dy2 = bbox
    return (
        min(fw - 1, int(ox + dx1)),
        min(fh - 1, int(oy + dy1)),
        min(fw - 1, int(ox + dx2)),
        min(fh - 1, int(oy + dy2)),
    )


# ---------------------------------------------------------------------------
# Geometric association layer
# ---------------------------------------------------------------------------

def _containment(inner: list, outer: list) -> float:
    """inner kutusunun ne kadarı outer içinde (inner alanına oranla)."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def anatomical_region(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> list:
    """Kişi bbox'ından PPE tipine göre anatomik bölge döndür (frame koordinatı)."""
    pw, ph = x2 - x1, y2 - y1
    if ppe_type == "helmet":
        return [x1 + int(pw * 0.05), y1 - int(ph * 0.10),
                x2 - int(pw * 0.05), y1 + int(ph * 0.35)]
    elif ppe_type == "mask":
        return [x1 + int(pw * 0.15), y1,
                x2 - int(pw * 0.15), y1 + int(ph * 0.28)]
    else:  # vest
        return [x1, y1 + int(ph * 0.15),
                x2, y1 + int(ph * 0.85)]


def _collect_dets(model: YOLO, result, allowed_ids: list[int], min_conf: float) -> list[dict]:
    """Tüm geçerli tespitleri döndür: [{label, conf, bbox_in_crop}, ...]"""
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
    """
    PPE bbox'ını geometrik olarak değerlendirir.

    Döner: (final_label, own_score, max_neighbor_score, reason)
      reason ∈ {"accepted", "rejected", "ambiguous", "no_target"}
      final_label = label  →  accepted
      final_label = "unknown" →  rejected | ambiguous | no_target
    """
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


def validate_ppe(
    ppe_bbox_frame: list,
    label: str,
    target_tid: int,
    all_persons_frame: list[dict],
    ppe_type: str,
) -> str:
    """PPE bbox'ını geometrik olarak doğrula; "unknown" ya da onaylı label döndürür."""
    return _validate_ppe_scored(ppe_bbox_frame, label, target_tid, all_persons_frame, ppe_type)[0]


def neighbor_overlap_score(
    crop_box: list,
    all_persons_frame: list[dict],
    target_tid: int,
) -> float:
    """Crop bölgesine en çok giren komşunun overlap oranını döndür (0.0–1.0).
    Sıfır → komşu yok; 1.0 → crop tamamen komşunun içinde."""
    max_score = 0.0
    for p in all_persons_frame:
        if p["tid"] == target_tid:
            continue
        score = _containment(p["box"], crop_box)
        if score > max_score:
            max_score = score
    return max_score


def crop_has_neighbor(
    crop_box: list,
    all_persons_frame: list[dict],
    target_tid: int,
    thresh: float = 0.25,
) -> bool:
    """Crop bölgesine komşu kişi giriyorsa True — neighbor_overlap_score üstüne thin wrapper."""
    return neighbor_overlap_score(crop_box, all_persons_frame, target_tid) > thresh


def _iou(a: list, b: list) -> float:
    """İki bbox arasında Intersection over Union."""
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
    """
    Greedy one-to-one: aynı fiziksel PPE bbox birden fazla kişiye gitmesin.
    Adaylar conf × own_score'a göre azalan sırada işlenir.
    Döner: {track_id: cand}  — sadece "accepted" adaylar.
    """
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
    """Anatomik bölge çok küçükse True — uzak/küçük kişilerde gereksiz çıkarımı önler."""
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


def draw_ppe_box(frame, x1, y1, x2, y2, label: str, color, tag: str):
    """PPE tespiti için ince, küçük etiketli kutu."""
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
    h_ok   = hvote == "Hardhat"
    v_ok   = vvote == "Safety Vest"
    m_ok   = mvote == "Mask"
    h_miss = hvote == "NO-Hardhat"
    v_miss = vvote == "NO-Safety Vest"
    m_miss = mvote == "NO-Mask"
    viols  = []
    if h_miss:
        viols.append("no_helmet")
    if v_miss:
        viols.append("no_vest")
    if m_miss:
        viols.append("no_mask")
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


def draw_hud(frame, event_id, status, repeat, viols_per_person):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 200) if status in ("new", "active") else (0, 200, 0)
    cv2.putText(frame, f"EVENT: {event_id or 'N/A'} [{status.upper()}]",
                (10, 30), font, 0.7, color, 2)
    cv2.putText(frame, f"Repeat: {repeat}  Active violations: {len(viols_per_person)}",
                (10, 58), font, 0.6, color, 2)


# ---------------------------------------------------------------------------
# Event kayit
# ---------------------------------------------------------------------------

def save_event(
    event_info:       dict,
    frame,
    results_dir:      Path,
    persons_snapshot: list[dict],
    fire_conf:        float = 0.0,
    smoke_detected:   bool  = False,
    smoke_conf:       float = 0.0,
) -> None:
    event_id     = event_info["event_id"]
    event_status = event_info["event_status"]
    base_sig     = event_info.get("signature", {})

    event_dir = results_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    json_path = event_dir / f"{event_id}_new.json"
    img_path  = event_dir / f"{event_id}_new.jpg"

    # event_type — signature'dan türet
    has_ppe   = base_sig.get("helmet_violation") or base_sig.get("vest_violation") or base_sig.get("mask_violation")
    has_fire  = base_sig.get("fire_detected", False)
    if has_ppe and has_fire:
        event_type = "multi_hazard"
    elif has_fire:
        event_type = "fire_detected"
    else:
        event_type = "ppe_violation"

    # duration_sec per-person — event_manager'dan gelir, snapshot ile birleştir
    dur_map = {p["track_id"]: p.get("duration_sec", 0.0)
               for p in event_info.get("person_violations", [])}

    persons_detail = [
        {
            "track_id":     p["track_id"],
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

    alarm_text = _build_alarm_text(event_type, persons_detail)

    payload = {
        "event_id":      event_id,
        "event_type":    event_type,
        "event_status":  event_status,
        "timestamp":     datetime.now().isoformat(),
        "duration_sec":  event_info.get("duration_sec", 0.0),
        "change_reason": event_info.get("change_reason", ""),
        "persons":       persons_detail,
        "scene":         scene,
        "signature":     base_sig,
        "alarm_text":    alarm_text,
        "llm_report":    None,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cv2.imwrite(str(img_path), frame)
    print(f"  [KAYIT] {event_id}/new  alarm: {alarm_text}")

    llm_cfg = _load_llm_cfg()
    if llm_cfg.get("enabled", False):
        t = threading.Thread(
            target=_llm_report_async,
            args=(payload, json_path, llm_cfg),
            daemon=True,
        )
        t.start()

    try:
        import winsound
        winsound.Beep(1000, 400)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def run(args):
    device = args.device

    print("Modeller yukleniyor...")
    person_model = YOLO(PERSON_MODEL_PATH)
    helmet_model = YOLO(HELMET_MODEL_PATH)
    vest_model   = YOLO(VEST_MODEL_PATH)
    mask_model   = YOLO(MASK_MODEL_PATH)
    fire_model   = YOLO(FIRE_MODEL_PATH)
    print("  person, helmet, vest, mask, fire modelleri hazir.")

    p_ids = class_ids(person_model, ["person"])
    h_ids = class_ids(helmet_model, HELMET_CLASSES)
    v_ids = class_ids(vest_model,   VEST_CLASSES)
    m_ids = class_ids(mask_model,   MASK_CLASSES)

    from event_manager import PersonEventManager
    from track_reattacher import TrackReattacher
    event_manager = PersonEventManager(
        new_confirm_sec=3.0,
        resolved_confirm_sec=5.0,
        fire_confirm_frames=20,   # artırıldı: false positive bastırma (eski: 2)
        fire_clear_frames=10,
    )
    reattacher = TrackReattacher()

    states = defaultdict(lambda: {
        "hardhat":     deque(maxlen=TEMPORAL_WIN),
        "vest":        deque(maxlen=TEMPORAL_WIN),
        "mask":        deque(maxlen=TEMPORAL_WIN),
        "frame_count": 0,
    })

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Mevcut event sayısından devam et — yeniden başlatmada üzerine yazma
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
    seen_stable_pids: set[int] = set()   # bellek temizliği için (stable_pid bazlı)

    print("Basladi." + (" ESC = cikis." if display else " Ctrl+C = cikis.") + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # --- Kayıp track temizliği (bellek sızıntısı önleme) ---
            if frame_idx % STATES_CLEANUP_EVERY == 0 and seen_stable_pids:
                stale = set(states.keys()) - seen_stable_pids
                for pid in stale:
                    del states[pid]
                if stale:
                    print(f"  [CLEANUP] {len(stale)} kayip stable_pid temizlendi. Kalan: {len(states)}")
                seen_stable_pids.clear()

            # --- Person tracking ---
            p_result = person_model.track(
                frame, classes=p_ids, tracker=TRACKER, persist=True,
                imgsz=IMGSZ, conf=PERSON_CONF, device=device, verbose=False,
            )[0]

            persons_with_ppe: list[dict] = []
            draw_frame = frame.copy() if display else frame
            fh, fw = frame.shape[:2]

            boxes = p_result.boxes
            # Tüm kişileri geometrik doğrulama için önceden topla
            all_persons_frame: list[dict] = []
            stable_map: dict[int, int] = {}
            if boxes is not None and boxes.id is not None:
                all_persons_frame = [
                    {"tid": int(tid), "box": list(map(int, box.tolist()))}
                    for box, tid in zip(boxes.xyxy, boxes.id)
                ]
                # stable_pid: kısa oklüzyon sonrası aynı kişi yeni raw_tid alsa bile
                # state ve event_manager aynı kimliği görür
                stable_map = reattacher.update(all_persons_frame)

            if boxes is not None and boxes.id is not None:
                # ── Faz 1: çıkarım + aday toplama ───────────────────────────────
                h_cands: list[dict] = []
                v_cands: list[dict] = []
                m_cands: list[dict] = []
                person_coords: dict[int, tuple] = {}

                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id  = int(tid)
                    stable_pid = stable_map.get(track_id, track_id)
                    seen_stable_pids.add(stable_pid)
                    states[stable_pid]["frame_count"] += 1
                    x1, y1, x2, y2 = map(int, box.tolist())
                    person_coords[stable_pid] = (x1, y1, x2, y2)

                    # Helmet
                    hcrop, hox, hoy = crop_ppe(frame, x1, y1, x2, y2, "helmet")
                    if _crop_ok(hcrop) and not is_region_too_small(x1, y1, x2, y2, "helmet"):
                        hres = helmet_model.predict(
                            hcrop, classes=h_ids, imgsz=IMGSZ,
                            conf=HELMET_CONF, device=device, verbose=False,
                        )[0]
                        hdets = _collect_dets(helmet_model, hres, h_ids, HELMET_CONF)
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
                        else:
                            h_nb_crop = neighbor_overlap_score(
                                [hox, hoy, hox + hcrop.shape[1], hoy + hcrop.shape[0]],
                                all_persons_frame, track_id)
                            if TELEMETRY and h_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="helmet",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=h_nb_crop, ambiguous=(h_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                    # Vest
                    vcrop, vox, voy = crop_ppe(frame, x1, y1, x2, y2, "vest")
                    if _crop_ok(vcrop):
                        vres = vest_model.predict(
                            vcrop, classes=v_ids, imgsz=IMGSZ,
                            conf=VEST_CONF, device=device, verbose=False,
                        )[0]
                        vdets = _collect_dets(vest_model, vres, v_ids, VEST_CONF)
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
                        else:
                            v_nb_crop = neighbor_overlap_score(
                                [vox, voy, vox + vcrop.shape[1], voy + vcrop.shape[0]],
                                all_persons_frame, track_id)
                            if TELEMETRY and v_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="vest",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=v_nb_crop, ambiguous=(v_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                    # Mask
                    mcrop, mox, moy = crop_ppe(frame, x1, y1, x2, y2, "mask")
                    if _crop_ok(mcrop) and not is_region_too_small(x1, y1, x2, y2, "mask"):
                        mres = mask_model.predict(
                            mcrop, classes=m_ids, imgsz=MASK_IMGSZ,
                            conf=MASK_CONF, device=device, verbose=False,
                        )[0]
                        mdets = _collect_dets(mask_model, mres, m_ids, MASK_CONF)
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
                        else:
                            m_nb_crop = neighbor_overlap_score(
                                [mox, moy, mox + mcrop.shape[1], moy + mcrop.shape[0]],
                                all_persons_frame, track_id)
                            if TELEMETRY and m_nb_crop > 0.10:
                                _log_ppe_decision(PPEDecision(
                                    frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="mask",
                                    raw_label="no_det", conf=0.0, own_score=0.0,
                                    neighbor_pen=m_nb_crop, ambiguous=(m_nb_crop > 0.25),
                                    accepted=False, reason="no_det",
                                ))

                # ── Faz 2: global one-to-one atama ──────────────────────────────
                h_assigned = _global_assign_ppe(h_cands)
                v_assigned = _global_assign_ppe(v_cands)
                m_assigned = _global_assign_ppe(m_cands)

                # ── Faz 3: oy güncelleme + persons_with_ppe + görüntü ───────────
                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id   = int(tid)
                    stable_pid = stable_map.get(track_id, track_id)
                    x1, y1, x2, y2 = person_coords[stable_pid]

                    hcand = h_assigned.get(stable_pid)
                    hlabel = hcand["label"] if hcand else "unknown"
                    hconf  = hcand["conf"]  if hcand else 0.0
                    hbbox  = hcand["bbox_f"] if hcand else None
                    if TELEMETRY and hcand:
                        _log_ppe_decision(PPEDecision(
                            frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="helmet",
                            raw_label=hcand["raw_label"], conf=hcand["conf"],
                            own_score=hcand["own_score"], neighbor_pen=hcand["neighbor_pen"],
                            ambiguous=False, accepted=True, reason=hcand["reason"],
                        ))
                    states[stable_pid]["hardhat"].append(hlabel)
                    hvote = vote(states[stable_pid]["hardhat"], min_known=3)

                    vcand = v_assigned.get(stable_pid)
                    vlabel = vcand["label"] if vcand else "unknown"
                    vconf  = vcand["conf"]  if vcand else 0.0
                    vbbox  = vcand["bbox_f"] if vcand else None
                    if TELEMETRY and vcand:
                        _log_ppe_decision(PPEDecision(
                            frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="vest",
                            raw_label=vcand["raw_label"], conf=vcand["conf"],
                            own_score=vcand["own_score"], neighbor_pen=vcand["neighbor_pen"],
                            ambiguous=False, accepted=True, reason=vcand["reason"],
                        ))
                    states[stable_pid]["vest"].append(vlabel)
                    vvote = vote(states[stable_pid]["vest"], min_known=2)

                    mcand = m_assigned.get(stable_pid)
                    mlabel = mcand["label"] if mcand else "unknown"
                    mconf  = mcand["conf"]  if mcand else 0.0
                    mbbox  = mcand["bbox_f"] if mcand else None
                    if TELEMETRY and mcand:
                        _log_ppe_decision(PPEDecision(
                            frame_idx=frame_idx, stable_pid=stable_pid, ppe_type="mask",
                            raw_label=mcand["raw_label"], conf=mcand["conf"],
                            own_score=mcand["own_score"], neighbor_pen=mcand["neighbor_pen"],
                            ambiguous=False, accepted=True, reason=mcand["reason"],
                        ))
                    states[stable_pid]["mask"].append(mlabel)
                    mvote = vote(states[stable_pid]["mask"], min_known=1)

                    # Ghost track filtresi: yeterince görülmemiş track'ları atla
                    if states[stable_pid]["frame_count"] < MIN_TRACK_FRAMES:
                        if display:
                            draw_box(draw_frame, x1, y1, x2, y2, "...", COLOR_UNKNOWN)
                        continue

                    color, viols = compliance_color(hvote, vvote, mvote)
                    persons_with_ppe.append({
                        "track_id":     stable_pid,
                        "violations":   viols,
                        "helmet_status": "ok" if hvote == "Hardhat" else "violation" if hvote == "NO-Hardhat" else "unknown",
                        "vest_status":   "ok" if vvote == "Safety Vest" else "violation" if vvote == "NO-Safety Vest" else "unknown",
                        "mask_status":   "ok" if mvote == "Mask" else "violation" if mvote == "NO-Mask" else "unknown",
                        "helmet_conf":   round(hconf, 2),
                        "vest_conf":     round(vconf, 2),
                        "mask_conf":     round(mconf, 2),
                    })

                    if display:
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

            # --- Fire / smoke detection ---
            fire_res = fire_model.predict(
                frame, imgsz=IMGSZ, conf=FIRE_CONF, device=device, verbose=False,
            )[0]
            fire_raw = False
            fire_conf_max = 0.0
            smoke_raw = False
            smoke_conf_max = 0.0
            if fire_res.boxes:
                for box in fire_res.boxes:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = fire_model.names[cid]
                    if name == "fire":
                        fire_raw = True
                        fire_conf_max = max(fire_conf_max, conf)
                    elif name == "smoke":
                        smoke_raw = True
                        smoke_conf_max = max(smoke_conf_max, conf)
                    # "other" → yok sayılır
            if (fire_raw or smoke_raw) and display:
                label = "FIRE" if fire_raw else "SMOKE"
                cv2.putText(draw_frame, f"{label} DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FIRE, 3)

            # --- Event state machine ---
            event_info = event_manager.process_frame(persons_with_ppe, fire_raw or smoke_raw)

            if event_info["should_save"] and event_info.get("event_id"):
                event_count += 1
                save_event(
                    event_info, draw_frame, results_dir, persons_with_ppe,
                    fire_conf=fire_conf_max,
                    smoke_detected=smoke_raw,
                    smoke_conf=smoke_conf_max,
                )

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
        print(f"\nToplam frame: {frame_idx} | Kaydedilen event: {event_count}")
        print(f"Sonuclar: results/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Factory Safety — crop-based PPE pipeline")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video",  help="Video dosyasi yolu")
    src.add_argument("--camera", default=0, help="Kamera indeksi (varsayilan: 0)")
    parser.add_argument("--device",  default=_DEVICE, help="cuda device veya cpu")
    parser.add_argument("--display", action="store_true", help="OpenCV penceresi goster (varsayilan: kapali)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
