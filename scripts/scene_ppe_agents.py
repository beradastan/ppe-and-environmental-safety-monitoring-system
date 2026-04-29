# -*- coding: utf-8 -*-
"""
scene_ppe_agents.py
===================
Sahne tabanlı, cok-ajan, kisi-merkezli PPE durum uretici.

Mimari:
  - helmet_agent  : Hardhat / NO-Hardhat
  - vest_agent    : Safety Vest / NO-Safety Vest
  - mask_model    : Mask / NO-Mask  +  Person (anchor)   [Vinayak orijinal]

Her model sadece kendi uzmanlık siniflarini katkida bulunur.
Person detection Vinayak'tan gelir (mask modeli zaten calisiyor).
Atama + fusion logic scene_ppe_prototype.py ile aynıdır.

Kullanim:
    python scripts/scene_ppe_agents.py --video test/noppe_test.mp4
    python scripts/scene_ppe_agents.py --video test/nohat_test.mp4 --conf 0.20
    python scripts/scene_ppe_agents.py --video test/novest_test.mp4 --show
    python scripts/scene_ppe_agents.py --video test/noppe_test.mp4 --show --show-regions --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Model yollari
# ---------------------------------------------------------------------------

HELMET_MODEL_PATH     = ROOT / "models/vinayak_trained_byBera/helmet_agent_final_best.pt"
CROPHELMET_MODEL_PATH = ROOT / "models/vinayak_trained_byBera/crophelmet_agent_final_best.pt"
VEST_MODEL_PATH       = ROOT / "models/vinayak_trained_byBera/vest_agent_final_best.pt"
MASK_MODEL_PATH       = ROOT / "models/pretrained/vinayakmane/ppe.pt"   # fallback (person anchor)
PERSON_MODEL_PATH     = ROOT / "models/person_agent_scene_vinayakstyle_best.pt"
MASK_AGENT_PATH       = ROOT / "models/mask_agent_scene_200ep_best.pt"  # dedicated mask agent

# Her model yalnizca bu label'lari sisteme katar
HELMET_LABELS:   frozenset[str] = frozenset({"Hardhat", "NO-Hardhat"})
VEST_LABELS:     frozenset[str] = frozenset({"Safety Vest", "NO-Safety Vest"})
MASK_LABELS:     frozenset[str] = frozenset({"Mask", "NO-Mask", "Person"})
MASK_ONLY_LABELS: frozenset[str] = frozenset({"Mask", "NO-Mask"})  # mask_model when person comes from elsewhere
CROPMASK_LABELS: frozenset[str] = frozenset({"Mask", "NO-Mask"})   # dedicated 2-class model
PERSON_LABELS:   frozenset[str] = frozenset({"Person", "person"})   # COCO + PPE modelleri

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

MIN_SCORE    = 0.08
CONF_DEFAULT = 0.20
SCORE_W_IOU  = 0.70
SCORE_W_INS  = 0.30

# --- Helmet-specific tuning -------------------------------------------
# Head region: mask icin genel "head" kullanilmaya devam eder;
# helmet "helmet_head" ile calisir (daha dar, daha odakli).
HELMET_HEAD_W_EXPAND     = 0.05   # yatay genisletme  (genel head: 0.10)
HELMET_HEAD_H_EXPAND     = 0.05   # yukari genisletme (genel head: 0.10)
HELMET_HEAD_H_RATIO      = 0.32   # asagi sinir       (genel head: 0.38)
HELMET_MIN_SCORE         = 0.12   # helmet atama esigi (genel: MIN_SCORE=0.08)
HELMET_CONFLICT_RATIO    = 2.0    # ok karari icin pos/neg oran esigi (genel: 1.5)
HELMET_POS_MIN_COMBINED  = 0.04   # Hardhat pozitif sayilmasi icin min combined skor
# ----------------------------------------------------------------------

# --- Kişi crop + helmet tuning (finalmodelpipeline CropPPEMatcher mantığı) ---
CROPUPPER_PAD = 0.40   # 4 yönde padding oranı (CropPPEMatcher default: 0.40)
# ----------------------------------------------------------------------

PPE_META: dict[str, dict] = {
    "Hardhat":        {"category": "helmet", "polarity": "pos"},
    "NO-Hardhat":     {"category": "helmet", "polarity": "neg"},
    "Mask":           {"category": "mask",   "polarity": "pos"},
    "NO-Mask":        {"category": "mask",   "polarity": "neg"},
    "Safety Vest":    {"category": "vest",   "polarity": "pos"},
    "NO-Safety Vest": {"category": "vest",   "polarity": "neg"},
}
PERSON_LABEL    = "Person"
# helmet -> helmet_head (dar bolge); mask -> head (genel); vest -> torso
CATEGORY_REGION = {"helmet": "helmet_head", "mask": "head", "vest": "torso"}

_COL = {
    "ok":        (30,  200,  30),
    "violation": (20,   20, 220),
    "unknown":   (20,  200, 220),
    "helmet":    (0,   140, 255),
    "vest":      (0,   210, 210),
    "mask":      (220,  80,  80),
    "person":    (180, 180, 180),
    "unassigned":(80,   80,  80),
}


# ---------------------------------------------------------------------------
# Veri yapilari
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label:    str
    conf:     float
    bbox:     list[float]
    class_id: int
    source:   str = ""          # hangi ajandan geldi (debug icin)


@dataclass
class PPEAssignment:
    det:         Detection
    region_iou:  float
    inside_frac: float
    score:       float


@dataclass
class PersonPPEState:
    person_idx:    int
    bbox:          list[float]
    helmet_status: str
    vest_status:   str
    mask_status:   str
    evidence:      dict
    assignments:   list[PPEAssignment] = field(default_factory=list)


@dataclass
class FrameSummary:
    frame_idx:      int
    person_count:   int
    persons:        list[dict]
    unassigned_ppe: list[dict]


# ---------------------------------------------------------------------------
# Ajan seti
# ---------------------------------------------------------------------------

@dataclass
class AgentSet:
    helmet:         object        # YOLO — helmet_agent veya crophelmet_agent
    vest:           object        # YOLO
    mask:           object        # YOLO — ayni zamanda Person anchor
    cropmask:       object = None # YOLO — opsiyonel dedicated mask agent (2-class)
    person_model:   object = None # YOLO — opsiyonel dedicated person detector (COCO vb.)
    use_crophelmet: bool = False  # True -> full-person crop -> crophelmet_agent
    use_cropupper:  bool = False  # True -> full-person+pad crop -> helmet_agent
    use_cropmask:   bool = False  # True -> person crop -> cropmask_agent (mask only)


_UNSET = object()   # sentinel: "caller did not pass this argument"


def load_agents(
    device:         str,
    use_crophelmet: bool = False,
    helmet_path:    Path | None = None,
    vest_path:      Path | None = None,
    mask_path:      Path | None = None,
    cropmask_path:  object = _UNSET,   # _UNSET → default; None → devre disi
    person_path:    Path | None = None,
) -> AgentSet:
    from ultralytics import YOLO

    if helmet_path is None:
        helmet_path = CROPHELMET_MODEL_PATH if use_crophelmet else HELMET_MODEL_PATH
    if vest_path is None:
        vest_path = VEST_MODEL_PATH
    if mask_path is None:
        mask_path = MASK_MODEL_PATH
    if person_path is None:
        person_path = PERSON_MODEL_PATH
    if cropmask_path is _UNSET:
        cropmask_path = MASK_AGENT_PATH   # varsayilan: crop modu aktif
    # cropmask_path=None → crop modu devre disi, mask_model tam sahneye
    mode_label = "crophelmet_agent" if use_crophelmet else "helmet_agent"

    required = [
        (mode_label,     helmet_path),
        ("vest_agent",   vest_path),
        ("mask_model",   mask_path),
        ("person_agent", person_path),
    ]
    if cropmask_path is not None:
        required.append(("mask_agent", cropmask_path))
    for name, path in required:
        if not path.exists():
            raise FileNotFoundError(f"{name} bulunamadi: {path}")

    print(f"  {mode_label:<16}: {helmet_path.name}")
    helmet = YOLO(str(helmet_path))

    print(f"  vest_agent      : {vest_path.name}")
    vest = YOLO(str(vest_path))

    print(f"  mask_model      : {mask_path.name}  (person anchor fallback)")
    mask = YOLO(str(mask_path))

    if cropmask_path is not None:
        print(f"  mask_agent      : {cropmask_path.name}  (dedicated mask — crop modu)")
        cropmask = YOLO(str(cropmask_path))
    else:
        print(f"  mask_agent      : DEVRE DISI — mask_model tam sahneye uygulanacak")
        cropmask = None

    print(f"  person_agent    : {person_path.name}  (dedicated person detector)")
    person_model = YOLO(str(person_path))

    return AgentSet(helmet=helmet, vest=vest, mask=mask, cropmask=cropmask,
                    person_model=person_model,
                    use_crophelmet=use_crophelmet, use_cropupper=False,
                    use_cropmask=(cropmask is not None))


# ---------------------------------------------------------------------------
# 1. Cok-ajan tespit toplama
# ---------------------------------------------------------------------------

def _run_model_filtered(
    model, frame, allowed: frozenset[str], source: str,
    conf: float, device: str, use_half: bool,
) -> list[Detection]:
    """Tek model predict — yalnizca allowed label'lari dondurur."""
    preds = model.predict(frame, conf=conf, device=device, half=use_half, verbose=False)
    if not preds or preds[0].boxes is None:
        return []
    out = []
    for box in preds[0].boxes:
        cid   = int(box.cls[0])
        label = model.names[cid]
        if label not in allowed:
            continue
        out.append(Detection(
            label=label,
            conf=float(box.conf[0]),
            bbox=list(map(float, box.xyxy[0].tolist())),
            class_id=cid,
            source=source,
        ))
    return out


CROPHELMET_PAD = 0.30   # benchmark ile ayni padding orani

def _crophelmet_for_person(
    model,
    person_bbox: list[float],
    frame,
    conf:   float,
    device: str,
    fh:     int,
    fw:     int,
) -> list[Detection]:
    """
    Kisi bbox'unu CROPHELMET_PAD ile genisleterek full-person crop alir,
    crophelmet modelini calistirir, tespitleri full-frame koordinatina cevirir.
    Benchmark'taki crop_with_padding() ile ayni mantik.
    """
    x1, y1, x2, y2 = person_bbox
    pw = (x2 - x1) * CROPHELMET_PAD
    ph = (y2 - y1) * CROPHELMET_PAD
    cx1 = int(max(0,  x1 - pw))
    cy1 = int(max(0,  y1 - ph))
    cx2 = int(min(fw, x2 + pw))
    cy2 = int(min(fh, y2 + ph))
    if cx2 <= cx1 or cy2 <= cy1:
        return []
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    use_half = (device == "cuda")
    preds = model.predict(crop, conf=conf, device=device, half=use_half, verbose=False)
    if not preds or preds[0].boxes is None:
        return []

    out = []
    for box in preds[0].boxes:
        cid   = int(box.cls[0])
        label = model.names[cid]
        if label not in HELMET_LABELS:
            continue
        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
        out.append(Detection(
            label=label,
            conf=float(box.conf[0]),
            bbox=[cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2],
            class_id=cid,
            source="crophelmet_agent",
        ))
    return out


def _make_person_crop(
    person_bbox: list[float],
    frame,
    fh: int,
    fw: int,
) -> tuple[object, int, int] | None:
    """
    CropPPEMatcher mantığı: kişinin tam bbox'ını CROPUPPER_PAD ile 4 yönde genişletir.
    Döndürür: (crop, cx1, cy1) veya None (geçersiz crop).
    """
    x1, y1, x2, y2 = person_bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * CROPUPPER_PAD
    pad_y = bh * CROPUPPER_PAD
    cx1 = int(max(0,  x1 - pad_x))
    cy1 = int(max(0,  y1 - pad_y))
    cx2 = int(min(fw, x2 + pad_x))
    cy2 = int(min(fh, y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop, cx1, cy1


def _cropmask_batch(
    model,
    person_dets: list,
    frame,
    conf:   float,
    device: str,
    fh:     int,
    fw:     int,
) -> list[Detection]:
    """
    Dedicated cropmask agent: her kişi için full bbox + CROPUPPER_PAD crop alır,
    2-class Mask/NO-Mask modelini çalıştırır, full-frame koordinatına çevirir.
    """
    crops, offsets = [], []
    for d in person_dets:
        result = _make_person_crop(d.bbox, frame, fh, fw)
        if result is None:
            continue
        crop, cx1, cy1 = result
        crops.append(crop)
        offsets.append((cx1, cy1))

    if not crops:
        return []

    use_half = (device == "cuda")
    batch_preds = model.predict(
        crops, conf=conf, device=device, half=use_half, verbose=False
    )

    out = []
    for preds, (cx1, cy1) in zip(batch_preds, offsets):
        if preds.boxes is None:
            continue
        for box in preds.boxes:
            cid   = int(box.cls[0])
            label = model.names[cid]
            if label not in CROPMASK_LABELS:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            out.append(Detection(
                label=label,
                conf=float(box.conf[0]),
                bbox=[cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2],
                class_id=cid,
                source="cropmask_agent",
            ))
    return out


def _cropupper_batch(
    model,
    person_dets: list,
    frame,
    conf:   float,
    device: str,
    fh:     int,
    fw:     int,
) -> list[Detection]:
    """
    finalmodelpipeline CropPPEMatcher mantığı:
    - Her kişi için tam bbox + CROPUPPER_PAD padding ile crop alır
    - Tüm crop'ları tek batch olarak inference'a gönderir (verimli GPU kullanımı)
    - Tespitleri full-frame koordinatına çevirir
    """
    crops, offsets = [], []
    for d in person_dets:
        result = _make_person_crop(d.bbox, frame, fh, fw)
        if result is None:
            continue
        crop, cx1, cy1 = result
        crops.append(crop)
        offsets.append((cx1, cy1))

    if not crops:
        return []

    use_half = (device == "cuda")
    batch_preds = model.predict(
        crops, conf=conf, device=device, half=use_half, verbose=False
    )

    out = []
    for preds, (cx1, cy1) in zip(batch_preds, offsets):
        if preds.boxes is None:
            continue
        for box in preds.boxes:
            cid   = int(box.cls[0])
            label = model.names[cid]
            if label not in HELMET_LABELS:
                continue
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            out.append(Detection(
                label=label,
                conf=float(box.conf[0]),
                bbox=[cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2],
                class_id=cid,
                source="cropperson_helmet",
            ))
    return out


def _run_person_model(
    model, frame, conf: float, device: str, use_half: bool
) -> list[Detection]:
    """COCO veya PPE modelinden sadece person tespitlerini alır, label'ı 'Person' olarak normalize eder."""
    preds = model.predict(frame, conf=conf, device=device, half=use_half, verbose=False)
    if not preds or preds[0].boxes is None:
        return []
    out = []
    for box in preds[0].boxes:
        cid   = int(box.cls[0])
        label = model.names[cid]
        if label not in PERSON_LABELS:
            continue
        out.append(Detection(
            label=PERSON_LABEL,   # normalize: her zaman "Person"
            conf=float(box.conf[0]),
            bbox=list(map(float, box.xyxy[0].tolist())),
            class_id=cid,
            source="person_model",
        ))
    return out


def collect_detections(
    agents: AgentSet,
    frame,
    conf:   float,
    device: str,
    fh:     int = 0,
    fw:     int = 0,
) -> list[Detection]:
    """
    Adım 1 — Person anchor:
      person_model varsa → dedicated detector (COCO vb.), label normalize edilir.
      Yoksa → mask_model filtreli (use_cropmask=True → sadece Person, değilse MASK_LABELS).
      person_model yoksa ve use_cropmask=False → mask_model Mask/NO-Mask de verir.

    Adım 2 — Mask:
      use_cropmask=True → _cropmask_batch (person crop → dedicated 2-class model).
      person_model varsa ve use_cropmask=False → mask_model Mask/NO-Mask tam kare.
      Aksi → zaten Adım 1'de MASK_LABELS içinde geldi.

    Adım 3 — Helmet: crophelmet / cropupper / scene tam kare.
    Adım 4 — Vest: her zaman tam kare.
    """
    use_half = (device == "cuda")
    dets: list[Detection] = []

    # --- Adim 1: Person anchor -------------------------------------------------
    if agents.person_model is not None:
        anchor_dets = _run_person_model(agents.person_model, frame, conf, device, use_half)
        dets.extend(anchor_dets)
        person_dets = anchor_dets  # hepsi zaten Person
    else:
        if agents.use_cropmask:
            mask_allowed = frozenset({PERSON_LABEL})
        else:
            mask_allowed = MASK_LABELS
        anchor_dets = _run_model_filtered(
            agents.mask, frame, mask_allowed, "mask_model", conf, device, use_half
        )
        dets.extend(anchor_dets)
        person_dets = [d for d in anchor_dets if d.label == PERSON_LABEL]

    # --- Adim 2: Mask ----------------------------------------------------------
    if agents.use_cropmask and agents.cropmask is not None:
        if person_dets and fh and fw:
            dets.extend(_cropmask_batch(
                agents.cropmask, person_dets, frame, conf, device, fh, fw
            ))
    elif agents.person_model is not None:
        # Dedicated person detector kullanildi → mask_model yalnizca Mask/NO-Mask
        dets.extend(_run_model_filtered(
            agents.mask, frame, MASK_ONLY_LABELS, "mask_model", conf, device, use_half
        ))
    # else: mask_model zaten Adim 1'de MASK_LABELS ile calistirildi

    # --- Adim 3: Helmet --------------------------------------------------------
    if agents.use_crophelmet:
        for d in person_dets:
            if fh and fw:
                dets.extend(_crophelmet_for_person(
                    agents.helmet, d.bbox, frame, conf, device, fh, fw
                ))
    elif agents.use_cropupper:
        if person_dets and fh and fw:
            dets.extend(_cropupper_batch(
                agents.helmet, person_dets, frame, conf, device, fh, fw
            ))
    else:
        dets.extend(_run_model_filtered(
            agents.helmet, frame, HELMET_LABELS, "helmet_agent", conf, device, use_half
        ))

    # --- Adim 4: Vest ----------------------------------------------------------
    dets.extend(_run_model_filtered(
        agents.vest, frame, VEST_LABELS, "vest_agent", conf, device, use_half
    ))

    return dets


# ---------------------------------------------------------------------------
# 2. Ayirma
# ---------------------------------------------------------------------------

def split_person_and_ppe(
    dets: list[Detection],
) -> tuple[list[Detection], list[Detection]]:
    person_dets: list[Detection] = []
    ppe_dets:    list[Detection] = []
    for d in dets:
        if d.label == PERSON_LABEL:
            person_dets.append(d)
        elif d.label in PPE_META:
            ppe_dets.append(d)
    return person_dets, ppe_dets


# ---------------------------------------------------------------------------
# 3. Anatomik bolgeler
# ---------------------------------------------------------------------------

def person_regions(bbox: list[float], fh: int, fw: int) -> dict[str, list[float]]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    def clip(b):
        return [max(0.0, b[0]), max(0.0, b[1]),
                min(float(fw), b[2]), min(float(fh), b[3])]

    # Mask icin genel head (genis): maske yuz ortasinda, biraz daha fazla alan lazim
    head  = clip([x1 - 0.10 * w, y1 - 0.10 * h,
                  x2 + 0.10 * w, y1 + 0.38 * h])

    # Helmet icin dar head: kask tam tepede, komsu kisi overlap'ini azaltir
    helmet_head = clip([x1 - HELMET_HEAD_W_EXPAND * w, y1 - HELMET_HEAD_H_EXPAND * h,
                        x2 + HELMET_HEAD_W_EXPAND * w, y1 + HELMET_HEAD_H_RATIO  * h])

    torso = clip([x1 - 0.05 * w, y1 + 0.10 * h,
                  x2 + 0.05 * w, y1 + 0.85 * h])

    return {"head": head, "helmet_head": helmet_head, "torso": torso}


# ---------------------------------------------------------------------------
# 4. Geometri
# ---------------------------------------------------------------------------

def iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    a_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / (a_a + a_b - inter)


def inside_fraction(ppe_box: list[float], person_box: list[float]) -> float:
    ix1 = max(ppe_box[0], person_box[0])
    iy1 = max(ppe_box[1], person_box[1])
    ix2 = min(ppe_box[2], person_box[2])
    iy2 = min(ppe_box[3], person_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ppe_area = max(1.0, (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1]))
    return inter / ppe_area


# ---------------------------------------------------------------------------
# 5. Atama aday listesi
# ---------------------------------------------------------------------------

def build_assignment_candidates(
    person_dets: list[Detection],
    ppe_dets:    list[Detection],
    fh: int,
    fw: int,
) -> list[tuple[int, int, float, float, float]]:
    candidates: list[tuple[int, int, float, float, float]] = []
    for pi, person in enumerate(person_dets):
        regions = person_regions(person.bbox, fh, fw)
        for qi, ppe in enumerate(ppe_dets):
            meta      = PPE_META[ppe.label]
            cat       = meta["category"]
            region_box = regions[CATEGORY_REGION[cat]]
            r_iou  = iou(ppe.bbox, region_box)
            i_frac = inside_fraction(ppe.bbox, person.bbox)
            score  = SCORE_W_IOU * r_iou + SCORE_W_INS * i_frac
            # Helmet icin daha siki atama esigi
            threshold = HELMET_MIN_SCORE if cat == "helmet" else MIN_SCORE
            if score >= threshold:
                candidates.append((pi, qi, score, r_iou, i_frac))
    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 6. Greedy one-to-one atama
# ---------------------------------------------------------------------------

def assign_ppe_to_persons(
    candidates:  list[tuple[int, int, float, float, float]],
    person_dets: list[Detection],
    ppe_dets:    list[Detection],
) -> tuple[dict[int, list[PPEAssignment]], list[Detection]]:
    assigned_ppe: set[int] = set()
    best_per_person_label: dict[tuple[int, str], tuple[int, PPEAssignment]] = {}

    for pi, qi, score, r_iou, i_frac in candidates:
        if qi in assigned_ppe:
            continue
        ppe = ppe_dets[qi]
        key = (pi, ppe.label)
        if key not in best_per_person_label:
            asgn = PPEAssignment(
                det=ppe, region_iou=r_iou, inside_frac=i_frac, score=score
            )
            best_per_person_label[key] = (qi, asgn)
            assigned_ppe.add(qi)

    assignments: dict[int, list[PPEAssignment]] = {
        i: [] for i in range(len(person_dets))
    }
    for (pi, _), (_, asgn) in best_per_person_label.items():
        assignments[pi].append(asgn)

    unassigned = [ppe_dets[qi] for qi in range(len(ppe_dets))
                  if qi not in assigned_ppe]
    return assignments, unassigned


# ---------------------------------------------------------------------------
# 7. PPE durum birlestirme
# ---------------------------------------------------------------------------

def _resolve_category(
    pos_scores: list[float],
    neg_scores: list[float],
    conflict_ratio: float = 1.5,
) -> str:
    if not pos_scores and not neg_scores:
        return "unknown"
    if not neg_scores:
        return "ok"
    if not pos_scores:
        return "violation"
    best_pos = max(pos_scores)
    best_neg = max(neg_scores)
    if best_pos > best_neg * conflict_ratio:
        return "ok"
    if best_neg > best_pos * conflict_ratio:
        return "violation"
    return "unknown"


def fuse_person_ppe_state(
    person_idx: int,
    person_det: Detection,
    asgns:      list[PPEAssignment],
) -> PersonPPEState:
    evidence: dict[str, dict] = {}
    status:   dict[str, str]  = {}

    for cat in ("helmet", "vest", "mask"):
        pos: list[float] = []
        neg: list[float] = []
        ev_pos: list[dict] = []
        ev_neg: list[dict] = []

        for a in asgns:
            meta = PPE_META.get(a.det.label)
            if meta is None or meta["category"] != cat:
                continue
            combined = a.det.conf * a.score

            # Zayif Hardhat pozitifi filtrele: cok dusuk combined skor
            # "ok" karari veren yanlis Hardhat atamalarini azaltir
            if (cat == "helmet" and meta["polarity"] == "pos"
                    and combined < HELMET_POS_MIN_COMBINED):
                continue

            entry = {
                "label":       a.det.label,
                "conf":        round(a.det.conf, 3),
                "assoc_score": round(a.score, 3),
                "combined":    round(combined, 3),
                "source":      a.det.source,
            }
            if meta["polarity"] == "pos":
                pos.append(combined); ev_pos.append(entry)
            else:
                neg.append(combined); ev_neg.append(entry)

        # Helmet icin daha siki conflict resolution
        ratio = HELMET_CONFLICT_RATIO if cat == "helmet" else 1.5
        evidence[cat] = {"positive": ev_pos, "negative": ev_neg}
        status[cat]   = _resolve_category(pos, neg, conflict_ratio=ratio)

    return PersonPPEState(
        person_idx=person_idx,
        bbox=person_det.bbox,
        helmet_status=status["helmet"],
        vest_status=status["vest"],
        mask_status=status["mask"],
        evidence=evidence,
        assignments=asgns,
    )


# ---------------------------------------------------------------------------
# 8. Frame ozeti
# ---------------------------------------------------------------------------

def build_frame_summary(
    frame_idx:      int,
    person_states:  list[PersonPPEState],
    unassigned_ppe: list[Detection],
) -> FrameSummary:
    persons_data = []
    for s in person_states:
        persons_data.append({
            "person_idx":    s.person_idx,
            "bbox":          [round(v, 1) for v in s.bbox],
            "helmet_status": s.helmet_status,
            "vest_status":   s.vest_status,
            "mask_status":   s.mask_status,
            "evidence":      s.evidence,
        })
    unassigned_data = [
        {"label": d.label, "conf": round(d.conf, 3),
         "bbox": [round(v, 1) for v in d.bbox], "source": d.source}
        for d in unassigned_ppe
    ]
    return FrameSummary(
        frame_idx=frame_idx,
        person_count=len(person_states),
        persons=persons_data,
        unassigned_ppe=unassigned_data,
    )


# ---------------------------------------------------------------------------
# 9. Gorsellestime
# ---------------------------------------------------------------------------

def _worst_status(s: PersonPPEState) -> str:
    for st in (s.helmet_status, s.vest_status, s.mask_status):
        if st == "violation":
            return "violation"
    for st in (s.helmet_status, s.vest_status, s.mask_status):
        if st == "unknown":
            return "unknown"
    return "ok"


def _draw_box(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def _put_label(img, text: str, x: int, y: int, color, font_scale=0.55, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 2), color, -1)
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_col = (0, 0, 0) if brightness > 120 else (255, 255, 255)
    cv2.putText(img, text, (x + 3, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, thickness, cv2.LINE_AA)


def render_person_state(
    img,
    state: PersonPPEState,
    show_regions: bool = False,
    fh: int = 0,
    fw: int = 0,
) -> None:
    x1, y1, x2, y2 = map(int, state.bbox)
    worst = _worst_status(state)
    box_color = _COL[worst]

    _draw_box(img, x1, y1, x2, y2, box_color, thickness=3)

    if show_regions and fh and fw:
        regs = person_regions(state.bbox, fh, fw)
        _draw_box(img, *regs["head"],        (200, 200, 80),  thickness=1)  # mask head (genis)
        _draw_box(img, *regs["helmet_head"], (255, 140,  0),  thickness=1)  # helmet head (dar)
        _draw_box(img, *regs["torso"],       (80,  200, 200), thickness=1)

    def st_ch(st): return "OK" if st == "ok" else ("!!" if st == "violation" else "?")
    label = (f"#{state.person_idx}  "
             f"H:{st_ch(state.helmet_status)} "
             f"V:{st_ch(state.vest_status)} "
             f"M:{st_ch(state.mask_status)}")
    _put_label(img, label, x1, y1, box_color, font_scale=0.52)

    for a in state.assignments:
        meta = PPE_META[a.det.label]
        cat  = meta["category"]
        pol  = meta["polarity"]
        bx1, by1, bx2, by2 = map(int, a.det.bbox)
        col = _COL[cat] if pol == "neg" else (
            tuple(min(255, int(c * 1.4)) for c in _COL[cat])
        )
        _draw_box(img, bx1, by1, bx2, by2, col, thickness=2)
        short = a.det.label.replace("NO-", "NO ").replace("Safety ", "")
        _put_label(img, f"{short} {a.det.conf:.0%}", bx1, by1, col, font_scale=0.42)


def render_unassigned(img, unassigned: list[Detection]) -> None:
    for d in unassigned:
        bx1, by1, bx2, by2 = map(int, d.bbox)
        col = _COL["unassigned"]
        seg = 10
        for (sx, sy, ex, ey) in [
            (bx1, by1, bx1+seg, by1), (bx2-seg, by1, bx2, by1),
            (bx1, by2, bx1+seg, by2), (bx2-seg, by2, bx2, by2),
            (bx1, by1, bx1, by1+seg), (bx1, by2-seg, bx1, by2),
            (bx2, by1, bx2, by1+seg), (bx2, by2-seg, bx2, by2),
        ]:
            cv2.line(img, (sx, sy), (ex, ey), col, 2)
        short = d.label.replace("NO-", "NO ").replace("Safety ", "")
        _put_label(img, f"? {short} {d.conf:.0%}", bx1, by1, col, font_scale=0.38)


def render_hud(
    img, frame_idx: int, person_count: int, fps: float,
    conf: float, crophelmet: bool = False,
) -> None:
    helmet_mode = "crophelmet(bera)" if crophelmet else "helmet_scene(bera)"
    lines = [
        f"Frame: {frame_idx}  Persons: {person_count}",
        f"FPS: {fps:.0f}  conf: {conf:.2f}",
        f"H:{helmet_mode} V:vest(bera) M:vinayak",
        "H=helmet  V=vest  M=mask  OK/!!/? ",
    ]
    scale, thick, lh = 0.50, 2, 20
    mw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines)
    pw, ph = mw + 16, lh * len(lines) + 12
    cv2.rectangle(img, (6, 6), (6 + pw, 6 + ph), (20, 20, 20), -1)
    cv2.rectangle(img, (6, 6), (6 + pw, 6 + ph), (140, 140, 140), 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 6 + 16 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 10. Aggregate istatistik
# ---------------------------------------------------------------------------

def _aggregate_stats(frame_summaries: list[FrameSummary]) -> dict:
    total_person_frames = sum(s.person_count for s in frame_summaries)
    counts: dict[str, dict[str, int]] = {
        cat: {"ok": 0, "violation": 0, "unknown": 0}
        for cat in ("helmet", "vest", "mask")
    }
    for fs in frame_summaries:
        for p in fs.persons:
            for cat in ("helmet", "vest", "mask"):
                st = p[f"{cat}_status"]
                counts[cat][st] += 1

    rates: dict[str, dict] = {}
    for cat, c in counts.items():
        known = c["ok"] + c["violation"]
        rates[cat] = {
            "ok":             c["ok"],
            "violation":      c["violation"],
            "unknown":        c["unknown"],
            "violation_rate": round(c["violation"] / known, 3) if known else None,
        }
    return {
        "total_person_frames": total_person_frames,
        "per_category":        rates,
    }


# ---------------------------------------------------------------------------
# 11. Ana pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    agents:       AgentSet,
    conf:         float,
    output_dir:   Path,
    show:         bool = False,
    show_regions: bool = False,
    device:       str  = "cpu",
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Video acilamadi: {video_path}")
        return

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem         = video_path.stem
    out_vid_path  = output_dir / f"{stem}_agents_ppe.mp4"
    out_json_path = output_dir / f"{stem}_agents_ppe.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps_cap, (fw, fh))

    frame_summaries: list[FrameSummary] = []
    fps_ring: list[float] = []
    frame_idx = 0

    if show:
        cv2.namedWindow("Scene PPE Agents", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Scene PPE Agents", 1280, 720)

    print(f"  Islenecek: {video_path.name}  ({total_f} frame  {fps_cap:.0f}fps  {fw}x{fh})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()

            dets = collect_detections(agents, frame, conf, device, fh=fh, fw=fw)
            person_dets, ppe_dets = split_person_and_ppe(dets)

            person_states:  list[PersonPPEState] = []
            unassigned_ppe: list[Detection]      = ppe_dets

            if person_dets:
                candidates = build_assignment_candidates(
                    person_dets, ppe_dets, fh, fw
                )
                assignments, unassigned_ppe = assign_ppe_to_persons(
                    candidates, person_dets, ppe_dets
                )
                for pi, person_det in enumerate(person_dets):
                    state = fuse_person_ppe_state(pi, person_det, assignments[pi])
                    person_states.append(state)

            fs = build_frame_summary(frame_idx, person_states, unassigned_ppe)
            frame_summaries.append(fs)

            draw = frame.copy()
            for state in person_states:
                render_person_state(draw, state,
                                    show_regions=show_regions, fh=fh, fw=fw)
            render_unassigned(draw, unassigned_ppe)

            elapsed = time.perf_counter() - t0
            fps_ring.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_ring) > 30:
                fps_ring.pop(0)
            avg_fps = sum(fps_ring) / len(fps_ring)

            render_hud(draw, frame_idx, len(person_states), avg_fps, conf,
                       crophelmet=agents.use_crophelmet)

            writer.write(draw)
            if show:
                cv2.imshow("Scene PPE Agents", draw)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"    [{frame_idx}/{total_f}]  {avg_fps:.0f} fps")

    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    aggregate = _aggregate_stats(frame_summaries)
    summary = {
        "video":        str(video_path),
        "total_frames": frame_idx,
        "conf":         conf,
        "agents": {
            "helmet": str(HELMET_MODEL_PATH),
            "vest":   str(VEST_MODEL_PATH),
            "mask":   str(MASK_MODEL_PATH),
        },
        "aggregate": aggregate,
        "frames": [
            {
                "frame_idx":      fs.frame_idx,
                "person_count":   fs.person_count,
                "persons":        fs.persons,
                "unassigned_ppe": fs.unassigned_ppe,
            }
            for fs in frame_summaries
        ],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Video : {out_vid_path}")
    print(f"  JSON  : {out_json_path}")
    print(f"  Frames: {frame_idx}  Toplam kisi-frame: {aggregate['total_person_frames']}")
    print("  Ihlal oranlari (bilinen durumlar arasinda):")
    for cat, r in aggregate["per_category"].items():
        vr = r["violation_rate"]
        vr_str = f"{vr:.1%}" if vr is not None else "N/A"
        print(f"    {cat:<8}: violation={r['violation']}  ok={r['ok']}  "
              f"unknown={r['unknown']}  rate={vr_str}")


# ---------------------------------------------------------------------------
# Giris noktasi
# ---------------------------------------------------------------------------

def main() -> None:
    global MIN_SCORE  # noqa: PLW0603

    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True, help="Video dosyasi yolu")
    ap.add_argument("--conf",         type=float, default=CONF_DEFAULT)
    ap.add_argument("--min-score",    type=float, default=MIN_SCORE,
                    help="PPE atama minimum skoru (varsayilan: 0.08)")
    ap.add_argument("--output-dir",   default="runs/scene_ppe_agents")
    ap.add_argument("--show",         action="store_true", help="Canli pencere")
    ap.add_argument("--show-regions", action="store_true",
                    help="Head/torso anatomik bolgeleri goster")
    ap.add_argument("--crophelmet",    action="store_true",
                    help="Helmet icin kisi head crop'u kullan (crophelmet_agent)")
    ap.add_argument("--cropupper",     action="store_true",
                    help="Helmet icin kisi ust bolge crop'u kullan (helmet_agent ile)")
    ap.add_argument("--helmet-model",  default=None,
                    help="Helmet model yolu (varsayilan: HELMET_MODEL_PATH sabiti)")
    ap.add_argument("--vest-model",    default=None,
                    help="Vest model yolu (varsayilan: VEST_MODEL_PATH sabiti)")
    ap.add_argument("--mask-model",    default=None,
                    help="Mask+person model yolu (varsayilan: MASK_MODEL_PATH sabiti)")
    ap.add_argument("--cropmask-model", default=None,
                    help="Dedicated 2-class Mask/NO-Mask model yolu (kisi crop ile)")
    ap.add_argument("--no-cropmask",    action="store_true",
                    help="Mask ajanini tam sahneye uygula (kisi crop devre disi)")
    ap.add_argument("--person-model",   default=None,
                    help="Dedicated person detector yolu (COCO vb., mask_model yerine anchor)")
    ap.add_argument("--device",        default="cuda")
    args = ap.parse_args()

    MIN_SCORE = args.min_score

    import torch
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / args.video
    if not video_path.exists():
        print(f"Video bulunamadi: {video_path}")
        return

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    helmet_path = None
    if args.helmet_model:
        helmet_path = Path(args.helmet_model)
        if not helmet_path.is_absolute():
            helmet_path = ROOT / args.helmet_model
        if not helmet_path.exists():
            print(f"Helmet model bulunamadi: {helmet_path}")
            return

    vest_path = None
    if args.vest_model:
        vest_path = Path(args.vest_model)
        if not vest_path.is_absolute():
            vest_path = ROOT / args.vest_model
        if not vest_path.exists():
            print(f"Vest model bulunamadi: {vest_path}")
            return

    mask_path = None
    if args.mask_model:
        mask_path = Path(args.mask_model)
        if not mask_path.is_absolute():
            mask_path = ROOT / args.mask_model
        if not mask_path.exists():
            print(f"Mask model bulunamadi: {mask_path}")
            return

    cropmask_path = None
    if args.cropmask_model:
        cropmask_path = Path(args.cropmask_model)
        if not cropmask_path.is_absolute():
            cropmask_path = ROOT / args.cropmask_model
        if not cropmask_path.exists():
            print(f"CropMask model bulunamadi: {cropmask_path}")
            return

    person_path = None
    if args.person_model:
        person_path = Path(args.person_model)
        if not person_path.is_absolute():
            person_path = ROOT / args.person_model
        if not person_path.exists():
            print(f"Person model bulunamadi: {person_path}")
            return

    if args.crophelmet and args.cropupper:
        print("Hata: --crophelmet ve --cropupper ayni anda kullanilamaz.")
        return

    mode = "crophelmet" if args.crophelmet else ("cropupper" if args.cropupper else "scene helmet")

    if args.no_cropmask:
        # Sahne modu: mask_agent tam kareye, crop devre disi
        if mask_path is None:
            mask_path = MASK_AGENT_PATH
        effective_cropmask = None          # crop kapalı
        mask_mode = "scene"
    elif cropmask_path is not None:
        effective_cropmask = cropmask_path  # CLI'dan gelen yol
        mask_mode = "crop (custom)"
    else:
        effective_cropmask = _UNSET        # varsayilan: MASK_AGENT_PATH
        mask_mode = "crop (default)"

    print(f"Ajanlar yukleniyor  device={device}  helmet_mode={mode}  mask_mode={mask_mode}")
    agents = load_agents(device, use_crophelmet=args.crophelmet,
                         helmet_path=helmet_path, vest_path=vest_path,
                         mask_path=mask_path, cropmask_path=effective_cropmask,
                         person_path=person_path)
    agents.use_cropupper = args.cropupper

    process_video(
        video_path=video_path,
        agents=agents,
        conf=args.conf,
        output_dir=output_dir,
        show=args.show,
        show_regions=args.show_regions,
        device=device,
    )


if __name__ == "__main__":
    main()
