# -*- coding: utf-8 -*-
"""
scene_ppe_prototype.py
======================
Sahne tabanlı, tek-model, kisi-merkezli PPE durum uretici (prototip).

Mimari:
  1. Vinayak modeli tam kareye uygulanir.
  2. Ciktilar person_dets / ppe_dets olarak ayrilir.
  3. Her kisi icin anatomik bolgeler (head, torso) hesaplanir.
  4. Her PPE kutusunu en uygun kisiye atayan greedy one-to-one assignment
     yapilir (skor = 0.7 * region_iou + 0.3 * inside_fraction).
  5. Her kisi icin helmet / vest / mask durumu cikarilir.
  6. Annotated video ve JSON summary yazilir.

Kullanim:
    python scripts/scene_ppe_prototype.py --video test/noppe_test.mp4
    python scripts/scene_ppe_prototype.py --video test/nohat_test.mp4 --conf 0.20
    python scripts/scene_ppe_prototype.py --video test/novest_test.mp4 --show
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
# Sabitler
# ---------------------------------------------------------------------------

MODEL_PATH   = ROOT / "models/pretrained/vinayakmane/ppe.pt"
MIN_SCORE    = 0.08    # atama icin minimum birlesik skor
CONF_DEFAULT = 0.20    # model confidence esigi
SCORE_W_IOU  = 0.70    # region_iou agirligi
SCORE_W_INS  = 0.30    # inside_fraction agirligi

# Vinayak sinif tanimlari
PPE_META: dict[str, dict] = {
    "Hardhat":        {"category": "helmet", "polarity": "pos"},
    "NO-Hardhat":     {"category": "helmet", "polarity": "neg"},
    "Mask":           {"category": "mask",   "polarity": "pos"},
    "NO-Mask":        {"category": "mask",   "polarity": "neg"},
    "Safety Vest":    {"category": "vest",   "polarity": "pos"},
    "NO-Safety Vest": {"category": "vest",   "polarity": "neg"},
}
PERSON_LABEL  = "Person"
IGNORE_LABELS = {"Safety Cone", "machinery", "vehicle"}

# PPE kategorisinin hangi anatomik bolgeye gittigi
CATEGORY_REGION = {"helmet": "head", "mask": "head", "vest": "torso"}

# Gorsellestirme renkleri (BGR)
_COL = {
    "ok":        (30,  200,  30),
    "violation": (20,   20, 220),
    "unknown":   (20,  200, 220),
    "helmet":    (0,   140, 255),   # turuncu
    "vest":      (0,   210, 210),   # sari
    "mask":      (220,  80,  80),   # mavi
    "person":    (180, 180, 180),   # gri
    "unassigned":(80,   80,  80),
}


# ---------------------------------------------------------------------------
# Veri yapilari
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label:    str
    conf:     float
    bbox:     list[float]   # [x1, y1, x2, y2]
    class_id: int


@dataclass
class PPEAssignment:
    det:         Detection
    region_iou:  float
    inside_frac: float
    score:       float      # 0.7*region_iou + 0.3*inside_frac


@dataclass
class PersonPPEState:
    person_idx:     int
    bbox:           list[float]
    helmet_status:  str         # ok / violation / unknown
    vest_status:    str
    mask_status:    str
    evidence:       dict        # {category: {pos: [...], neg: [...]}}
    assignments:    list[PPEAssignment] = field(default_factory=list)


@dataclass
class FrameSummary:
    frame_idx:      int
    person_count:   int
    persons:        list[dict]
    unassigned_ppe: list[dict]


# ---------------------------------------------------------------------------
# 1. Tespit toplama
# ---------------------------------------------------------------------------

def collect_detections(model, frame, conf: float, device: str = "cpu") -> list[Detection]:
    """Modeli tam kareye uygular, Detection listesi dondurur."""
    preds = model.predict(frame, conf=conf, device=device,
                          half=(device == "cuda"), verbose=False)
    dets: list[Detection] = []
    if not preds or preds[0].boxes is None:
        return dets
    for box in preds[0].boxes:
        cid   = int(box.cls[0])
        label = model.names[cid]
        dets.append(Detection(
            label=label,
            conf=float(box.conf[0]),
            bbox=list(map(float, box.xyxy[0].tolist())),
            class_id=cid,
        ))
    return dets


# ---------------------------------------------------------------------------
# 2. Ayirma
# ---------------------------------------------------------------------------

def split_person_and_ppe(
    dets: list[Detection],
) -> tuple[list[Detection], list[Detection]]:
    """
    Dondurur: (person_dets, ppe_dets)
    IGNORE_LABELS ve tanimlanmamis siniflar her iki listede de olmaz.
    """
    person_dets: list[Detection] = []
    ppe_dets:    list[Detection] = []
    for d in dets:
        if d.label == PERSON_LABEL:
            person_dets.append(d)
        elif d.label in PPE_META:
            ppe_dets.append(d)
        # IGNORE_LABELS: sessizce atla
    return person_dets, ppe_dets


# ---------------------------------------------------------------------------
# 3. Anatomik bolgeler
# ---------------------------------------------------------------------------

def person_regions(bbox: list[float], fh: int, fw: int) -> dict[str, list[float]]:
    """
    Kisi bbox'indan head ve torso anatomik bolgelerini uretir.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    def clip(b):
        return [max(0.0, b[0]), max(0.0, b[1]),
                min(float(fw), b[2]), min(float(fh), b[3])]

    head  = clip([x1 - 0.10 * w, y1 - 0.10 * h,
                  x2 + 0.10 * w, y1 + 0.38 * h])
    torso = clip([x1 - 0.05 * w, y1 + 0.10 * h,
                  x2 + 0.05 * w, y1 + 0.85 * h])
    return {"head": head, "torso": torso}


# ---------------------------------------------------------------------------
# 4. Geometri
# ---------------------------------------------------------------------------

def iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU iki bbox arasinda."""
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
    """PPE kutusunun kisi kutusu icinde kalan orani."""
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
    """
    Her (person_idx, ppe_idx) cifti icin skor hesaplar.
    Dondurur: [(person_idx, ppe_idx, score, region_iou, inside_frac), ...]
    MIN_SCORE altindaki adaylar listeden cikarilir.
    """
    candidates: list[tuple[int, int, float, float, float]] = []
    for pi, person in enumerate(person_dets):
        regions = person_regions(person.bbox, fh, fw)
        for qi, ppe in enumerate(ppe_dets):
            meta = PPE_META[ppe.label]
            region_box = regions[CATEGORY_REGION[meta["category"]]]

            r_iou = iou(ppe.bbox, region_box)
            i_frac = inside_fraction(ppe.bbox, person.bbox)
            score = SCORE_W_IOU * r_iou + SCORE_W_INS * i_frac

            if score >= MIN_SCORE:
                candidates.append((pi, qi, score, r_iou, i_frac))

    # En yuksek skora gore sirala
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
    """
    Greedy one-to-one: her PPE kutusu en fazla bir kisiye atanir.
    Ayni kisi + label cifti icin sadece en yuksek skorlu tutulur.

    Dondurur:
        assignments   : {person_idx -> [PPEAssignment, ...]}
        unassigned    : atanmayan Detection listesi
    """
    assigned_ppe: set[int]   = set()   # ppe indeksleri
    # (person_idx, label) -> (ppe_idx, PPEAssignment)  — en iyi tut
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
) -> str:
    if not pos_scores and not neg_scores:
        return "unknown"
    if not neg_scores:
        return "ok"
    if not pos_scores:
        return "violation"
    best_pos = max(pos_scores)
    best_neg = max(neg_scores)
    if best_pos > best_neg * 1.5:
        return "ok"
    if best_neg > best_pos * 1.5:
        return "violation"
    return "unknown"


def fuse_person_ppe_state(
    person_idx: int,
    person_det: Detection,
    asgns:      list[PPEAssignment],
) -> PersonPPEState:
    """
    Atanmis PPE listesinden her kategori icin ok/violation/unknown uretir.
    Combined score = det.conf * assignment.score kullanilir.
    """
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
            entry = {"label": a.det.label, "conf": round(a.det.conf, 3),
                     "assoc_score": round(a.score, 3),
                     "combined": round(combined, 3)}
            if meta["polarity"] == "pos":
                pos.append(combined); ev_pos.append(entry)
            else:
                neg.append(combined); ev_neg.append(entry)

        evidence[cat] = {"positive": ev_pos, "negative": ev_neg}
        status[cat]   = _resolve_category(pos, neg)

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
         "bbox": [round(v, 1) for v in d.bbox]}
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
    """Kisi kutusu, PPE atamalar ve durum etiketlerini cizer."""
    x1, y1, x2, y2 = map(int, state.bbox)
    worst = _worst_status(state)
    box_color = _COL[worst]

    # Kisi kutusu
    _draw_box(img, x1, y1, x2, y2, box_color, thickness=3)

    # Anatomik bolgeler (opsiyonel)
    if show_regions and fh and fw:
        regs = person_regions(state.bbox, fh, fw)
        _draw_box(img, *regs["head"],  (200, 200, 80),  thickness=1)
        _draw_box(img, *regs["torso"], (80,  200, 200), thickness=1)

    # Durum etiketi (kisi kutusunun ust kisminda)
    def st_ch(st): return "OK" if st == "ok" else ("!!" if st == "violation" else "?")
    label = (f"#{state.person_idx}  "
             f"H:{st_ch(state.helmet_status)} "
             f"V:{st_ch(state.vest_status)} "
             f"M:{st_ch(state.mask_status)}")
    _put_label(img, label, x1, y1, box_color, font_scale=0.52)

    # Her atanmis PPE kutusunu ciz
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
    """Atanmamis PPE kutularini kesik kenarlı gosterir."""
    for d in unassigned:
        bx1, by1, bx2, by2 = map(int, d.bbox)
        col = _COL["unassigned"]
        # kesik cizgi efekti: sadece koseleri ciz
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


def render_hud(img, frame_idx: int, person_count: int, fps: float, conf: float) -> None:
    lines = [
        f"Frame: {frame_idx}  Persons: {person_count}",
        f"FPS: {fps:.0f}  conf: {conf:.2f}",
        "H=helmet  V=vest  M=mask  OK/!!/? ",
    ]
    scale, thick, lh = 0.55, 2, 22
    mw = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for l in lines)
    pw, ph = mw + 16, lh * len(lines) + 12
    cv2.rectangle(img, (6, 6), (6 + pw, 6 + ph), (20, 20, 20), -1)
    cv2.rectangle(img, (6, 6), (6 + pw, 6 + ph), (140, 140, 140), 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 6 + 18 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 10. Ana pipeline
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
            "ok":            c["ok"],
            "violation":     c["violation"],
            "unknown":       c["unknown"],
            "violation_rate": round(c["violation"] / known, 3) if known else None,
        }
    return {
        "total_person_frames": total_person_frames,
        "per_category":        rates,
    }


def process_video(
    video_path: Path,
    model,
    conf:         float,
    output_dir:   Path,
    show:         bool  = False,
    show_regions: bool  = False,
    device:       str   = "cpu",
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Video acilamadi: {video_path}")
        return

    fps_cap   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = video_path.stem
    out_vid_path  = output_dir / f"{stem}_scene_ppe.mp4"
    out_json_path = output_dir / f"{stem}_scene_ppe.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps_cap, (fw, fh))

    frame_summaries: list[FrameSummary] = []
    fps_ring: list[float] = []
    frame_idx = 0

    if show:
        cv2.namedWindow("Scene PPE Prototype", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Scene PPE Prototype", 1280, 720)

    print(f"  Islenecek: {video_path.name}  ({total_f} frame  {fps_cap:.0f}fps  {fw}x{fh})")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()

            # --- Tespit ---
            dets = collect_detections(model, frame, conf, device=device)

            # --- Ayir ---
            person_dets, ppe_dets = split_person_and_ppe(dets)

            # --- Atama ---
            person_states: list[PersonPPEState] = []
            unassigned_ppe: list[Detection]     = ppe_dets  # varsayilan

            if person_dets:
                candidates = build_assignment_candidates(
                    person_dets, ppe_dets, fh, fw
                )
                assignments, unassigned_ppe = assign_ppe_to_persons(
                    candidates, person_dets, ppe_dets
                )
                for pi, person_det in enumerate(person_dets):
                    state = fuse_person_ppe_state(
                        pi, person_det, assignments[pi]
                    )
                    person_states.append(state)

            # --- Ozet ---
            fs = build_frame_summary(frame_idx, person_states, unassigned_ppe)
            frame_summaries.append(fs)

            # --- Gorsellestime ---
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

            render_hud(draw, frame_idx, len(person_states), avg_fps, conf)

            writer.write(draw)
            if show:
                cv2.imshow("Scene PPE Prototype", draw)
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

    # --- JSON ---
    aggregate = _aggregate_stats(frame_summaries)
    summary = {
        "video":        str(video_path),
        "total_frames": frame_idx,
        "conf":         conf,
        "aggregate":    aggregate,
        "frames": [
            {
                "frame_idx":    fs.frame_idx,
                "person_count": fs.person_count,
                "persons":      fs.persons,
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
                    help="PPE atama minimum skoru")
    ap.add_argument("--output-dir",   default="runs/scene_ppe")
    ap.add_argument("--show",         action="store_true", help="Canli pencere")
    ap.add_argument("--show-regions", action="store_true",
                    help="Head/torso anatomik bolgeleri goster")
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    MIN_SCORE = args.min_score

    import torch
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    if not MODEL_PATH.exists():
        print(f"Model bulunamadi: {MODEL_PATH}")
        return

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / args.video
    if not video_path.exists():
        print(f"Video bulunamadi: {video_path}")
        return

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model yukleniyor: {MODEL_PATH.name}  device={device}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    process_video(
        video_path=video_path,
        model=model,
        conf=args.conf,
        output_dir=output_dir,
        show=args.show,
        show_regions=args.show_regions,
        device=device,
    )


if __name__ == "__main__":
    main()
