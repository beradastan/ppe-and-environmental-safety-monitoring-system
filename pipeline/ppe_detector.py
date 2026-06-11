from __future__ import annotations

from collections import Counter, deque

from pipeline.config import (
    MIN_CROP_PX, MIN_HEAD_PX, MIN_TORSO_PX,
    PPE_MIN_SELF_SCORE, PPE_MAX_NEIGHBOR_RATIO,
    INSIDE_FRAC_THR, IMGSZ,
    COLOR_OK, COLOR_WARN, COLOR_DANGER, COLOR_UNKNOWN,
)

def vote(q: deque, min_known: int = 3, ratio_threshold: float = 0.5) -> str:
    if not q:
        return "unknown"
    top_label = Counter(q).most_common(1)[0][0]
    if top_label != "unknown":
        return top_label
    known = [v for v in q if v != "unknown"]
    if len(known) < min_known:
        return "unknown"
    best, count = Counter(known).most_common(1)[0]
    return best if count / len(known) >= ratio_threshold else "unknown"

def compliance_color(
    hvote: str, vvote: str, mvote: str,
) -> tuple[tuple[int, int, int], list[str]]:
    viols: list[str] = []
    if hvote == "NO-Hardhat":     viols.append("no_helmet")
    if vvote == "NO-Safety Vest": viols.append("no_vest")
    if mvote == "NO-Mask":        viols.append("no_mask")

    if hvote == "Hardhat" and vvote == "Safety Vest" and mvote == "Mask":
        return COLOR_OK, []
    if viols:
        return (COLOR_DANGER if len(viols) >= 2 else COLOR_WARN), viols
    return COLOR_UNKNOWN, viols

def crop_ppe(frame, x1: int, y1: int, x2: int, y2: int, ppe: str):
    fh, fw = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if ppe == "helmet":
        cx1 = max(0,  x1 - int(pw * 0.10))
        cy1 = max(0,  y1 - int(ph * 0.15))
        cx2 = min(fw, x2 + int(pw * 0.10))
        cy2 = min(fh, y1 + int(ph * 0.40))
    elif ppe == "vest":
        cx1 = max(0,  x1 - int(pw * 0.15))
        cy1 = max(0,  y1 + int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.90))
    else:
        cx1 = max(0,  x1 - int(pw * 0.15))
        cy1 = max(0,  y1 - int(ph * 0.10))
        cx2 = min(fw, x2 + int(pw * 0.15))
        cy2 = min(fh, y1 + int(ph * 0.45))
    return frame[cy1:cy2, cx1:cx2], cx1, cy1

def crop_ok(crop) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return h >= MIN_CROP_PX and w >= MIN_CROP_PX

def crop_to_frame(bbox, ox: int, oy: int, fh: int, fw: int) -> tuple[int, int, int, int]:
    dx1, dy1, dx2, dy2 = bbox
    return (
        min(fw - 1, int(ox + dx1)),
        min(fh - 1, int(oy + dy1)),
        min(fw - 1, int(ox + dx2)),
        min(fh - 1, int(oy + dy2)),
    )

def anatomical_region(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> list[int]:
    pw, ph = x2 - x1, y2 - y1
    if ppe_type == "helmet":
        return [x1 + int(pw * 0.05), y1 - int(ph * 0.10),
                x2 - int(pw * 0.05), y1 + int(ph * 0.35)]
    if ppe_type == "mask":
        return [x1 + int(pw * 0.15), y1,
                x2 - int(pw * 0.15), y1 + int(ph * 0.28)]
    return [x1, y1 + int(ph * 0.15), x2, y1 + int(ph * 0.85)]

def is_region_too_small(x1: int, y1: int, x2: int, y2: int, ppe_type: str) -> bool:
    region = anatomical_region(x1, y1, x2, y2, ppe_type)
    w = region[2] - region[0]
    return w < (MIN_HEAD_PX if ppe_type in ("helmet", "mask") else MIN_TORSO_PX)

def _containment(inner: list, outer: list) -> float:
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area

def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)

def collect_dets(model, result, allowed_ids: list[int], min_conf: float) -> list[dict]:
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

def validate_ppe_scored(
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

def global_assign_ppe(candidates: list[dict], iou_thresh: float = 0.40) -> dict[int, dict]:
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

def scene_dets(
    model,
    frame,
    allowed_ids: set[int],
    min_conf: float,
    device: str,
    half: bool,
) -> list[tuple[str, float, list]]:
    res = model.predict(frame, imgsz=IMGSZ, conf=min_conf,
                        device=device, half=half, verbose=False)[0]
    if not res.boxes:
        return []
    return [
        (str(model.names[int(b.cls[0])]), float(b.conf[0]), b.xyxy[0].tolist())
        for b in res.boxes
        if int(b.cls[0]) in allowed_ids
    ]

def best_scene(
    dets: list[tuple[str, float, list]],
    person_box: list,
) -> tuple[str, float, list | None]:
    best = None
    for lbl, c, bbox in dets:
        if _inside_frac(bbox, person_box) >= INSIDE_FRAC_THR:
            if best is None or c > best[1]:
                best = (lbl, c, bbox)
    return best if best else ("unknown", 0.0, None)
