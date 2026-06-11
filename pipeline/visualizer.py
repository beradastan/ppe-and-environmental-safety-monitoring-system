from __future__ import annotations

import cv2

from pipeline.config import COLOR_OK, COLOR_WARN, COLOR_DANGER, COLOR_UNKNOWN

def draw_box(frame, x1: int, y1: int, x2: int, y2: int, text: str, color) -> None:
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

def draw_ppe_box(frame, x1: int, y1: int, x2: int, y2: int, label: str, color, tag: str) -> None:
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

def draw_hud(frame, event_id, status: str, repeat: int, viols_per_person: list) -> None:
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 200) if status in ("new", "active") else (0, 200, 0)
    cv2.putText(frame, f"EVENT: {event_id or 'N/A'} [{status.upper()}]",
                (10, 30), font, 0.7, color, 2)
    cv2.putText(frame, f"Repeat: {repeat}  Active violations: {len(viols_per_person)}",
                (10, 58), font, 0.6, color, 2)

def draw_ppe_annotations(
    frame,
    hbbox, hvote: str, hconf: float,
    vbbox, vvote: str, vconf: float,
    mbbox, mvote: str, mconf: float,
) -> None:
    ppe_items = [
        (hbbox, hvote, hconf, "H",
         COLOR_OK if hvote == "Hardhat"      else COLOR_DANGER if hvote == "NO-Hardhat"      else COLOR_UNKNOWN),
        (vbbox, vvote, vconf, "V",
         COLOR_OK if vvote == "Safety Vest"  else COLOR_DANGER if vvote == "NO-Safety Vest"  else COLOR_UNKNOWN),
        (mbbox, mvote, mconf, "M",
         COLOR_OK if mvote == "Mask"         else COLOR_WARN   if mvote == "NO-Mask"          else COLOR_UNKNOWN),
    ]
    for bbox, vote_label, conf, tag, c in ppe_items:
        if bbox is None or vote_label == "unknown":
            continue
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        draw_ppe_box(frame, bx1, by1, bx2, by2, f"{vote_label[:8]}({conf:.2f})", c, tag)
