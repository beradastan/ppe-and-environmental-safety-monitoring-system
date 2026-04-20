# -*- coding: utf-8 -*-
"""
run_live_video.py
=================
Crop-tabanlı PPE pipeline — nihai modeller:
  Helmet : crophelmet_agent_final_best.pt  pad=0.80  conf=0.20
  Vest   : vest_agent_final_best.pt        pad=0.60  conf=0.30

Kullanim:
    python run_live_video.py
    python run_live_video.py --video test/nohat_test.mp4
    python run_live_video.py --camera 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
import cv2
from collections import Counter, defaultdict, deque
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
# Sabitler
# ---------------------------------------------------------------------------

HELMET_MODEL_PATH = "models/crophelmet_agent_final_best.pt"
VEST_MODEL_PATH   = "models/vest_agent_final_best.pt"
FIRE_MODEL_PATH   = "models/fire_best.pt"
PERSON_MODEL_PATH = "models/pretrained/person/person_yolov8s-seg.pt"

HELMET_PAD   = 0.80
VEST_PAD     = 0.60
HELMET_CONF  = 0.20
VEST_CONF    = 0.30
FIRE_CONF    = 0.50
PERSON_CONF  = 0.25
IMGSZ        = 640
TRACKER      = "bytetrack.yaml"
TEMPORAL_WIN = 10

HELMET_CLASSES = ["Hardhat", "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest", "NO-Safety Vest"]
FIRE_CLASS     = "fire"

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


def best_det(model: YOLO, result, allowed_ids: list[int], min_conf: float) -> tuple[str, float]:
    best: tuple[str, float] | None = None
    for box in result.boxes:
        cid  = int(box.cls[0])
        conf = float(box.conf[0])
        if cid not in allowed_ids or conf < min_conf:
            continue
        label = str(model.names[cid])
        if best is None or conf > best[1]:
            best = (label, conf)
    return best if best else ("unknown", 0.0)


def vote(q: deque) -> str:
    if not q:
        return "unknown"
    return Counter(q).most_common(1)[0][0]


def compliance_color(hvote: str, vvote: str) -> tuple[tuple[int, int, int], list[str]]:
    h_ok   = hvote == "Hardhat"
    v_ok   = vvote == "Safety Vest"
    h_miss = hvote == "NO-Hardhat"
    v_miss = vvote == "NO-Safety Vest"
    viols  = []
    if h_miss:
        viols.append("no_helmet")
    if v_miss:
        viols.append("no_vest")
    if h_ok and v_ok:
        return COLOR_OK, []
    if h_miss and v_miss:
        return COLOR_DANGER, viols
    if h_miss or v_miss:
        return COLOR_WARN, viols
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

def save_event(event_info: dict, frame, results_dir: Path, update_counters: dict) -> None:
    event_id     = event_info["event_id"]
    event_status = event_info["event_status"]

    event_dir = results_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    if event_status == "new":
        suffix = "new"
    elif event_status == "resolved":
        suffix = "resolved"
    else:
        n = update_counters.get(event_id, 0) + 1
        update_counters[event_id] = n
        suffix = f"update_{n:02d}"

    json_path = event_dir / f"{event_id}_{suffix}.json"
    img_path  = event_dir / f"{event_id}_{suffix}.jpg"

    # person_violations'dan hangi track ID'lerin ihlali var → frontend için
    person_viols = event_info.get("person_violations", [])
    helmet_ids = [p["track_id"] for p in person_viols if "no_helmet" in p.get("violations", [])]
    vest_ids   = [p["track_id"] for p in person_viols if "no_vest"   in p.get("violations", [])]
    base_sig   = event_info.get("signature", {})
    signature  = {
        **base_sig,
        "helmet_missing_ids": helmet_ids,
        "vest_missing_ids":   vest_ids,
        "fire_detected":      base_sig.get("fire_detected", False),
    }

    payload = {
        "event_id":     event_id,
        "event_status": event_status,
        "timestamp":    datetime.now().isoformat(),
        "repeat_count": event_info.get("repeat_count", 0),
        "duration_sec": event_info.get("duration_sec", 0.0),
        "change_reason": event_info.get("change_reason", ""),
        "signature":    signature,
        "llm_report":   None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    cv2.imwrite(str(img_path), frame)
    print(f"  [KAYIT] {event_id}/{suffix}")

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
    fire_model   = YOLO(FIRE_MODEL_PATH)
    print("  person, helmet, vest, fire modelleri hazir.")

    p_ids = class_ids(person_model, ["person"])
    h_ids = class_ids(helmet_model, HELMET_CLASSES)
    v_ids = class_ids(vest_model,   VEST_CLASSES)
    f_ids = [cid for cid, name in fire_model.names.items() if name == FIRE_CLASS]

    from event_manager import PersonEventManager
    event_manager = PersonEventManager(
        new_confirm_sec=3.0,
        resolved_confirm_sec=5.0,
    )

    states = defaultdict(lambda: {
        "hardhat": deque(maxlen=TEMPORAL_WIN),
        "vest":    deque(maxlen=TEMPORAL_WIN),
    })

    results_dir     = Path("results")
    update_counters: dict[str, int] = {}

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(str(source) if args.video else int(source))
    if not cap.isOpened():
        sys.exit(f"Kaynak acilamadi: {source}")

    cv2.namedWindow("Factory Safety", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Factory Safety", 1280, 720)

    frame_idx        = 0
    event_count      = 0
    last_event       = {"event_id": None, "event_status": "idle", "repeat_count": 0}
    last_viols       = []
    prev_violator_ids: set[int] = set()  # ihlalci seti değişimini izlemek için

    print("Basladi. ESC = cikis.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # --- Person tracking ---
            p_result = person_model.track(
                frame, classes=p_ids, tracker=TRACKER, persist=True,
                imgsz=IMGSZ, conf=PERSON_CONF, device=device, verbose=False,
            )[0]

            persons_with_ppe: list[dict] = []
            draw_frame = frame.copy()

            boxes = p_result.boxes
            if boxes is not None and boxes.id is not None:
                for box, tid in zip(boxes.xyxy, boxes.id):
                    track_id = int(tid)
                    x1, y1, x2, y2 = map(int, box.tolist())

                    # Helmet
                    hcrop = crop_pad(frame, x1, y1, x2, y2, HELMET_PAD)
                    hres  = helmet_model.predict(
                        hcrop, classes=h_ids, imgsz=IMGSZ,
                        conf=HELMET_CONF, device=device, verbose=False,
                    )[0]
                    hlabel, hconf = best_det(helmet_model, hres, h_ids, HELMET_CONF)
                    states[track_id]["hardhat"].append(hlabel)
                    hvote = vote(states[track_id]["hardhat"])

                    # Vest
                    vcrop = crop_pad(frame, x1, y1, x2, y2, VEST_PAD)
                    vres  = vest_model.predict(
                        vcrop, classes=v_ids, imgsz=IMGSZ,
                        conf=VEST_CONF, device=device, verbose=False,
                    )[0]
                    vlabel, vconf = best_det(vest_model, vres, v_ids, VEST_CONF)
                    states[track_id]["vest"].append(vlabel)
                    vvote = vote(states[track_id]["vest"])

                    color, viols = compliance_color(hvote, vvote)
                    persons_with_ppe.append({"track_id": track_id, "violations": viols})

                    label = (
                        f"ID{track_id} "
                        f"H:{hvote[:6]}({hconf:.2f}) "
                        f"V:{vvote[:6]}({vconf:.2f})"
                    )
                    draw_box(draw_frame, x1, y1, x2, y2, label, color)

            # --- Fire detection ---
            fire_res = fire_model.predict(
                frame, classes=f_ids if f_ids else None,
                imgsz=IMGSZ, conf=FIRE_CONF, device=device, verbose=False,
            )[0]
            fire_raw = bool(fire_res.boxes and len(fire_res.boxes) > 0)
            if fire_raw:
                cv2.putText(draw_frame, "FIRE DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FIRE, 3)

            # --- Event state machine ---
            event_info = event_manager.process_frame(persons_with_ppe, fire_raw)
            last_event = event_info
            last_viols = event_info.get("person_violations", [])

            curr_violator_ids = {p["track_id"] for p in last_viols}

            # "new" → ilk ihlal kaydı
            if event_info["should_save"] and event_info.get("event_id"):
                event_count += 1
                save_event(event_info, draw_frame, results_dir, update_counters)

            # ihlalci seti daraldı (biri çıktı) ama hâlâ ihlal devam ediyor → update kaydet
            elif (
                event_info["event_status"] == "active"
                and event_info.get("event_id")
                and prev_violator_ids - curr_violator_ids  # biri çıktı
                and curr_violator_ids                       # ama başkası kaldı
            ):
                save_event(event_info, draw_frame, results_dir, update_counters)

            prev_violator_ids = curr_violator_ids

            # --- HUD ---
            draw_hud(
                draw_frame,
                event_info.get("event_id"),
                event_info["event_status"],
                event_info.get("repeat_count", 0),
                last_viols,
            )

            cv2.imshow("Factory Safety", draw_frame)
            if frame_idx % 60 == 0:
                print(f"  Frame {frame_idx} | Events: {event_count} | Status: {event_info['event_status']}")

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
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
    parser.add_argument("--device", default=_DEVICE, help="cuda device veya cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
