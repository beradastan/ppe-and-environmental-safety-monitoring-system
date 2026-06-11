from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2

import os
os.chdir(Path(__file__).resolve().parents[1])

from ultralytics import YOLO

from pipeline.camera_monitor import CameraMonitor
from pipeline.config import (
    CROP_HELMET_MODEL_PATH, CROP_MASK_MODEL_PATH, CROP_VEST_MODEL_PATH,
    DEVICE, FIRE_INFER_EVERY, FIRE_MODEL_PATH,
    HELMET_CLASSES, IMGSZ, MASK_CLASSES, PERSON_MODEL_PATH,
    PIPELINE_MAX_WIDTH, SCENE_HELMET_MODEL_PATH, SCENE_MASK_MODEL_PATH,
    SCENE_VEST_MODEL_PATH, TRACKER, VEST_CLASSES,
    COLOR_FIRE,
    class_ids, load_full_cfg,
)
from pipeline.event_io import cleanup_old_results, close_event, notify_camera_status, save_event
from pipeline.event_manager import PersonEventManager
from pipeline.fire_smoke_detector import FireSmokeDetector
from pipeline.ppe_processor import PPEModels, PPEProcessor, PPEThresholds
from pipeline.tracking_identity import TrackReattacher
from pipeline.visualizer import draw_hud

def run(args) -> None:
    device = args.device
    half   = (device == "cuda")
    mode   = args.mode

    print(f"Modeller yukleniyor... (mod: {mode})")
    person_model = YOLO(PERSON_MODEL_PATH)
    fire_model   = YOLO(FIRE_MODEL_PATH)

    if mode == "crop":
        helmet_model = YOLO(CROP_HELMET_MODEL_PATH)
        vest_model   = YOLO(CROP_VEST_MODEL_PATH)
        mask_model   = YOLO(CROP_MASK_MODEL_PATH)
        print(f"  Crop modelleri: {CROP_HELMET_MODEL_PATH}")
    else:
        helmet_model = YOLO(SCENE_HELMET_MODEL_PATH)
        vest_model   = YOLO(SCENE_VEST_MODEL_PATH)
        mask_model   = YOLO(SCENE_MASK_MODEL_PATH)
        print(f"  Scene modelleri: {SCENE_HELMET_MODEL_PATH}")

    _person_cls = next(n for n in person_model.names.values() if n.lower() == "person")
    p_ids = class_ids(person_model, [_person_cls])
    h_ids = class_ids(helmet_model, HELMET_CLASSES)
    v_ids = class_ids(vest_model,   VEST_CLASSES)
    m_ids = class_ids(mask_model,   MASK_CLASSES)

    _full_cfg = load_full_cfg()
    _ppe_cfg  = _full_cfg.get("ppe_pipeline", {})
    _em_cfg   = _full_cfg.get("event_manager", {})

    _temporal_win = (
        int(_ppe_cfg.get("scene_temporal_window", 30)) if mode == "scene"
        else int(_ppe_cfg.get("temporal_window", 20))
    )

    ppe_processor = PPEProcessor(
        mode=mode,
        models=PPEModels(helmet_model, vest_model, mask_model, h_ids, v_ids, m_ids),
        thresholds=PPEThresholds(
            crop_helmet  = float(_ppe_cfg.get("crop_helmet_conf",  0.20)),
            crop_vest    = float(_ppe_cfg.get("crop_vest_conf",    0.30)),
            crop_mask    = float(_ppe_cfg.get("crop_mask_conf",    0.25)),
            scene_helmet = float(_ppe_cfg.get("scene_helmet_conf", 0.25)),
            scene_vest   = float(_ppe_cfg.get("scene_vest_conf",   0.30)),
            scene_mask   = float(_ppe_cfg.get("scene_mask_conf",   0.05)),
            device=device,
            half=half,
        ),
        temporal_window=_temporal_win,
    )

    _fire_conf   = float(_ppe_cfg.get("fire_conf",   0.75))
    _person_conf = float(_ppe_cfg.get("person_conf", 0.25))
    _use_fire    = bool( _ppe_cfg.get("use_fire",    True))

    fire_smoke_detector = FireSmokeDetector(
        fire_min_area      = float(_ppe_cfg.get("fire_min_area_ratio", 0.01)),
        fire_growth_window = int(  _ppe_cfg.get("fire_growth_window",  10)),
        fire_growth_factor = float(_ppe_cfg.get("fire_growth_factor",  1.5)),
    )

    cam_monitor = CameraMonitor(
        freeze_frames = int(  _ppe_cfg.get("cam_freeze_frames", 60)),
        dark_frames   = int(  _ppe_cfg.get("cam_dark_frames",   60)),
        freeze_diff   = float(_ppe_cfg.get("cam_freeze_diff",   0.002)),
        dark_thresh   = float(_ppe_cfg.get("cam_dark_thresh",   0.03)),
    )

    event_manager = PersonEventManager(
        resolved_confirm_sec = float(_em_cfg.get("resolved_confirm_sec", 5.0)),
        fire_confirm_frames  = int(  _em_cfg.get("fire_confirm_frames",  20)),
        fire_clear_frames    = int(  _em_cfg.get("fire_clear_frames",    10)),
        check_helmet = _ppe_cfg.get("use_helmet", True),
        check_vest   = _ppe_cfg.get("use_vest",   True),
        check_mask   = _ppe_cfg.get("use_mask",   False),
    )
    reattacher = TrackReattacher()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    cleanup_old_results(results_dir, int(_full_cfg.get("results_keep_events", 50)))

    existing      = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("evt_")]
    start_counter = max((int(d.name.split("_")[1]) for d in existing), default=0)
    event_manager._counter = start_counter
    if start_counter:
        print(f"  Mevcut {start_counter} event bulundu, sayac {start_counter}'den devam ediyor.")

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        sys.exit(f"Kaynak acilamadi: {args.camera}")

    if args.display:
        cv2.namedWindow("Canli Goruntu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canli Goruntu", 1280, 720)

    frame_idx       = 0
    event_count     = 0
    _fire_raw       = False
    _fire_conf_max  = 0.0
    _smoke_raw      = False
    _smoke_conf_max = 0.0

    print("Basladi." + (" ESC = cikis." if args.display else " Ctrl+C = cikis.") + f" [{mode.upper()} modu]\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if cam_monitor.status != "offline":
                    notify_camera_status("offline", args.camera_id, args.zone)
                break
            frame_idx += 1

            if PIPELINE_MAX_WIDTH and frame.shape[1] > PIPELINE_MAX_WIDTH:
                scale = PIPELINE_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (PIPELINE_MAX_WIDTH, int(frame.shape[0] * scale)))

            new_cam_status = cam_monitor.update(frame)
            if new_cam_status is not None:
                notify_camera_status(new_cam_status, args.camera_id, args.zone)

            draw_frame = frame.copy() if args.display else frame
            fh, fw     = frame.shape[:2]

            p_result = person_model.track(
                frame, classes=p_ids, tracker=TRACKER, persist=True,
                imgsz=IMGSZ, conf=_person_conf, device=device, half=half, verbose=False,
            )[0]

            persons_with_ppe:  list[dict]     = []
            boxes                              = p_result.boxes
            all_persons_frame: list[dict]     = []
            stable_map:        dict[int, int] = {}

            if boxes is not None and boxes.id is not None:
                all_persons_frame = [
                    {"tid": int(tid), "box": list(map(int, box.tolist()))}
                    for box, tid in zip(boxes.xyxy, boxes.id)
                ]
                stable_map = reattacher.update(all_persons_frame)

                persons_with_ppe = ppe_processor.process_frame(
                    frame, draw_frame, boxes, stable_map, all_persons_frame,
                    frame_idx, args.display, fh, fw,
                )

            if _use_fire and frame_idx % FIRE_INFER_EVERY == 0:
                fire_res = fire_model.predict(
                    frame, imgsz=IMGSZ, conf=_fire_conf, device=device, half=half, verbose=False,
                )[0]
                frame_area = (fh * fw) or 1
                _fire_raw, _fire_conf_max, _smoke_raw, _smoke_conf_max = (
                    fire_smoke_detector.update(fire_res, frame_area)
                )

            if _fire_raw or _smoke_raw:
                label = "FIRE" if _fire_raw else "SMOKE"
                cv2.putText(draw_frame, f"{label} DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_FIRE, 3)

            event_info = event_manager.process_frame(persons_with_ppe, _fire_raw or _smoke_raw)

            if event_info["should_save"] and event_info.get("event_id"):
                event_count += 1
                save_event(
                    event_info, draw_frame, results_dir, persons_with_ppe,
                    fire_conf      = _fire_conf_max,
                    smoke_detected = _smoke_raw,
                    smoke_conf     = _smoke_conf_max,
                    camera_id      = args.camera_id,
                    zone           = args.zone,
                )

            if event_info["event_status"] == "closed" and event_info.get("event_id"):
                threading.Thread(
                    target=close_event,
                    args=(event_info["event_id"],
                          event_info.get("repeat_count"),
                          event_info.get("duration_sec")),
                    daemon=True,
                ).start()

            if args.display:
                draw_hud(
                    draw_frame,
                    event_info.get("event_id"),
                    event_info["event_status"],
                    event_info.get("repeat_count", 0),
                    event_info.get("person_violations", []),
                )
                cv2.imshow("Canli Goruntu", draw_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if frame_idx % 60 == 0:
                print(f"  Frame {frame_idx} | Events: {event_count} | Status: {event_info['event_status']}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        notify_camera_status("online", args.camera_id, args.zone)
        print(f"\nToplam frame: {frame_idx} | Kaydedilen event: {event_count}")
        print("Sonuclar: results/")
        if event_manager._active is not None:
            ev = event_manager._active
            close_event(ev.event_id, ev.repeat_count, ev.duration_sec)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",    default=0,      help="Kamera indeksi (varsayilan: 0)")
    parser.add_argument("--mode",      default="crop", choices=["crop", "scene"],
                        help="Tespit modu: crop-based veya scene-based (varsayilan: crop)")
    parser.add_argument("--device",    default=DEVICE, help="cuda veya cpu")
    parser.add_argument("--display",   action="store_true", help="OpenCV penceresi goster")
    parser.add_argument("--camera-id", default=None,   dest="camera_id",
                        help="Kamera kimligi (orn: cam_01)")
    parser.add_argument("--zone",      default=None,   help="Kamera bolgesi (orn: Uretim Hatti A)")
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())
