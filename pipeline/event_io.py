from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import requests

from pipeline.config import get_backend_url

def build_alarm_text(event_type: str, persons: list[dict]) -> str:
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

def notify_camera_status(
    status: str,
    camera_id: str | None = None,
    zone: str | None = None,
) -> None:
    try:
        requests.post(
            f"{get_backend_url()}/api/pipeline/camera-status",
            json={"status": status, "camera_id": camera_id or "", "zone": zone or ""},
            timeout=2,
        )
        print(f"  [KAMERA] Durum: {status}")
    except Exception:
        pass

def close_event(
    event_id: str,
    repeat_count: int | None = None,
    duration_sec: float | None = None,
) -> None:
    try:
        body: dict = {}
        if repeat_count is not None:
            body["repeat_count"] = repeat_count
        if duration_sec is not None:
            body["duration_sec"] = duration_sec
        requests.patch(
            f"{get_backend_url()}/api/events/{event_id}/close",
            json=body,
            timeout=5,
        )
        print(f"  [CLOSE] {event_id} kapatildi.")
    except Exception as exc:
        print(f"  [CLOSE] Hata: {exc}")

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

    has_ppe  = (base_sig.get("helmet_violation") or base_sig.get("vest_violation")
                or base_sig.get("mask_violation"))
    has_fire = base_sig.get("fire_detected", False)
    if has_ppe and has_fire:
        event_type = "multi_hazard"
    elif has_fire:
        event_type = "fire_detected"
    else:
        event_type = "ppe_violation"

    dur_map = {
        p["track_id"]: p.get("duration_sec", 0.0)
        for p in event_info.get("person_violations", [])
    }

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

    alarm_text   = build_alarm_text(event_type, persons_detail)
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
        "scene": {
            "fire_detected":  has_fire,
            "fire_conf":      round(fire_conf, 2),
            "smoke_detected": smoke_detected,
            "smoke_conf":     round(smoke_conf, 2),
        },
        "signature":  enriched_sig,
        "alarm_text": alarm_text,
        "llm_report": None,
        "camera_id":  camera_id,
        "zone":       zone,
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

def cleanup_old_results(results_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("evt_")],
        key=lambda d: d.name,
    )
    for d in dirs[:-keep] if len(dirs) > keep else []:
        try:
            shutil.rmtree(d)
            print(f"  [TEMIZLIK] Silindi: {d.name}")
        except Exception as exc:
            print(f"  [TEMIZLIK] Hata ({d.name}): {exc}")
