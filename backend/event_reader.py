# -*- coding: utf-8 -*-
"""
event_reader.py
===============
results/ dizinini okuyup Python dict'lerine çeviren saf yardımcı modül.
Flask ve watcher bu modülü kullanır; doğrudan dosya sistemi erişimi yapmaz.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

# evt_0001_new.json / evt_0001_update_03.jpg / evt_0001_closed.json
_FILENAME_RE = re.compile(
    r"^(evt_\d+)_(new|update_(\d+)|closed)\.(json|jpg)$"
)
_EVENT_ID_RE = re.compile(r"^evt_\d+$")


def parse_filename(filename: str) -> dict | None:
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    event_id, raw_status, update_num_str, ext = m.group(1), m.group(2), m.group(3), m.group(4)
    if raw_status == "new":
        status, update_num = "new", None
    elif raw_status == "closed":
        status, update_num = "closed", None
    else:
        status, update_num = "update", int(update_num_str)
    return {"event_id": event_id, "status": status, "update_num": update_num, "ext": ext}


def read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _status_sort_key(meta: dict) -> tuple:
    if meta["status"] == "new":
        return (0,)
    if meta["status"] == "update":
        return (meta["update_num"],)
    return (999_999,)


def _build_event_summary(event_id: str, data: dict, latest_path: Path) -> dict:
    jpg_stem = latest_path.stem
    has_image = (latest_path.parent / f"{jpg_stem}.jpg").exists()
    return {
        "event_id":     data.get("event_id", event_id),
        "event_status": data.get("event_status", ""),
        "timestamp":    data.get("timestamp", ""),
        "repeat_count": data.get("repeat_count", 0),
        "duration_sec": data.get("duration_sec", 0.0),
        "signature":    data.get("signature", {}),
        "llm_report":   data.get("llm_report"),
        "has_image":    has_image,
        "camera_id":    data.get("camera_id"),
        "zone":         data.get("zone"),
    }


def get_all_events(results_dir: Path) -> list[dict]:
    if not results_dir.exists():
        return []

    events: list[dict] = []
    for event_dir in sorted(results_dir.iterdir(), reverse=True):
        if not event_dir.is_dir() or not _EVENT_ID_RE.match(event_dir.name):
            continue
        event_id = event_dir.name
        json_files: list[tuple] = []
        for f in event_dir.iterdir():
            if f.suffix != ".json":
                continue
            meta = parse_filename(f.name)
            if meta is None or meta["event_id"] != event_id:
                continue
            json_files.append((_status_sort_key(meta), f, meta))
        if not json_files:
            continue
        json_files.sort(key=lambda x: x[0])
        _, latest_path, _ = json_files[-1]
        data = read_json_file(latest_path)
        if not data:
            continue
        events.append(_build_event_summary(event_id, data, latest_path))

    return events


def get_filtered_events(
    results_dir: Path,
    date_str: str | None = None,
    violation_type: str | None = None,
    status: str | None = None,
) -> list[dict]:
    events = get_all_events(results_dir)
    if date_str:
        events = [e for e in events if e["timestamp"].startswith(date_str)]
    if violation_type:
        vmap = {
            "helmet": lambda e: bool(e["signature"].get("helmet_missing_ids")),
            "vest":   lambda e: bool(e["signature"].get("vest_missing_ids")),
            "mask":   lambda e: bool(e["signature"].get("mask_missing_ids")),
            "fire":   lambda e: bool(e["signature"].get("fire_detected")),
        }
        fn = vmap.get(violation_type)
        if fn:
            events = [e for e in events if fn(e)]
    if status:
        events = [e for e in events if e["event_status"] == status]
    return events


def get_stats(results_dir: Path) -> dict:
    all_events = get_all_events(results_dir)
    today = date.today().isoformat()

    active   = sum(1 for e in all_events if e["event_status"] in ("new", "active"))
    today_ct = sum(1 for e in all_events if e["timestamp"].startswith(today))

    dist = {"helmet": 0, "vest": 0, "mask": 0, "fire": 0}
    for e in all_events:
        sig = e.get("signature", {})
        if sig.get("helmet_missing_ids"): dist["helmet"] += 1
        if sig.get("vest_missing_ids"):   dist["vest"]   += 1
        if sig.get("mask_missing_ids"):   dist["mask"]   += 1
        if sig.get("fire_detected"):      dist["fire"]   += 1

    return {
        "total_events":      len(all_events),
        "active_alarms":     active,
        "today_violations":  today_ct,
        "distribution":      dist,
        "recent":            all_events[:5],
    }


def get_report_data(results_dir: Path, period: str, date_str: str | None = None) -> list[dict]:
    all_events = get_all_events(results_dir)
    today = date.today()

    def _inc(buckets: dict, key: str, sig: dict) -> None:
        if key not in buckets:
            return
        if sig.get("helmet_missing_ids"): buckets[key]["helmet"] += 1
        if sig.get("vest_missing_ids"):   buckets[key]["vest"]   += 1
        if sig.get("mask_missing_ids"):   buckets[key]["mask"]   += 1
        if sig.get("fire_detected"):      buckets[key]["fire"]   += 1
        buckets[key]["total"] += 1

    def _empty() -> dict:
        return {"helmet": 0, "vest": 0, "mask": 0, "fire": 0, "total": 0}

    if period == "daily":
        target = date.fromisoformat(date_str) if date_str else today
        buckets = {f"{h:02d}:00": _empty() for h in range(24)}
        for e in all_events:
            ts = e.get("timestamp", "")
            if not ts.startswith(target.isoformat()):
                continue
            try:
                hour = datetime.fromisoformat(ts).hour
            except ValueError:
                continue
            _inc(buckets, f"{hour:02d}:00", e.get("signature", {}))
        return [{"label": k, **v} for k, v in sorted(buckets.items())]

    elif period == "weekly":
        days = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    else:  # monthly
        days = [(today - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]

    buckets = {d: _empty() for d in days}
    for e in all_events:
        _inc(buckets, e.get("timestamp", "")[:10], e.get("signature", {}))
    return [{"label": k, **v} for k, v in buckets.items()]


def get_event_timeline(results_dir: Path, event_id: str) -> list[dict]:
    event_dir = results_dir / event_id
    if not event_dir.exists():
        return []

    steps: list[tuple] = []
    for f in event_dir.iterdir():
        if f.suffix != ".json":
            continue
        meta = parse_filename(f.name)
        if meta is None or meta["event_id"] != event_id:
            continue
        steps.append((_status_sort_key(meta), f, meta))

    steps.sort(key=lambda x: x[0])

    timeline: list[dict] = []
    for _, path, meta in steps:
        data = read_json_file(path)
        if not data:
            continue
        jpg_name = path.stem + ".jpg"
        image_filename = jpg_name if (event_dir / jpg_name).exists() else None
        timeline.append({
            "event_id":       data.get("event_id", event_id),
            "event_status":   data.get("event_status", meta["status"]),
            "timestamp":      data.get("timestamp", ""),
            "repeat_count":   data.get("repeat_count", 0),
            "duration_sec":   data.get("duration_sec", 0.0),
            "change_reason":  data.get("change_reason", ""),
            "signature":      data.get("signature", {}),
            "llm_report":     data.get("llm_report"),
            "image_filename": image_filename,
            "camera_id":      data.get("camera_id"),
            "zone":           data.get("zone"),
        })

    return timeline


def get_notes(results_dir: Path, event_id: str) -> list[dict]:
    notes_path = results_dir / event_id / "notes.json"
    if not notes_path.exists():
        return []
    try:
        return json.loads(notes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_note(results_dir: Path, event_id: str, text: str) -> dict:
    event_dir = results_dir / event_id
    notes_path = event_dir / "notes.json"
    notes = get_notes(results_dir, event_id)
    entry = {"timestamp": datetime.now().isoformat(), "text": text}
    notes.append(entry)
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry
