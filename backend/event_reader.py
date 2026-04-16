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
from pathlib import Path

# evt_0001_new.json / evt_0001_update_03.jpg / evt_0001_resolved.json
_FILENAME_RE = re.compile(
    r"^(evt_\d+)_(new|update_(\d+)|resolved)\.(json|jpg)$"
)
_EVENT_ID_RE = re.compile(r"^evt_\d+$")


def parse_filename(filename: str) -> dict | None:
    """
    Dosya adından event meta verisini çıkar.

    Returns:
        {
            "event_id":   "evt_0001",
            "status":     "new" | "update" | "resolved",
            "update_num": int | None,   # sadece update için
            "ext":        "json" | "jpg",
        }
        veya None (tanınmayan dosya adı).
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        return None

    event_id, raw_status, update_num_str, ext = m.group(1), m.group(2), m.group(3), m.group(4)

    if raw_status == "new":
        status = "new"
        update_num = None
    elif raw_status == "resolved":
        status = "resolved"
        update_num = None
    else:
        status = "update"
        update_num = int(update_num_str)

    return {
        "event_id":   event_id,
        "status":     status,
        "update_num": update_num,
        "ext":        ext,
    }


def read_json_file(path: Path) -> dict:
    """
    JSON dosyasını oku. Hata durumunda boş dict döndür (asla raise etmez).
    Watchdog'un henüz yazılmamış dosya yarışını tolere eder.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _status_sort_key(meta: dict) -> tuple:
    """JSON dosyalarını sıralamak için: new=0, update_N=N, resolved=999999."""
    if meta["status"] == "new":
        return (0,)
    if meta["status"] == "update":
        return (meta["update_num"],)
    return (999_999,)


def get_all_events(results_dir: Path) -> list[dict]:
    """
    results/ altındaki tüm event klasörlerini tara.
    Her event için en güncel durum JSON'ını okur.

    Returns:
        Listesi: event_id'ye göre azalan sırada (en yeni önce).
    """
    if not results_dir.exists():
        return []

    events: list[dict] = []

    for event_dir in sorted(results_dir.iterdir(), reverse=True):
        if not event_dir.is_dir() or not _EVENT_ID_RE.match(event_dir.name):
            continue

        event_id = event_dir.name
        json_files: list[tuple] = []  # (sort_key, path, meta)

        for f in event_dir.iterdir():
            if f.suffix != ".json":
                continue
            meta = parse_filename(f.name)
            if meta is None or meta["event_id"] != event_id:
                continue
            json_files.append((_status_sort_key(meta), f, meta))

        if not json_files:
            continue

        # En son durumu al: resolved > update_N > new
        json_files.sort(key=lambda x: x[0])
        _, latest_path, _ = json_files[-1]

        data = read_json_file(latest_path)
        if not data:
            continue

        # İlişkili JPG var mı?
        jpg_stem = latest_path.stem
        has_image = (latest_path.parent / f"{jpg_stem}.jpg").exists()

        events.append({
            "event_id":     data.get("event_id", event_id),
            "event_status": data.get("event_status", ""),
            "timestamp":    data.get("timestamp", ""),
            "repeat_count": data.get("repeat_count", 0),
            "duration_sec": data.get("duration_sec", 0.0),
            "signature":    data.get("signature", {}),
            "llm_report":   data.get("llm_report"),
            "has_image":    has_image,
        })

    return events


def get_event_timeline(results_dir: Path, event_id: str) -> list[dict]:
    """
    Bir event'in tüm JSON dosyalarını kronolojik sırayla döndür.

    Returns:
        Timeline adımlarının listesi: new → update_01 → … → resolved
    """
    event_dir = results_dir / event_id
    if not event_dir.exists():
        return []

    steps: list[tuple] = []  # (sort_key, path, meta)

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

        # İlgili JPG dosyası var mı?
        jpg_name = path.stem + ".jpg"
        image_filename = jpg_name if (event_dir / jpg_name).exists() else None

        timeline.append({
            "event_id":      data.get("event_id", event_id),
            "event_status":  data.get("event_status", meta["status"]),
            "timestamp":     data.get("timestamp", ""),
            "repeat_count":  data.get("repeat_count", 0),
            "duration_sec":  data.get("duration_sec", 0.0),
            "change_reason": data.get("change_reason", ""),
            "signature":     data.get("signature", {}),
            "llm_report":    data.get("llm_report"),
            "image_filename": image_filename,
        })

    return timeline
