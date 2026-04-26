# -*- coding: utf-8 -*-
"""
watcher.py
==========
results/ dizinini izler; yeni .json dosyası görüldüğünde
Socket.IO üzerinden 'new_alert' event'i yayınlar.

Watchdog WindowsApiObserver kullanır (Windows'ta varsayılan).
async_mode='threading' ile çalışır — eventlet/gevent gerekmez.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from backend.event_reader import parse_filename, read_json_file

logger = logging.getLogger("watcher")

# DB writer — yoksa veya disabled ise None kalir
_db_write_event = None

# Deduplication: Windows'ta on_created aynı dosya için birden fazla tetiklenebilir
_seen_lock  = threading.Lock()
_seen_paths: dict[str, float] = {}   # path_str → son işlenme zamanı
_DEDUP_TTL  = 2.0  # saniye


def _load_db_writer() -> None:
    global _db_write_event
    try:
        import yaml
        from pathlib import Path as _P
        cfg_path = _P(__file__).resolve().parents[1] / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if cfg.get("database", {}).get("enabled", False):
            from backend.database.writer import write_event
            _db_write_event = write_event
            logger.info("DB yazici aktif.")
    except Exception as exc:
        logger.warning("DB yazici yuklenemedi: %s", exc)


class _Handler(FileSystemEventHandler):
    def __init__(self, socketio) -> None:
        super().__init__()
        self._socketio = socketio

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix != ".json":
            return

        meta = parse_filename(path.name)
        if meta is None:
            return

        # Deduplication: Windows'ta aynı dosya için birden fazla on_created gelebilir
        path_str = str(path)
        now = time.time()
        with _seen_lock:
            if path_str in _seen_paths and now - _seen_paths[path_str] < _DEDUP_TTL:
                return
            _seen_paths[path_str] = now

        # Windows'ta NTFS yazma tam bitmeden event tetiklenebilir
        time.sleep(0.2)

        data = read_json_file(path)
        if not data:
            logger.warning(f"Boş/hatalı JSON okundu: {path}")
            return

        # DB'ye yaz (etkinse)
        if _db_write_event is not None:
            jpg_stem = path.stem
            img_name = jpg_stem + ".jpg"
            image_filename = img_name if (path.parent / img_name).exists() else None
            _db_write_event(data, image_filename=image_filename)

        payload = {
            "event_id":     data.get("event_id", ""),
            "event_status": data.get("event_status", ""),
            "timestamp":    data.get("timestamp", ""),
            "signature":    data.get("signature", {}),
            "llm_report":   data.get("llm_report"),
        }

        logger.info(f"[ALERT] {payload['event_id']} → {payload['event_status']}")
        self._socketio.emit("new_alert", payload)


class ResultsWatcher:
    """
    results/ dizinini izleyen watchdog sarmalayıcısı.
    Flask başladığında start(), kapanırken stop() çağrılmalı.
    """

    def __init__(self, results_dir: Path, socketio) -> None:
        self._results_dir = results_dir
        self._socketio    = socketio
        self._observer: Observer | None = None

    def start(self) -> None:
        _load_db_writer()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        handler = _Handler(self._socketio)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._results_dir), recursive=True)
        self._observer.start()
        logger.info(f"Watcher başladı: {self._results_dir}")

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Watcher durduruldu.")
