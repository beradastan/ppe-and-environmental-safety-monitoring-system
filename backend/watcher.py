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
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from backend.event_reader import parse_filename, read_json_file

logger = logging.getLogger("watcher")


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

        # Windows'ta NTFS yazma tam bitmeden event tetiklenebilir
        time.sleep(0.2)

        data = read_json_file(path)
        if not data:
            logger.warning(f"Boş/hatalı JSON okundu: {path}")
            return

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
