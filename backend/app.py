# -*- coding: utf-8 -*-
"""
app.py
======
Flask + Socket.IO backend.

Başlatma:
    python backend/app.py
    # veya proje kökünden:
    python -m backend.app

Endpoint'ler:
    GET  /api/events                      → tüm event listesi
    GET  /api/events/<event_id>           → bir event'in tam timeline'ı
    GET  /api/images/<event_id>/<fname>   → ekran görüntüsü (JPEG)

Socket.IO:
    'new_alert' → results/ altına yeni .json yazıldığında yayınlanır
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify
from flask.helpers import send_file
from flask_cors import CORS
from flask_socketio import SocketIO

# Proje kökünü Python path'e ekle (python backend/app.py ile çalıştırıldığında)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.event_reader import get_all_events, get_event_timeline
from backend.watcher import ResultsWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_config      = _load_config()
_backend_cfg = _config.get("backend", {})
HOST         = _backend_cfg.get("host", "0.0.0.0")
PORT         = int(_backend_cfg.get("port", 5050))
CORS_ORIGINS = _backend_cfg.get("cors_origins", ["http://localhost:5173"])
RESULTS_DIR  = (PROJECT_ROOT / _config.get("results_dir", "results")).resolve()

# ---------------------------------------------------------------------------
# Doğrulama sabitleri
# ---------------------------------------------------------------------------

_EVENT_ID_RE = re.compile(r"^evt_\d+$")
_FILENAME_RE = re.compile(r"^evt_\d+_(new|update_\d+|resolved)\.jpg$")

# ---------------------------------------------------------------------------
# Flask + Socket.IO
# ---------------------------------------------------------------------------

app     = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

watcher: ResultsWatcher | None = None


@app.before_request
def _noop():  # ensures first-request context is set up
    pass


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/events")
def api_get_events():
    events = get_all_events(RESULTS_DIR)
    return jsonify({"events": events, "total": len(events)})


@app.route("/api/events/<event_id>")
def api_get_timeline(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id formatı.")

    event_dir = RESULTS_DIR / event_id
    if not event_dir.exists():
        abort(404, f"Event bulunamadı: {event_id}")

    timeline = get_event_timeline(RESULTS_DIR, event_id)
    return jsonify({"event_id": event_id, "timeline": timeline})


@app.route("/api/images/<event_id>/<filename>")
def api_get_image(event_id: str, filename: str):
    # Girdi doğrulama (path traversal koruması)
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if not _FILENAME_RE.match(filename):
        abort(400, "Geçersiz dosya adı.")

    img_path = (RESULTS_DIR / event_id / filename).resolve()

    # Çözümlenen yolun hâlâ results/ altında olduğunu doğrula
    if not str(img_path).startswith(str(RESULTS_DIR)):
        abort(403)

    if not img_path.exists():
        abort(404)

    return send_file(str(img_path), mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# Socket.IO
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    logger.info("İstemci bağlandı.")


@socketio.on("disconnect")
def on_disconnect():
    logger.info("İstemci ayrıldı.")


# ---------------------------------------------------------------------------
# Başlat
# ---------------------------------------------------------------------------

def main() -> None:
    global watcher
    watcher = ResultsWatcher(RESULTS_DIR, socketio)
    watcher.start()

    logger.info(f"Backend başlıyor → http://{HOST}:{PORT}")
    logger.info(f"Results dizini: {RESULTS_DIR}")

    try:
        socketio.run(
            app,
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,  # reloader çift process açar → watchdog çakışır
        )
    finally:
        if watcher:
            watcher.stop()


if __name__ == "__main__":
    main()
