# -*- coding: utf-8 -*-
"""
app.py — Flask + Socket.IO backend

Endpoint'ler:
    GET  /api/events                             → filtreli event listesi (?date=&violation_type=&status=)
    GET  /api/events/<event_id>                  → timeline + notlar
    POST /api/events/<event_id>/note             → operatör notu ekle
    GET  /api/images/<event_id>/<fname>          → JPEG
    GET  /api/stats                              → dashboard istatistikleri
    GET  /api/reports?period=daily|weekly|monthly&date=YYYY-MM-DD
    GET  /api/config                             → ppe_pipeline config
    PUT  /api/config                             → ppe_pipeline config güncelle

Socket.IO:
    'new_alert' → results/ altına yeni .json yazıldığında yayınlanır
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify, request
from flask.helpers import send_file
from flask_cors import CORS
from flask_socketio import SocketIO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.watcher import ResultsWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Config (must be defined before DB setup so _load_config is available)
# ---------------------------------------------------------------------------

_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
_PPE_PIPELINE_DEFAULTS = {
    "helmet_conf": 0.20, "vest_conf": 0.30, "mask_conf": 0.25,
    "fire_conf": 0.50, "person_conf": 0.25, "temporal_window": 10,
    "use_helmet": True, "use_vest": True, "use_mask": True, "use_fire": True,
    "new_confirm_sec": 3.0,
}


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


# ---------------------------------------------------------------------------
# Reader secimi: database.enabled=true ise DB, aksi halde dosya sistemi
# ---------------------------------------------------------------------------

def _db_enabled() -> bool:
    try:
        cfg = _load_config()
        return bool(cfg.get("database", {}).get("enabled", False))
    except Exception:
        return False


_USE_DB = False

if _db_enabled():
    try:
        from backend.database.connection import init_pool, init_db
        import backend.database.reader as _db_reader
        init_pool()
        init_db()
        _USE_DB = True
        logger.info("DB reader aktif.")
    except Exception as _db_exc:
        logger.warning("DB baglantisi kurulamadi, dosya sistemine donuluyor: %s", _db_exc)

if not _USE_DB:
    import backend.event_reader as _file_reader


# Endpoint'lerin cagirdigi tekduzenlesmis yardimcilar
def _get_all_events():
    if _USE_DB:
        return _db_reader.get_all_events()
    return _file_reader.get_all_events(RESULTS_DIR)


def _get_filtered_events(date_str, violation_type, status):
    if _USE_DB:
        return _db_reader.get_filtered_events(date_str, violation_type, status)
    return _file_reader.get_filtered_events(RESULTS_DIR, date_str, violation_type, status)


def _get_event_timeline(event_id):
    if _USE_DB:
        return _db_reader.get_event_timeline(event_id)
    return _file_reader.get_event_timeline(RESULTS_DIR, event_id)


def _get_notes(event_id):
    if _USE_DB:
        return _db_reader.get_notes(event_id)
    return _file_reader.get_notes(RESULTS_DIR, event_id)


def _save_note(event_id, text):
    if _USE_DB:
        return _db_reader.save_note(event_id, text)
    return _file_reader.save_note(RESULTS_DIR, event_id, text)


def _get_stats():
    if _USE_DB:
        return _db_reader.get_stats()
    return _file_reader.get_stats(RESULTS_DIR)


def _get_report_data(period, date_str):
    if _USE_DB:
        return _db_reader.get_report_data(period, date_str)
    return _file_reader.get_report_data(RESULTS_DIR, period, date_str)


def _event_exists(event_id):
    if _USE_DB:
        # DB'de event var mi kontrol et
        from backend.database.connection import db_cursor
        with db_cursor() as cur:
            cur.execute("SELECT 1 FROM events WHERE event_id = %s", (event_id,))
            return cur.fetchone() is not None
    return (RESULTS_DIR / event_id).exists()

_config      = _load_config()
_backend_cfg = _config.get("backend", {})
HOST         = _backend_cfg.get("host", "0.0.0.0")
PORT         = int(_backend_cfg.get("port", 5050))
CORS_ORIGINS = _backend_cfg.get("cors_origins", ["http://localhost:5173"])
RESULTS_DIR  = (PROJECT_ROOT / _config.get("results_dir", "results")).resolve()

# ---------------------------------------------------------------------------
# Doğrulama
# ---------------------------------------------------------------------------

_EVENT_ID_RE = re.compile(r"^evt_\d+$")
_FILENAME_RE = re.compile(r"^evt_\d+_(new|update_\d+|resolved)\.jpg$")
_PERIOD_RE   = re.compile(r"^(daily|weekly|monthly)$")
_DATE_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VT_RE       = re.compile(r"^(helmet|vest|mask|fire)$")
_STATUS_RE   = re.compile(r"^(new|active|update)$")

# ---------------------------------------------------------------------------
# Flask + Socket.IO
# ---------------------------------------------------------------------------

app      = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

watcher: ResultsWatcher | None = None

# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

@app.route("/api/events")
def api_get_events():
    date_str       = request.args.get("date", "")
    violation_type = request.args.get("violation_type", "")
    status         = request.args.get("status", "")

    date_str       = date_str       if _DATE_RE.match(date_str)       else None
    violation_type = violation_type if _VT_RE.match(violation_type)   else None
    status         = status         if _STATUS_RE.match(status)        else None

    if date_str or violation_type or status:
        events = _get_filtered_events(date_str, violation_type, status)
    else:
        events = _get_all_events()

    return jsonify({"events": events, "total": len(events)})


@app.route("/api/events/<event_id>")
def api_get_timeline(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if not _event_exists(event_id):
        abort(404, f"Event bulunamadı: {event_id}")
    timeline = _get_event_timeline(event_id)
    notes    = _get_notes(event_id)
    return jsonify({"event_id": event_id, "timeline": timeline, "notes": notes})


@app.route("/api/events/<event_id>/note", methods=["POST"])
def api_add_note(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if not _event_exists(event_id):
        abort(404)
    body = request.get_json(silent=True) or {}
    text = str(body.get("note", "")).strip()
    if not text:
        abort(400, "Not boş olamaz.")
    entry = _save_note(event_id, text)
    return jsonify({"ok": True, "note": entry})


@app.route("/api/images/<event_id>/<filename>")
def api_get_image(event_id: str, filename: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400)
    if not _FILENAME_RE.match(filename):
        abort(400)
    img_path = (RESULTS_DIR / event_id / filename).resolve()
    if not str(img_path).startswith(str(RESULTS_DIR)):
        abort(403)
    if not img_path.exists():
        abort(404)
    return send_file(str(img_path), mimetype="image/jpeg")


@app.route("/api/stats")
def api_get_stats():
    return jsonify(_get_stats())


@app.route("/api/reports")
def api_get_reports():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "")
    if not _PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not _DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")
    data = _get_report_data(period, date_str or None)
    return jsonify({"period": period, "data": data})


@app.route("/api/events/<event_id>/llm", methods=["PATCH"])
def api_patch_llm(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    body   = request.get_json(silent=True) or {}
    report = str(body.get("llm_report", "")).strip()
    if not report:
        abort(400, "llm_report boş olamaz.")
    if _USE_DB:
        from backend.database.writer import update_llm_report
        update_llm_report(event_id, report)
    socketio.emit("llm_updated", {"event_id": event_id, "llm_report": report})
    return jsonify({"ok": True})


@app.route("/api/events/<event_id>/resolve", methods=["PATCH"])
def api_resolve_event(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if _USE_DB:
        from backend.database.writer import resolve_event
        resolve_event(event_id)
    socketio.emit("event_resolved", {"event_id": event_id})
    return jsonify({"ok": True})


@app.route("/api/config", methods=["GET"])
def api_get_config():
    cfg      = _load_config()
    pipeline = {**_PPE_PIPELINE_DEFAULTS, **cfg.get("ppe_pipeline", {})}
    return jsonify(pipeline)


@app.route("/api/config", methods=["PUT"])
def api_put_config():
    body    = request.get_json(silent=True) or {}
    allowed = set(_PPE_PIPELINE_DEFAULTS.keys())
    patch   = {k: v for k, v in body.items() if k in allowed}
    cfg     = _load_config()
    current = {**_PPE_PIPELINE_DEFAULTS, **cfg.get("ppe_pipeline", {})}
    current.update(patch)
    cfg["ppe_pipeline"] = current
    _save_config(cfg)
    return jsonify({"ok": True, "config": current})

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
        socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        if watcher:
            watcher.stop()


if __name__ == "__main__":
    main()
