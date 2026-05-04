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
import os as _os
import re
import subprocess as _subprocess
import sys
import tempfile as _tempfile
from pathlib import Path

import yaml
from flask import Flask, Response, abort, jsonify, request
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
    "fire_min_area_ratio": 0.01,
    "fire_growth_window":  10,
    "fire_growth_factor":  1.5,
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

_EVENT_ID_RE = re.compile(r"^evt_[a-zA-Z0-9_]+$")
_FILENAME_RE = re.compile(r"^evt_[a-zA-Z0-9_]+(new|update_\d+|closed)\.jpg$")
_PERIOD_RE   = re.compile(r"^(daily|weekly|monthly)$")
_DATE_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VT_RE       = re.compile(r"^(helmet|vest|mask|fire)$")
_STATUS_RE   = re.compile(r"^(new|active|update|closed)$")

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


def _wide_fetch_start(period: str, start: str) -> str:
    """Karşılaştırma için önceki dönemi de kapsayan başlangıç tarihi."""
    from datetime import date as _date, timedelta
    import calendar
    s = _date.fromisoformat(start)
    if period == "daily":
        return (s - timedelta(days=1)).isoformat()
    if period == "weekly":
        return (s - timedelta(days=7)).isoformat()
    # monthly: bir önceki ayın 1'i
    if s.month == 1:
        return _date(s.year - 1, 12, 1).isoformat()
    return _date(s.year, s.month - 1, 1).isoformat()


def _summary_date_range(period: str, date_str: str | None):
    from datetime import date as _date, timedelta
    import calendar
    today = _date.today()
    if period == "daily":
        d = _date.fromisoformat(date_str) if date_str else today
        return d.isoformat(), d.isoformat()
    if period == "weekly":
        anchor = _date.fromisoformat(date_str) if date_str else today
        start  = anchor - timedelta(days=anchor.weekday())   # Pazartesi
        end    = min(start + timedelta(days=6), today)       # Pazar veya bugün
        return start.isoformat(), end.isoformat()
    # monthly
    anchor    = _date.fromisoformat(date_str) if date_str else today
    start     = anchor.replace(day=1)
    last_day  = calendar.monthrange(start.year, start.month)[1]
    end       = min(start.replace(day=last_day), today)
    return start.isoformat(), end.isoformat()


@app.route("/api/reports/summary")
def api_get_report_summary():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not _PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not _DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")

    if not _USE_DB:
        return jsonify({"error": "summary requires database mode"}), 503

    start, end = _summary_date_range(period, date_str)

    from backend.database.reader import get_events_for_summary
    from backend.reports.services import EventAnalyticsService, ReportSummaryService

    fetch_start = _wide_fetch_start(period, start)
    events    = get_events_for_summary(fetch_start, end)
    analytics = EventAnalyticsService(events)
    svc       = ReportSummaryService(analytics)

    if period == "daily":
        summary = svc.generate_daily_summary(start)
    elif period == "weekly":
        summary = svc.generate_weekly_summary(start, end)
    else:
        summary = svc.generate_monthly_summary(start, end)

    return jsonify(summary)


def _run_llm_and_save(period: str, date_str: str | None, auto_generated: bool = False) -> None:
    """LLM raporu üret, DB'ye kaydet, socket ile bildir."""
    from backend.database.reader import get_events_for_summary
    from backend.database.writer import save_llm_report
    from backend.reports.services import EventAnalyticsService, ReportSummaryService
    from llm.safety_report_agent import SafetyReportAgent

    start, end  = _summary_date_range(period, date_str)
    fetch_start = _wide_fetch_start(period, start)
    events      = get_events_for_summary(fetch_start, end)
    analytics   = EventAnalyticsService(events)
    svc       = ReportSummaryService(analytics)

    if period == "daily":
        summary = svc.generate_daily_summary(start)
    elif period == "weekly":
        summary = svc.generate_weekly_summary(start, end)
    else:
        summary = svc.generate_monthly_summary(start, end)

    try:
        _llm_cfg = _load_config().get("llm", {})
        text = SafetyReportAgent(
            ollama_base_url=_llm_cfg.get("base_url", "http://localhost:11434"),
            model_name=_llm_cfg.get("model", "qwen3:8b"),
            timeout=int(_llm_cfg.get("timeout", 120)),
        ).generate_report(summary)
        save_llm_report(period, start, text, auto_generated=auto_generated)
        socketio.emit("report_llm_ready", {
            "period": period, "date": date_str or "",
            "llm_text": text, "auto_generated": auto_generated,
        })
        if auto_generated:
            logger.info("Otomatik %s raporu oluşturuldu (%s).", period, start)
    except Exception as exc:
        logger.error("LLM rapor hatasi (%s): %s", period, exc)
        socketio.emit("report_llm_error", {"period": period, "date": date_str or "", "error": str(exc)})


def _start_report_scheduler() -> None:
    """Gece 23:55'te günlük, Pazar günlük+haftalık, ay sonu aylık rapor üretir."""
    import threading as _th
    import time as _ti
    from datetime import datetime as _dt, timedelta as _td

    _ran: dict[str, str] = {}

    def _loop():
        while True:
            now   = _dt.now()
            today = now.strftime("%Y-%m-%d")
            if now.hour == 23 and now.minute == 55:
                if _ran.get("daily") != today:
                    _ran["daily"] = today
                    _th.Thread(target=_run_llm_and_save, args=("daily", None, True), daemon=True).start()
                if now.weekday() == 6 and _ran.get("weekly") != today:
                    _ran["weekly"] = today
                    _th.Thread(target=_run_llm_and_save, args=("weekly", None, True), daemon=True).start()
                tomorrow = now + _td(days=1)
                if tomorrow.month != now.month and _ran.get("monthly") != today:
                    _ran["monthly"] = today
                    _th.Thread(target=_run_llm_and_save, args=("monthly", None, True), daemon=True).start()
            _ti.sleep(30)

    _th.Thread(target=_loop, daemon=True, name="report-scheduler").start()
    logger.info("Rapor zamanlayıcısı başlatıldı.")


@app.route("/api/reports/summary/llm", methods=["POST"])
def api_generate_report_llm():
    import threading as _threading
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not _PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if not _USE_DB:
        return jsonify({"error": "requires database mode"}), 503

    _threading.Thread(
        target=_run_llm_and_save, args=(period, date_str, False), daemon=True
    ).start()
    return jsonify({"ok": True, "pending": True})


@app.route("/api/reports/saved")
def api_get_saved_reports():
    if not _USE_DB:
        return jsonify({"error": "requires database mode"}), 503
    period = request.args.get("period") or None
    if period and not _PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    from backend.database.reader import get_saved_reports
    return jsonify({"reports": get_saved_reports(period=period)})


@app.route("/api/reports/saved/<int:report_id>")
def api_get_saved_report(report_id: int):
    if not _USE_DB:
        return jsonify({"error": "requires database mode"}), 503
    from backend.database.reader import get_saved_report
    r = get_saved_report(report_id)
    if r is None:
        abort(404)
    return jsonify(r)


@app.route("/api/events/<event_id>/close", methods=["PATCH"])
def api_close_event(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if _USE_DB:
        from backend.database.writer import close_event
        body = request.get_json(silent=True) or {}
        rc = body.get("repeat_count")
        ds = body.get("duration_sec")
        close_event(
            event_id,
            repeat_count=int(rc) if rc is not None else None,
            duration_sec=float(ds) if ds is not None else None,
        )
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})


@app.route("/api/events/<event_id>/resolve", methods=["PATCH"])
def api_resolve_event(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if _USE_DB:
        from backend.database.writer import resolve_event
        resolve_event(event_id)
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})


@app.route("/api/events/<event_id>/false-positive", methods=["PATCH"])
def api_false_positive(event_id: str):
    if not _EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if _USE_DB:
        from backend.database.writer import mark_false_positive
        mark_false_positive(event_id)
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})



_pipeline_proc:      "_subprocess.Popen | None" = None
_pipeline_source:    str = ""
_pipeline_camera_id: str = ""
_pipeline_zone:      str = ""
_pipeline_mode:      str = "crop"
_STREAM_FRAME_PATH = _os.path.join(_tempfile.gettempdir(), "factory_safety_stream.jpg")
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


@app.route("/api/pipeline/status")
def api_pipeline_status():
    global _pipeline_proc
    running = _pipeline_proc is not None and _pipeline_proc.poll() is None
    if not running:
        _pipeline_proc = None
    return jsonify({
        "running":   running,
        "source":    _pipeline_source    if running else "",
        "camera_id": _pipeline_camera_id if running else "",
        "zone":      _pipeline_zone      if running else "",
        "mode":      _pipeline_mode      if running else "",
    })


@app.route("/api/pipeline/browse")
def api_pipeline_browse():
    try:
        ps = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        result = _subprocess.run(
            [ps, '-STA', '-Command',
             '[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null; '
             '$d = New-Object System.Windows.Forms.OpenFileDialog; '
             '$d.Filter = "Video (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All (*.*)|*.*"; '
             '$d.Title = "Video sec"; '
             '$d.ShowDialog() | Out-Null; '
             '$d.FileName'],
            capture_output=True, text=True, timeout=60,
        )
        path = result.stdout.strip()
    except Exception as exc:
        logger.warning("Browse hatasi: %s", exc)
        path = ""
    return jsonify({"path": path})


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global _pipeline_proc, _pipeline_source, _pipeline_camera_id, _pipeline_zone, _pipeline_mode
    if _pipeline_proc is not None and _pipeline_proc.poll() is None:
        return jsonify({"ok": False, "error": "Pipeline zaten calisiyor."})
    body      = request.get_json(silent=True) or {}
    source    = str(body.get("source", "")).strip()
    camera_id = str(body.get("camera_id", "")).strip()
    zone      = str(body.get("zone", "")).strip()
    mode      = str(body.get("mode", "crop")).strip()
    if mode not in ("crop", "scene"):
        mode = "crop"
    if not source:
        abort(400, "Kaynak belirtilmedi.")
    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        _device = "cpu"
    cmd = [sys.executable, str(PROJECT_ROOT / "run_live_video.py"),
           "--display", "--device", _device, "--mode", mode]
    if source.isdigit():
        cmd += ["--camera", source]
    else:
        src_path = Path(source)
        if not src_path.is_absolute():
            src_path = PROJECT_ROOT / source
        if not src_path.exists():
            abort(400, "Dosya bulunamadi.")
        if src_path.suffix.lower() not in _VIDEO_EXTS:
            abort(400, "Desteklenmeyen video formati.")
        cmd += ["--video", str(src_path)]
    if camera_id:
        cmd += ["--camera-id", camera_id]
    if zone:
        cmd += ["--zone", zone]
    _pipeline_proc      = _subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    _pipeline_source    = source
    _pipeline_camera_id = camera_id
    _pipeline_zone      = zone
    _pipeline_mode      = mode
    return jsonify({"ok": True})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global _pipeline_proc, _pipeline_source, _pipeline_camera_id, _pipeline_zone, _pipeline_mode
    if _pipeline_proc is None or _pipeline_proc.poll() is not None:
        _pipeline_proc = None
    else:
        _pipeline_proc.terminate()
        _pipeline_proc = None
    _pipeline_source    = ""
    _pipeline_camera_id = ""
    _pipeline_zone      = ""
    _pipeline_mode      = "crop"
    return jsonify({"ok": True})


@app.route("/api/pipeline/camera-status", methods=["POST"])
def api_camera_status():
    body      = request.get_json(silent=True) or {}
    status    = str(body.get("status",    "online"))
    camera_id = str(body.get("camera_id", ""))
    zone      = str(body.get("zone",      ""))
    socketio.emit("camera_status", {"status": status, "camera_id": camera_id, "zone": zone})
    return jsonify({"ok": True})


@app.route("/api/stream/frame")
def api_stream_frame():
    try:
        with open(_STREAM_FRAME_PATH, 'rb') as f:
            data = f.read()
        return Response(data, mimetype='image/jpeg',
                        headers={'Cache-Control': 'no-store, no-cache'})
    except Exception:
        abort(404)


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
    if _USE_DB:
        _start_report_scheduler()
    logger.info(f"Backend başlıyor → http://{HOST}:{PORT}")
    logger.info(f"Results dizini: {RESULTS_DIR}")
    try:
        socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        if watcher:
            watcher.stop()


if __name__ == "__main__":
    main()
