from __future__ import annotations

import logging
import threading
from datetime import date as _date, timedelta
import calendar

from flask import Blueprint, Response, abort, jsonify, request

from backend.config_manager import (
    DATE_RE, PERIOD_RE, USE_DB,
    get_report_data, load_config,
)
from backend.extensions import socketio

logger = logging.getLogger("reports")
bp     = Blueprint("reports", __name__)

_LLM_LOCK = threading.Lock()

_PERIOD_FILE = {"daily": "Gunluk", "weekly": "Haftalik", "monthly": "Aylik"}

def _summary_date_range(period: str, date_str: str | None) -> tuple[str, str]:
    today = _date.today()
    if period == "daily":
        d = _date.fromisoformat(date_str) if date_str else today
        return d.isoformat(), d.isoformat()
    if period == "weekly":
        anchor = _date.fromisoformat(date_str) if date_str else today
        start  = anchor - timedelta(days=anchor.weekday())
        end    = min(start + timedelta(days=6), today)
        return start.isoformat(), end.isoformat()
    anchor   = _date.fromisoformat(date_str) if date_str else today
    start    = anchor.replace(day=1)
    last_day = calendar.monthrange(start.year, start.month)[1]
    end      = min(start.replace(day=last_day), today)
    return start.isoformat(), end.isoformat()

def _wide_fetch_start(period: str, start: str) -> str:
    s = _date.fromisoformat(start)
    if period == "daily":
        return (s - timedelta(days=1)).isoformat()
    if period == "weekly":
        return (s - timedelta(days=7)).isoformat()
    if s.month == 1:
        return _date(s.year - 1, 12, 1).isoformat()
    return _date(s.year, s.month - 1, 1).isoformat()

def _run_llm_and_save(period: str, date_str: str | None, auto_generated: bool = False) -> None:
    if not _LLM_LOCK.acquire(blocking=False):
        logger.warning("LLM raporu zaten üretiliyor, istek atlandı (%s).", period)
        socketio.emit("report_llm_error", {
            "period": period, "date": date_str or "", "error": "LLM meşgul, lütfen bekleyin."
        })
        return
    try:
        _run_llm_inner(period, date_str, auto_generated)
    finally:
        _LLM_LOCK.release()

def _run_llm_inner(period: str, date_str: str | None, auto_generated: bool) -> None:
    from backend.database.reader import get_events_for_summary
    from backend.database.writer import save_llm_report
    from backend.reports.services import EventAnalyticsService, ReportSummaryService
    from llm.safety_report_agent import SafetyReportAgent

    start, end  = _summary_date_range(period, date_str)
    fetch_start = _wide_fetch_start(period, start)
    events      = get_events_for_summary(fetch_start, end)
    analytics   = EventAnalyticsService(events)
    svc         = ReportSummaryService(analytics)

    if period == "daily":
        summary = svc.generate_daily_summary(start)
    elif period == "weekly":
        summary = svc.generate_weekly_summary(start, end)
    else:
        summary = svc.generate_monthly_summary(start, end)

    try:
        _llm_cfg = load_config().get("llm", {})
        text = SafetyReportAgent(
            ollama_base_url = _llm_cfg.get("base_url", "http://localhost:11434"),
            model_name      = _llm_cfg.get("model",    "qwen3:8b"),
            timeout         = int(_llm_cfg.get("timeout", 120)),
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
        socketio.emit("report_llm_error", {
            "period": period, "date": date_str or "", "error": str(exc)
        })

def start_report_scheduler() -> None:
    import time
    from datetime import datetime

    _ran: dict[str, str] = {}

    def _loop():
        while True:
            now   = datetime.now()
            today = now.strftime("%Y-%m-%d")
            if now.hour == 23 and now.minute == 55:
                if _ran.get("daily") != today:
                    _ran["daily"] = today
                    threading.Thread(target=_run_llm_and_save, args=("daily", None, True), daemon=True).start()
                if now.weekday() == 6 and _ran.get("weekly") != today:
                    _ran["weekly"] = today
                    threading.Thread(target=_run_llm_and_save, args=("weekly", None, True), daemon=True).start()
                tomorrow = now + timedelta(days=1)
                if tomorrow.month != now.month and _ran.get("monthly") != today:
                    _ran["monthly"] = today
                    threading.Thread(target=_run_llm_and_save, args=("monthly", None, True), daemon=True).start()
            time.sleep(30)

    threading.Thread(target=_loop, daemon=True, name="report-scheduler").start()
    logger.info("Rapor zamanlayıcısı başlatıldı.")

@bp.route("/api/reports")
def api_get_reports():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "")
    if not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")
    return jsonify({"period": period, "data": get_report_data(period, date_str or None)})

@bp.route("/api/reports/summary")
def api_get_report_summary():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")
    if not USE_DB:
        return jsonify({"error": "summary requires database mode"}), 503

    start, end = _summary_date_range(period, date_str)
    from backend.database.reader import get_events_for_summary
    from backend.reports.services import EventAnalyticsService, ReportSummaryService

    events    = get_events_for_summary(_wide_fetch_start(period, start), end)
    analytics = EventAnalyticsService(events)
    svc       = ReportSummaryService(analytics)

    if period == "daily":
        summary = svc.generate_daily_summary(start)
    elif period == "weekly":
        summary = svc.generate_weekly_summary(start, end)
    else:
        summary = svc.generate_monthly_summary(start, end)

    return jsonify(summary)

@bp.route("/api/reports/summary/llm", methods=["POST"])
def api_generate_report_llm():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if not USE_DB:
        return jsonify({"error": "requires database mode"}), 503
    threading.Thread(target=_run_llm_and_save, args=(period, date_str, False), daemon=True).start()
    return jsonify({"ok": True, "pending": True})

@bp.route("/api/reports/saved")
def api_get_saved_reports():
    if not USE_DB:
        return jsonify({"error": "requires database mode"}), 503
    period = request.args.get("period") or None
    if period and not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    from backend.database.reader import get_saved_reports
    return jsonify({"reports": get_saved_reports(period=period)})

@bp.route("/api/reports/saved/<int:report_id>")
def api_get_saved_report(report_id: int):
    if not USE_DB:
        return jsonify({"error": "requires database mode"}), 503
    from backend.database.reader import get_saved_report
    r = get_saved_report(report_id)
    if r is None:
        abort(404)
    return jsonify(r)

@bp.route("/api/reports/export/csv")
def api_export_csv():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")
    if not USE_DB:
        return jsonify({"error": "requires database mode"}), 503

    start, end = _summary_date_range(period, date_str)
    from backend.database.reader import get_events_for_summary
    from backend.reports.exporter import generate_csv

    events   = get_events_for_summary(start, end)
    filename = f"guvenlik_raporu_{_PERIOD_FILE.get(period, period)}_{start}.csv"
    return Response(
        generate_csv(events, period, start, end),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@bp.route("/api/reports/export/pdf")
def api_export_pdf():
    period   = request.args.get("period", "weekly")
    date_str = request.args.get("date", "") or None
    if not PERIOD_RE.match(period):
        abort(400, "period: daily|weekly|monthly")
    if date_str and not DATE_RE.match(date_str):
        abort(400, "date: YYYY-MM-DD")
    if not USE_DB:
        return jsonify({"error": "requires database mode"}), 503

    start, end = _summary_date_range(period, date_str)
    from backend.database.reader import get_events_for_summary
    from backend.reports.exporter import generate_pdf

    events   = get_events_for_summary(start, end)
    filename = f"guvenlik_raporu_{_PERIOD_FILE.get(period, period)}_{start}.pdf"
    try:
        pdf_bytes = generate_pdf(events, period, start, end)
    except ImportError:
        return jsonify({"error": "reportlab kurulu degil: pip install reportlab"}), 500

    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
