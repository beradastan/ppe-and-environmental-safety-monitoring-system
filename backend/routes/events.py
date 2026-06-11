from __future__ import annotations

from flask import Blueprint, abort, jsonify, request
from flask.helpers import send_file

from backend.config_manager import (
    DATE_RE, EVENT_ID_RE, FILENAME_RE, RESULTS_DIR, STATUS_RE, USE_DB, VT_RE,
    event_exists, get_all_events, get_event_timeline, get_filtered_events,
    get_notes, get_stats, save_note,
)
from backend.extensions import socketio

bp = Blueprint("events", __name__)

@bp.route("/api/events")
def api_get_events():
    date_str       = request.args.get("date", "")
    violation_type = request.args.get("violation_type", "")
    status         = request.args.get("status", "")

    date_str       = date_str       if DATE_RE.match(date_str)       else None
    violation_type = violation_type if VT_RE.match(violation_type)   else None
    status         = status         if STATUS_RE.match(status)        else None

    if date_str or violation_type or status:
        events = get_filtered_events(date_str, violation_type, status)
    else:
        events = get_all_events()

    return jsonify({"events": events, "total": len(events)})

@bp.route("/api/events/<event_id>")
def api_get_timeline(event_id: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if not event_exists(event_id):
        abort(404, f"Event bulunamadı: {event_id}")
    return jsonify({
        "event_id": event_id,
        "timeline": get_event_timeline(event_id),
        "notes":    get_notes(event_id),
    })

@bp.route("/api/events/<event_id>/note", methods=["POST"])
def api_add_note(event_id: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if not event_exists(event_id):
        abort(404)
    body = request.get_json(silent=True) or {}
    text = str(body.get("note", "")).strip()
    if not text:
        abort(400, "Not boş olamaz.")
    return jsonify({"ok": True, "note": save_note(event_id, text)})

@bp.route("/api/images/<event_id>/<filename>")
def api_get_image(event_id: str, filename: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400)
    if not FILENAME_RE.match(filename):
        abort(400)
    img_path = (RESULTS_DIR / event_id / filename).resolve()
    if not str(img_path).startswith(str(RESULTS_DIR)):
        abort(403)
    if not img_path.exists():
        abort(404)
    return send_file(str(img_path), mimetype="image/jpeg")

@bp.route("/api/stats")
def api_get_stats():
    return jsonify(get_stats())

@bp.route("/api/events/<event_id>/close", methods=["PATCH"])
def api_close_event(event_id: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if USE_DB:
        from backend.database.writer import close_event
        body = request.get_json(silent=True) or {}
        rc   = body.get("repeat_count")
        ds   = body.get("duration_sec")
        close_event(
            event_id,
            repeat_count = int(rc)   if rc is not None else None,
            duration_sec = float(ds) if ds is not None else None,
        )
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})

@bp.route("/api/events/<event_id>/resolve", methods=["PATCH"])
def api_resolve_event(event_id: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if USE_DB:
        from backend.database.writer import resolve_event
        resolve_event(event_id)
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})

@bp.route("/api/events/<event_id>/false-positive", methods=["PATCH"])
def api_false_positive(event_id: str):
    if not EVENT_ID_RE.match(event_id):
        abort(400, "Geçersiz event_id.")
    if USE_DB:
        from backend.database.writer import mark_false_positive
        mark_false_positive(event_id)
    socketio.emit("event_closed", {"event_id": event_id})
    return jsonify({"ok": True})
