from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from flask import Blueprint, abort, jsonify, request

from backend.config_manager import PROJECT_ROOT
from backend.extensions import socketio

bp = Blueprint("pipeline", __name__)

_proc:      "subprocess.Popen | None" = None
_source:    str = ""
_camera_id: str = ""
_zone:      str = ""
_mode:      str = "crop"

STREAM_FRAME_PATH = os.path.join(tempfile.gettempdir(), "stream_frame.jpg")

def _detect_device() -> str:
    try:
        import torch as _torch
        return "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

@bp.route("/api/pipeline/status")
def api_pipeline_status():
    global _proc
    running = _proc is not None and _proc.poll() is None
    if not running:
        _proc = None
    return jsonify({
        "running":   running,
        "source":    _source    if running else "",
        "camera_id": _camera_id if running else "",
        "zone":      _zone      if running else "",
        "mode":      _mode      if running else "",
    })

@bp.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global _proc, _source, _camera_id, _zone, _mode
    if _proc is not None and _proc.poll() is None:
        return jsonify({"ok": False, "error": "Pipeline zaten calisiyor."})

    body = request.get_json(silent=True) or {}
    source    = str(body.get("source",    "")).strip()
    camera_id = str(body.get("camera_id", "")).strip()
    zone      = str(body.get("zone",      "")).strip()
    mode      = str(body.get("mode",      "crop")).strip()

    if mode not in ("crop", "scene"):
        mode = "crop"
    if not source:
        abort(400, "Kaynak belirtilmedi.")
    if not source.isdigit():
        abort(400, "Geçersiz kamera indeksi.")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "run_live_video.py"),
        "--device", _detect_device(),
        "--mode",   mode,
        "--camera", source,
    ]
    if camera_id:
        cmd += ["--camera-id", camera_id]
    if zone:
        cmd += ["--zone", zone]

    _proc      = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    _source    = source
    _camera_id = camera_id
    _zone      = zone
    _mode      = mode
    return jsonify({"ok": True})

@bp.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global _proc, _source, _camera_id, _zone, _mode
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc      = None
    _source    = ""
    _camera_id = ""
    _zone      = ""
    _mode      = "crop"
    return jsonify({"ok": True})

@bp.route("/api/pipeline/camera-status", methods=["POST"])
def api_camera_status():
    body = request.get_json(silent=True) or {}
    socketio.emit("camera_status", {
        "status":    str(body.get("status",    "online")),
        "camera_id": str(body.get("camera_id", "")),
        "zone":      str(body.get("zone",      "")),
    })
    return jsonify({"ok": True})

@bp.route("/api/stream/frame")
def api_stream_frame():
    try:
        with open(STREAM_FRAME_PATH, "rb") as f:
            data = f.read()
        from flask import Response
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store, no-cache"})
    except Exception:
        abort(404)
