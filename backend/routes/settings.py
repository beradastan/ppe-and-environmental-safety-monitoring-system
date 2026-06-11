from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.config_manager import _PPE_PIPELINE_DEFAULTS, load_config, save_config

bp = Blueprint("settings", __name__)

@bp.route("/api/config", methods=["GET"])
def api_get_config():
    cfg      = load_config()
    pipeline = {**_PPE_PIPELINE_DEFAULTS, **cfg.get("ppe_pipeline", {})}
    return jsonify(pipeline)

@bp.route("/api/config", methods=["PUT"])
def api_put_config():
    body    = request.get_json(silent=True) or {}
    allowed = set(_PPE_PIPELINE_DEFAULTS.keys())
    patch   = {k: v for k, v in body.items() if k in allowed}
    cfg     = load_config()
    current = {**_PPE_PIPELINE_DEFAULTS, **cfg.get("ppe_pipeline", {})}
    current.update(patch)
    cfg["ppe_pipeline"] = current
    save_config(cfg)
    return jsonify({"ok": True, "config": current})
