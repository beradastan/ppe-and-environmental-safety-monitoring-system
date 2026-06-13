from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("config_manager")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_PPE_PIPELINE_DEFAULTS: dict = {
    "crop_helmet_conf": 0.15, "crop_vest_conf": 0.35, "crop_mask_conf": 0.10,
    "scene_helmet_conf": 0.25, "scene_vest_conf": 0.30, "scene_mask_conf": 0.05,
    "fire_conf": 0.75, "person_conf": 0.25,
    "temporal_window": 20, "scene_temporal_window": 30,
    "use_helmet": True, "use_vest": True, "use_mask": True, "use_fire": True,
    "fire_min_area_ratio": 0.01,
    "fire_growth_window":  10,
    "fire_growth_factor":  1.5,
}

EVENT_ID_RE  = re.compile(r"^evt_[a-zA-Z0-9_]+$")
FILENAME_RE  = re.compile(r"^evt_[a-zA-Z0-9_]+(new|update_\d+|closed)\.jpg$")
PERIOD_RE    = re.compile(r"^(daily|weekly|monthly)$")
DATE_RE      = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VT_RE        = re.compile(r"^(helmet|vest|mask|fire)$")
STATUS_RE    = re.compile(r"^(new|active|update|closed)$")

def load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

def _db_enabled() -> bool:
    try:
        return bool(load_config().get("database", {}).get("enabled", False))
    except Exception:
        return False

_cfg         = load_config()
_backend_cfg = _cfg.get("backend", {})

HOST         = _backend_cfg.get("host", "0.0.0.0")
PORT         = int(_backend_cfg.get("port", 5050))
CORS_ORIGINS = _backend_cfg.get("cors_origins", ["http://localhost:5173"])
RESULTS_DIR  = (PROJECT_ROOT / _cfg.get("results_dir", "results")).resolve()

USE_DB      = False
_db_reader  = None
_file_reader = None

if _db_enabled():
    try:
        from backend.database.connection import init_pool, init_db
        import backend.database.reader as _dbr
        init_pool()
        init_db()
        _db_reader = _dbr
        USE_DB     = True
        logger.info("Database reader active.")
    except Exception as exc:
        logger.warning("Database connection failed, falling back to file system: %s", exc)

if not USE_DB:
    import backend.event_reader as _fr
    _file_reader = _fr

def get_all_events():
    if USE_DB:
        return _db_reader.get_all_events()
    return _file_reader.get_all_events(RESULTS_DIR)

def get_filtered_events(date_str, violation_type, status):
    if USE_DB:
        return _db_reader.get_filtered_events(date_str, violation_type, status)
    return _file_reader.get_filtered_events(RESULTS_DIR, date_str, violation_type, status)

def get_event_timeline(event_id: str):
    if USE_DB:
        return _db_reader.get_event_timeline(event_id)
    return _file_reader.get_event_timeline(RESULTS_DIR, event_id)

def get_notes(event_id: str):
    if USE_DB:
        return _db_reader.get_notes(event_id)
    return _file_reader.get_notes(RESULTS_DIR, event_id)

def save_note(event_id: str, text: str):
    if USE_DB:
        return _db_reader.save_note(event_id, text)
    return _file_reader.save_note(RESULTS_DIR, event_id, text)

def get_stats():
    if USE_DB:
        return _db_reader.get_stats()
    return _file_reader.get_stats(RESULTS_DIR)

def get_report_data(period: str, date_str: str | None):
    if USE_DB:
        return _db_reader.get_report_data(period, date_str)
    return _file_reader.get_report_data(RESULTS_DIR, period, date_str)

def event_exists(event_id: str) -> bool:
    if USE_DB:
        from backend.database.connection import db_cursor
        with db_cursor() as cur:
            cur.execute("SELECT 1 FROM events WHERE event_id = %s", (event_id,))
            return cur.fetchone() is not None
    return (RESULTS_DIR / event_id).exists()
