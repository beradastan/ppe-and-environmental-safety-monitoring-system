from __future__ import annotations

import yaml
from pathlib import Path

try:
    import torch as _torch
    _TORCH_CUDA = _torch.cuda.is_available()
except ImportError:
    _TORCH_CUDA = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_backend_url() -> str:
    try:
        with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
            _b = (yaml.safe_load(f) or {}).get("backend", {})
        return f"http://localhost:{_b.get('port', 5050)}"
    except Exception:
        return "http://localhost:5050"

def load_model_cfg() -> dict:
    try:
        with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("models", {})
    except Exception:
        return {}

def load_full_cfg() -> dict:
    try:
        with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

_MODEL_CFG = load_model_cfg()
_DEV_CFG   = _MODEL_CFG.get("device", "auto")
DEVICE     = ("cuda" if _TORCH_CUDA else "cpu") if _DEV_CFG == "auto" else _DEV_CFG

FIRE_MODEL_PATH   = _MODEL_CFG.get("fire_model",   "models/bera/fire_smoke_other_agent_final_best.pt")
PERSON_MODEL_PATH = _MODEL_CFG.get("person_model", "models/person_agent_scene_vinayakstyle_best.pt")

_CROP_CFG = _MODEL_CFG.get("crop", {})
CROP_HELMET_MODEL_PATH = _CROP_CFG.get("helmet_model", "models/bera/crophelmet_agent_final_best.pt")
CROP_VEST_MODEL_PATH   = _CROP_CFG.get("vest_model",   "models/bera/cropvest_agent_final_best.pt")
CROP_MASK_MODEL_PATH   = _CROP_CFG.get("mask_model",   "models/bera/cropmask_agent_final_best.pt")
CROP_MASK_IMGSZ        = int(_CROP_CFG.get("mask_imgsz", 640))

_SCENE_CFG = _MODEL_CFG.get("scene", {})
SCENE_HELMET_MODEL_PATH = _SCENE_CFG.get("helmet_model", "models/vinayak_trained_byBera/helmet_agent_final_best.pt")
SCENE_VEST_MODEL_PATH   = _SCENE_CFG.get("vest_model",   "models/vinayak_trained_byBera/vest_agent_final_best.pt")
SCENE_MASK_MODEL_PATH   = _SCENE_CFG.get("mask_model",   "models/mask_agent_scene_200ep_yolov8m_best.pt")

IMGSZ                = 640
TRACKER              = "config/bytetrack.yaml"
STATES_CLEANUP_EVERY = 300
PPE_INFER_EVERY      = 4
FIRE_INFER_EVERY     = 5
SCENE_PPE_INFER_EVERY = 2
MIN_TRACK_FRAMES     = 10
MIN_CROP_PX          = 40
MIN_HEAD_PX          = 30
MIN_TORSO_PX         = 50
INSIDE_FRAC_THR      = 0.20
PIPELINE_MAX_WIDTH   = 1920

PPE_MIN_SELF_SCORE:     dict[str, float] = {"helmet": 0.20, "vest": 0.20, "mask": 0.15}
PPE_MAX_NEIGHBOR_RATIO: dict[str, float] = {"helmet": 0.80, "vest": 0.75, "mask": 0.90}

HELMET_CLASSES = ["Hardhat",      "NO-Hardhat"]
VEST_CLASSES   = ["Safety Vest",  "NO-Safety Vest"]
MASK_CLASSES   = ["Mask",         "NO-Mask"]

COLOR_OK      = (0, 200,   0)
COLOR_WARN    = (0, 100, 255)
COLOR_DANGER  = (0,   0, 230)
COLOR_UNKNOWN = (0, 200, 255)
COLOR_FIRE    = (0,  60, 255)

def class_ids(model, names: list[str]) -> list[int]:
    name_to_id = {n: cid for cid, n in model.names.items()}
    missing = [n for n in names if n not in name_to_id]
    if missing:
        raise ValueError(f"Model'de eksik class: {missing} | Mevcut: {list(model.names.values())}")
    return [name_to_id[n] for n in names]
