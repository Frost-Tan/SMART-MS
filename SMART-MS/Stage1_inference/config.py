"""
Stage1 Inference: Default paths resolved relative to the repository root, aligned with the Stage1 training output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Relative to repo root: default directory where training writes .joblib files (consistent with Stage1/config)
DEFAULT_MODELS_DIR_REL = "SMART-MS/Stage1/output"
# Relative to repo root: can be specified via --mgf on the command line; placeholder default here
DEFAULT_INPUT_MGF_REL = "dataset/input.mgf"


def project_root() -> Path:
    """SMART-MS/Stage1_inference/config.py -> Three levels up is the repository root."""
    return Path(__file__).resolve().parent.parent.parent


def stage1_inference_dir() -> Path:
    """Directory where this inference module resides."""
    return Path(__file__).resolve().parent


def default_inference_config() -> Dict[str, Any]:
    root = project_root()
    out = stage1_inference_dir() / "output"
    return {
        "MODELS_DIR": str((root / DEFAULT_MODELS_DIR_REL.replace("\\", "/")).resolve()),
        "INPUT_MGF": str((root / DEFAULT_INPUT_MGF_REL.replace("\\", "/")).resolve()),
        "OUTPUT_DIR": str(out.resolve()),
        "OUT_H_ONLY": "Task1_inference_H_hypothesis_only.csv",
        "OUT_FA_ONLY": "Task1_inference_FA_hypothesis_only.csv",
        "OUT_BEST_MATCH": "Task1_inference_Best_Match_Final.csv",
        "BIN_SIZE_SPEC": 0.1,
        "MAX_MZ": 2000,
        "BIN_SIZE_LOSS": 0.1,
        "MAX_LOSS": 1500,
        "INT_THRESHOLD": 0.01,
    }
