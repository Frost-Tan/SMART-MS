"""
Stage1 Binary Classification Task: Default Paths and Hyperparameter Configuration.
Data defaults to the relative path from the repository root: dataset/FINAL_task1_neg.csv.
Reports, images, and models are written by default to the output subdirectory under this module (auto-created).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Relative path to the data file from the repository root (for documentation and parsing)
DATA_RELPATH = "dataset/FINAL_task1_neg.csv"


def project_root() -> Path:
    """SMART-MS/Stage1/config.py -> Three levels up is the repository root."""
    return Path(__file__).resolve().parent.parent.parent


def stage1_dir() -> Path:
    """Directory where this Stage1 code resides."""
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    """Default output directory: output/ under the code directory."""
    return stage1_dir() / "output"


def default_config() -> Dict[str, Any]:
    root = project_root()
    out = default_output_dir()
    return {
        "DATA_PATH": str((root / DATA_RELPATH).resolve()),
        "OUTPUT_DIR": str(out),
        "MODEL_DIR": str(out),
        "BIN_SIZE_SPEC": 0.1,
        "MAX_MZ": 2000,
        "BIN_SIZE_LOSS": 0.1,
        "MAX_LOSS": 1500,
        "INT_THRESHOLD": 0.01,
        "CV_FOLDS": 5,
        "RANDOM_SEED": 42,
        "ADDUCT_MASS_DIFF": {
            "[M-H]-": -1.0073,
            "[M+FA-H]-": 44.9982,
            "[M+HCOO]-": 44.9982,
            "[M+Cl]-": 34.9689,
            "[M+CH3COO]-": 59.0138,
        },
    }
