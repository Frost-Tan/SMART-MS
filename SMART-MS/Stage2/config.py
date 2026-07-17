"""
Stage2 Cascaded Ablation: Default Paths and Hyperparameters.
Data defaults to the relative path from the repository root: dataset/FINAL_task2_neg.csv.
Results are written by default to the output directory under this module (auto-created).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

DATA_RELPATH = "dataset/FINAL_task2_neg.csv"

STRATEGIES: Dict[int, str] = {
    1: "Strategy1_Independent",
    2: "Strategy2_Cascaded_NoConstraint",
    3: "Strategy3_Full_Pipeline",
}


def project_root() -> Path:
    """SMART-MS/Stage2/config.py -> Three levels up is the repository root."""
    return Path(__file__).resolve().parent.parent.parent


def stage2_dir() -> Path:
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    return stage2_dir() / "output"


def default_config() -> Dict[str, Any]:
    root = project_root()
    out = default_output_dir()
    return {
        "DATA_PATH": str((root / DATA_RELPATH).resolve()),
        "OUTPUT_DIR": str(out.resolve()),
        "RUN_ALL_ABLATIONS": False,
        "MIN_PEAKS": 5,
        "BIN_SIZE": 0.2,
        "LOSS_BIN_SIZE": 0.2,
        "MAX_MZ": 2000,
        "MAX_LOSS": 600,
        "RANDOM_SEED": 42,
        "OUTER_FOLDS": 10,
        "INNER_FOLDS": 10,
        "LGBM_DEVICE": "cpu",
        "LGBM_N_ESTIMATORS": 1500,
        "LGBM_LEARNING_RATE": 0.005,
        "LGBM_NUM_LEAVES": 31,
        "LGBM_FEATURE_FRACTION": 0.6,
        "LGBM_BAGGING_FRACTION": 0.7,
        "LGBM_BAGGING_FREQ": 1,
        "LGBM_MIN_CHILD_SAMPLES": 3,
    }
