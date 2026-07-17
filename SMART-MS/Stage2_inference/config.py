"""
Stage2 Inference: Default paths resolved relative to the repository root, aligned with the Stage2 training Strategy 3 output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Relative to repo root: default uses the full pipeline (Strategy 3) trained model directory
DEFAULT_MODEL_DIR_REL = "SMART-MS/Stage2/output/Strategy3_Full_Pipeline"
DEFAULT_DB_REL = "dataset/merged_saponin_named.csv"
DEFAULT_MGF_REL = "dataset/input.mgf"


def project_root() -> Path:
    """SMART-MS/Stage2_inference/config.py -> Three levels up is the repository root."""
    return Path(__file__).resolve().parent.parent.parent


def stage2_inference_dir() -> Path:
    """Directory where this inference module resides."""
    return Path(__file__).resolve().parent


def default_inference_config() -> Dict[str, Any]:
    root = project_root()
    out = stage2_inference_dir() / "output"
    site_cols: List[str] = ["C3", "C6", "C20", "C26", "C28", "Other_sugar"]
    return {
        "MODEL_DIR": str((root / DEFAULT_MODEL_DIR_REL.replace("\\", "/")).resolve()),
        "MGF_FILE": str((root / DEFAULT_MGF_REL.replace("\\", "/")).resolve()),
        "DB_FILE": str((root / DEFAULT_DB_REL.replace("\\", "/")).resolve()),
        "OUTPUT_DIR": str(out.resolve()),
        "OUTPUT_CSV_ALL": "Task2_inference_results_detailed_final.csv",
        "OUTPUT_CSV_MATCHED": "Task2_inference_results_matched_only.csv",
        # Physical constants (consistent with the original inference script)
        "H_PLUS_MASS": 1.007276466,
        "FA_ION_MASS": 44.998201,
        "FA_NEUTRAL_MASS": 46.005479,
        "MASS_TOLERANCE_PPM": 10.0,
        "PROB_RATIO_THRESHOLD": 0.4,
        "TOP_K_RESULTS": 3,
        "MIN_PMASS": 300.0,
        "MIN_PEAKS": 5,
        "BIN_SIZE": 0.2,
        "LOSS_BIN_SIZE": 0.2,
        "MAX_MZ": 2000,
        "MAX_LOSS": 600,
        "SITE_COLS": site_cols,
        "ADDUCT_MODES": ["[M-H]-", "[M+HCOO]-"],
        "OUTER_FOLDS": 10,
    }
