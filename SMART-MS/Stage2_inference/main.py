"""
Stage2 Cascaded Inference Entry Point. It is recommended to run from the repository root (consistent with Stage2 training).
All relative paths are resolved relative to the repository root.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

_STAGEDIR = os.path.dirname(os.path.abspath(__file__))
if _STAGEDIR not in sys.path:
    sys.path.insert(0, _STAGEDIR)

from config import (  # noqa: E402
    DEFAULT_DB_REL,
    DEFAULT_MGF_REL,
    DEFAULT_MODEL_DIR_REL,
    default_inference_config,
    project_root,
)


def _resolve_user_path(path: str) -> str:
    """Resolve relative paths against the repository root; normalize absolute paths as-is."""
    p = path.strip().strip('"')
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(str(project_root() / p.replace("\\", "/")))


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = default_inference_config()
    if args.model_dir:
        cfg["MODEL_DIR"] = _resolve_user_path(args.model_dir)
    if args.mgf:
        cfg["MGF_FILE"] = _resolve_user_path(args.mgf)
    if args.db:
        cfg["DB_FILE"] = _resolve_user_path(args.db)
    if args.output_dir:
        cfg["OUTPUT_DIR"] = _resolve_user_path(args.output_dir)
    if args.out_all:
        cfg["OUTPUT_CSV_ALL"] = args.out_all
    if args.out_matched:
        cfg["OUTPUT_CSV_MATCHED"] = args.out_matched
    if args.outer_folds is not None:
        cfg["OUTER_FOLDS"] = int(args.outer_folds)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 Saponin Cascaded Inference (Detailed CSV + Matched Subset)")
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=f"Directory containing le_type and fold .joblib files, relative to the repository root (default: {DEFAULT_MODEL_DIR_REL})",
    )
    p.add_argument(
        "--mgf",
        type=str,
        default=None,
        help=f"Input MGF path, relative to the repository root (default: {DEFAULT_MGF_REL})",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help=f"Saponin database CSV (exact_mass, type, site columns, etc.), relative to the repository root (default: {DEFAULT_DB_REL})",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for both CSVs, relative to the repository root (default: output subdirectory under this module)",
    )
    p.add_argument("--out-all", type=str, default=None, help="Detailed results CSV filename (default: Task2_inference_results_detailed_final.csv)")
    p.add_argument(
        "--out-matched",
        type=str,
        default=None,
        help="CSV filename for entries with DB detailed matches only (default: Task2_inference_results_matched_only.csv)",
    )
    p.add_argument(
        "--outer-folds",
        type=int,
        default=None,
        help="Number of ensemble model folds, must match training OUTER_FOLDS (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from inference_runner import run_inference  # noqa: WPS433

    cfg = build_cfg(args)
    run_inference(cfg)


if __name__ == "__main__":
    main()
