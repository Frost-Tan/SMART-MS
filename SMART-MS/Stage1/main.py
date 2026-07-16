"""
Stage1 Binary Classification Training Entry Point.
It is recommended to run this script from the repository root (see command_line_instructions.txt).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

# Ensure modules in the same directory can be imported when running from the repo root or Stage1 directory
_STAGEDIR = os.path.dirname(os.path.abspath(__file__))
if _STAGEDIR not in sys.path:
    sys.path.insert(0, _STAGEDIR)

from config import DATA_RELPATH, default_config, project_root  # noqa: E402


def _resolve_user_path(path: str) -> str:
    """Resolve relative paths against the repository root; normalize absolute paths as-is."""
    p = path.strip().strip('"')
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(str(project_root() / p.replace("\\", "/")))


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = default_config()
    if args.data:
        cfg["DATA_PATH"] = _resolve_user_path(args.data)
    if args.output_dir:
        cfg["OUTPUT_DIR"] = _resolve_user_path(args.output_dir)
        cfg["MODEL_DIR"] = cfg["OUTPUT_DIR"]
    if args.model_dir:
        cfg["MODEL_DIR"] = _resolve_user_path(args.model_dir)
    if args.cv_folds is not None:
        cfg["CV_FOLDS"] = args.cv_folds
    if args.seed is not None:
        cfg["RANDOM_SEED"] = args.seed
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 LightGBM Binary Classification (Negative Ion Saponins) Training")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"CSV data path, relative to the repository root (default: {DATA_RELPATH})",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports and images, relative to the repository root (default: output subdirectory under this script's directory)",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model save directory, relative to the repository root (default: same as --output-dir, i.e., the output directory)",
    )
    p.add_argument("--cv-folds", type=int, default=None, help="Number of cross-validation folds (default: 5)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from training import run_training  # Import after parse_args to avoid loading lightgbm etc. before --help exits

    cfg = build_cfg(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
