"""
Stage2 Cascaded Training Entry Point. It is recommended to run from the repository root (see command_line_instructions.txt).
By default, only Strategy 3 (full pipeline) is trained; add --all-ablations for all three ablation strategies.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

_STAGEDIR = os.path.dirname(os.path.abspath(__file__))
if _STAGEDIR not in sys.path:
    sys.path.insert(0, _STAGEDIR)

from config import DATA_RELPATH, default_config, project_root  # noqa: E402


def _resolve_user_path(path: str) -> str:
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
    if args.all_ablations:
        cfg["RUN_ALL_ABLATIONS"] = True
    if args.outer_folds is not None:
        cfg["OUTER_FOLDS"] = args.outer_folds
    if args.inner_folds is not None:
        cfg["INNER_FOLDS"] = args.inner_folds
    if args.seed is not None:
        cfg["RANDOM_SEED"] = args.seed
    if args.device:
        cfg["LGBM_DEVICE"] = args.device.lower()
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 Saponin Cascaded LightGBM (default: Strategy 3 Full Pipeline)")
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
        help="Root output directory, relative to the repository root (default: SMART-MS/Stage2/output)",
    )
    p.add_argument(
        "--all-ablations",
        action="store_true",
        help="Run all three ablation strategies (1/2/3); without this flag, only Strategy 3 (full pipeline) is trained",
    )
    p.add_argument("--outer-folds", type=int, default=None, help="Number of outer cross-validation folds (default: 10)")
    p.add_argument("--inner-folds", type=int, default=None, help="Number of inner OOF folds (default: 10)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    p.add_argument(
        "--device",
        type=str,
        choices=("cpu", "gpu"),
        default=None,
        help="LightGBM device: cpu or gpu (default: cpu, suitable for environments without GPU)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from training import run_ablation  # noqa: WPS433

    cfg = build_cfg(args)
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    run_ablation(cfg)


if __name__ == "__main__":
    main()
