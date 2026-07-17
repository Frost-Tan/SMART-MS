"""
Stage1 Inference Entry Point. It is recommended to run from the repository root (consistent with Stage1 training).
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
    DEFAULT_INPUT_MGF_REL,
    DEFAULT_MODELS_DIR_REL,
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
    if args.models_dir:
        cfg["MODELS_DIR"] = _resolve_user_path(args.models_dir)
    if args.mgf:
        cfg["INPUT_MGF"] = _resolve_user_path(args.mgf)
    if args.output_dir:
        cfg["OUTPUT_DIR"] = _resolve_user_path(args.output_dir)
    if args.out_h:
        cfg["OUT_H_ONLY"] = args.out_h
    if args.out_fa:
        cfg["OUT_FA_ONLY"] = args.out_fa
    if args.out_best:
        cfg["OUT_BEST_MATCH"] = args.out_best
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage1 Saponin Binary Classification Inference (Three CSV Output)")
    p.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=f"Directory containing .joblib models, relative to the repository root (default: {DEFAULT_MODELS_DIR_REL})",
    )
    p.add_argument(
        "--mgf",
        type=str,
        default=None,
        help=f"Input MGF path, relative to the repository root (default: {DEFAULT_INPUT_MGF_REL})",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for three CSVs, relative to the repository root (default: output subdirectory under this module)",
    )
    p.add_argument("--out-h", type=str, default=None, help="H hypothesis result CSV filename (default: Task1_inference_...)")
    p.add_argument("--out-fa", type=str, default=None, help="FA hypothesis result CSV filename")
    p.add_argument("--out-best", type=str, default=None, help="Best match result CSV filename")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from inference_runner import run_inference  # noqa: WPS433

    cfg = build_cfg(args)
    run_inference(cfg)


if __name__ == "__main__":
    main()
