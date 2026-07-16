"""Dual-pathway feature vector: spectrum bin + loss bin + meta features."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def get_mh_baseline_mass_hypothesis(precursor_mz, hypothesis_type):
    if hypothesis_type == "H":
        diff = -1.0073
    else:  # FA
        diff = 44.9982
    neutral_mass = precursor_mz - diff
    return neutral_mass - 1.0073


def extract_features_vectorized(
    mz_list: List[float],
    int_list: List[float],
    precursor_mz: float,
    mh_baseline_mass: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    n_bins_spec = int(cfg["MAX_MZ"] / cfg["BIN_SIZE_SPEC"]) + 1
    n_bins_loss = int(cfg["MAX_LOSS"] / cfg["BIN_SIZE_LOSS"]) + 1
    vec_spec = np.zeros(n_bins_spec, dtype=np.float32)
    vec_loss = np.zeros(n_bins_loss, dtype=np.float32)

    if not mz_list:
        return np.concatenate(
            [vec_spec, vec_loss, [float(precursor_mz), float(mh_baseline_mass), 0.0, 0.0]]
        )

    mz_array, int_array = np.array(mz_list, dtype=np.float32), np.array(int_list, dtype=np.float32)
    max_int = np.max(int_array)
    if max_int <= 0:
        return np.concatenate(
            [vec_spec, vec_loss, [float(precursor_mz), float(mh_baseline_mass), 0.0, 0.0]]
        )

    int_array = int_array / max_int
    mask = int_array >= cfg["INT_THRESHOLD"]
    mz_array, int_array = mz_array[mask], int_array[mask]

    if len(mz_array) == 0:
        return np.concatenate(
            [vec_spec, vec_loss, [float(precursor_mz), float(mh_baseline_mass), 0.0, 0.0]]
        )

    max_int_mz = mz_array[np.argmax(int_array)]
    weighted_mean_mz = np.sum(mz_array * int_array) / np.sum(int_array)

    spec_indices = (mz_array / cfg["BIN_SIZE_SPEC"]).astype(np.int32)
    valid_spec = (mz_array >= 0) & (mz_array < cfg["MAX_MZ"])
    np.maximum.at(vec_spec, spec_indices[valid_spec], int_array[valid_spec])

    loss_array = mh_baseline_mass - mz_array
    loss_indices = (loss_array / cfg["BIN_SIZE_LOSS"]).astype(np.int32)
    valid_loss = (loss_array >= 0) & (loss_array < cfg["MAX_LOSS"])
    np.maximum.at(vec_loss, loss_indices[valid_loss], int_array[valid_loss])

    return np.concatenate(
        [
            vec_spec,
            vec_loss,
            [
                float(precursor_mz),
                float(mh_baseline_mass),
                float(max_int_mz),
                float(weighted_mean_mz),
            ],
        ]
    )
