"""Spectrum feature vector (consistent with the original Stage2_inference script, including adduct calibration)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def extract_advanced_features_from_spec(
    spec: Dict[str, Any],
    cfg: Dict[str, Any],
    adduct_mode: str,
) -> np.ndarray | None:
    max_mz = float(cfg["MAX_MZ"])
    bin_size = float(cfg["BIN_SIZE"])
    max_loss = float(cfg["MAX_LOSS"])
    loss_bin_size = float(cfg["LOSS_BIN_SIZE"])

    n_bins_spec = int(max_mz / bin_size) + 1
    n_bins_loss = int(max_loss / loss_bin_size) + 1
    vec_spec = np.zeros(n_bins_spec, dtype=np.float32)
    vec_loss = np.zeros(n_bins_loss, dtype=np.float32)

    pmass_obs = float(spec.get("PEPMASS", 0.0))
    if "[M+HCOO]-" in adduct_mode or "[M+FA-H]-" in adduct_mode:
        pmass_baseline = pmass_obs - float(cfg["FA_NEUTRAL_MASS"])
    else:
        pmass_baseline = pmass_obs

    mz_array = spec["mz"]
    int_array = spec["intensity"]
    if len(mz_array) == 0:
        return None
    if np.max(int_array) > 0:
        int_array = int_array / np.max(int_array)

    n_peaks = len(mz_array)
    max_int_mz = mz_array[np.argmax(int_array)]
    weighted_mean_mz = (
        np.sum(mz_array * int_array) / np.sum(int_array) if np.sum(int_array) > 0 else 0.0
    )

    for mz, intensity in zip(mz_array, int_array):
        if 0 <= mz < max_mz:
            vec_spec[int(mz / bin_size)] = max(vec_spec[int(mz / bin_size)], float(intensity))
        if pmass_baseline > mz:
            loss = pmass_baseline - mz
            if 0 <= loss < max_loss:
                idx = int(loss / loss_bin_size)
                vec_loss[idx] = max(vec_loss[idx], float(intensity))

    return np.concatenate(
        [
            vec_spec,
            vec_loss,
            np.array(
                [pmass_baseline, max_int_mz, weighted_mean_mz, n_peaks, 0.0, 0.0],
                dtype=np.float32,
            ),
        ]
    )
