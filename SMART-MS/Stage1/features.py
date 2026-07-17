"""Mass spectrum parsing and feature engineering."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_spectrum_and_filter(spectrum_str: Any, cfg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    mz_list, int_list = [], []
    if pd.isna(spectrum_str) or not isinstance(spectrum_str, str):
        return mz_list, int_list
    peaks = spectrum_str.split(" ")
    for peak in peaks:
        if ":" in peak:
            mz_val, intensity_val = peak.split(":")
            mz_list.append(float(mz_val))
            int_list.append(float(intensity_val))

    if not mz_list:
        return [], []

    int_array, mz_array = np.array(int_list), np.array(mz_list)
    if np.max(int_array) > 0:
        int_array = int_array / np.max(int_array)

    valid_indices = int_array >= cfg["INT_THRESHOLD"]
    return mz_array[valid_indices].tolist(), int_array[valid_indices].tolist()


def get_mh_baseline_mass(precursor_mz: float, adduct: Any, cfg: Dict[str, Any]) -> float:
    adduct_str = str(adduct).strip().upper() if not pd.isna(adduct) else ""
    diff = -1.0073

    for key, val in cfg["ADDUCT_MASS_DIFF"].items():
        if key.upper() in adduct_str:
            diff = val
            break

    neutral_mass = precursor_mz - diff
    mh_baseline_mass = neutral_mass + (-1.0073)
    return mh_baseline_mass


def extract_features(
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
    max_int_mz, weighted_mean_mz = 0.0, 0.0

    if mz_list and int_list:
        mz_array, int_array = np.array(mz_list), np.array(int_list)
        max_int_mz = mz_array[np.argmax(int_array)]
        weighted_mean_mz = np.sum(mz_array * int_array) / np.sum(int_array)

        for mz, intensity in zip(mz_array, int_array):
            if 0 <= mz < cfg["MAX_MZ"]:
                idx = int(mz / cfg["BIN_SIZE_SPEC"])
                vec_spec[idx] = max(vec_spec[idx], intensity)

            loss = mh_baseline_mass - mz
            if 0 <= loss < cfg["MAX_LOSS"]:
                l_idx = int(loss / cfg["BIN_SIZE_LOSS"])
                vec_loss[l_idx] = max(vec_loss[l_idx], intensity)

    scalars = [float(precursor_mz), float(mh_baseline_mass), float(max_int_mz), float(weighted_mean_mz)]
    return np.concatenate([vec_spec, vec_loss, scalars])


def generate_feature_names(cfg: Dict[str, Any]) -> List[str]:
    names = []
    n_bins_spec = int(cfg["MAX_MZ"] / cfg["BIN_SIZE_SPEC"]) + 1
    for i in range(n_bins_spec):
        names.append(f"Spec_mz_{i * cfg['BIN_SIZE_SPEC']:.2f}")

    n_bins_loss = int(cfg["MAX_LOSS"] / cfg["BIN_SIZE_LOSS"]) + 1
    for i in range(n_bins_loss):
        names.append(f"Loss_Da_{i * cfg['BIN_SIZE_LOSS']:.2f}")

    names.extend(["Precursor_MZ", "MH_Baseline_Mass", "BasePeak_MZ", "WeightedMean_MZ"])
    return names
