"""
Stage 1 - Dataset Normalization
=================================
Purpose
    Rescale per-band reflectance to a numerically well-behaved range
    before patch extraction / encoding. Hyperspectral bands have very
    different dynamic ranges (water-absorption bands near-zero, NIR
    plateau bands much higher); leaving this unnormalized destabilizes
    both the spectral CNN encoder (Stage 2) and cosine/SAM similarity in
    Stage 3.

Algorithm
    min-max: (x - min) / (max - min)               -> range [0, 1]
    z-score: (x - mean) / std                        -> zero mean, unit var
    Both computed per-band using DatasetStatistics (fit on the *same*
    cube here; for held-out generalisation studies fit on train only).
    Complexity: O(H*W*B).
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from hsi_caption.datatypes import HSICube
from hsi_caption.stage1_preprocessing.dataset_statistics import DatasetStatistics

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Raised for unsupported normalization strategies."""


class DatasetNormalizer:
    """Applies per-band min-max or z-score normalization to a HSICube."""

    _SUPPORTED = ("minmax", "zscore")

    def __init__(self, strategy: str = "minmax") -> None:
        if strategy not in self._SUPPORTED:
            raise NormalizationError(f"Unsupported normalization strategy: {strategy}")
        self.strategy = strategy

    def fit_transform(self, cube: HSICube, stats: DatasetStatistics) -> "tuple[HSICube, Dict[str, np.ndarray]]":
        if self.strategy == "minmax":
            denom = np.where((stats.band_max - stats.band_min) < 1e-8, 1.0, stats.band_max - stats.band_min)
            normalized = (cube.data - stats.band_min) / denom
            params = {"band_min": stats.band_min, "band_max": stats.band_max}
        else:  # zscore
            normalized = (cube.data - stats.band_mean) / stats.band_std
            params = {"band_mean": stats.band_mean, "band_std": stats.band_std}

        normalized = normalized.astype(np.float32)
        logger.info("Applied %s normalization (range now [%.4f, %.4f])",
                    self.strategy, float(normalized.min()), float(normalized.max()))
        new_cube = HSICube(
            data=normalized, labels=cube.labels, wavelengths_nm=cube.wavelengths_nm,
            class_names=cube.class_names, name=cube.name + f"_norm-{self.strategy}",
        )
        return new_cube, params
