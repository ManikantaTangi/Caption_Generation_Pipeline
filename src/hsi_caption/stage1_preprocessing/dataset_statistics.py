"""
Stage 1 - Dataset Statistics
==============================
Purpose
    Compute per-band and per-class statistics required by (a) the
    Normalization module immediately downstream, and (b) reporting
    (class imbalance ratio is a standard IEEE-paper table entry for
    WHU-Hi-style benchmarks).

Algorithm
    Vectorised NumPy reductions. Per-band mean/std: O(H*W*B).
    Class imbalance ratio: max class count / min non-zero class count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from hsi_caption.datatypes import HSICube

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    band_mean: np.ndarray
    band_std: np.ndarray
    band_min: np.ndarray
    band_max: np.ndarray
    class_counts: Dict[str, int]
    class_imbalance_ratio: float


class DatasetStatisticsComputer:
    """Computes normalization-relevant and reporting-relevant statistics."""

    def compute(self, cube: HSICube) -> DatasetStatistics:
        flat = cube.data.reshape(-1, cube.num_bands)
        band_mean = flat.mean(axis=0)
        band_std = flat.std(axis=0) + 1e-8
        band_min = flat.min(axis=0)
        band_max = flat.max(axis=0)

        class_counts: Dict[str, int] = {}
        imbalance_ratio = float("nan")
        if cube.labels is not None:
            for idx, cls_name in enumerate(cube.class_names):
                class_counts[cls_name] = int((cube.labels == idx).sum())
            nonzero_counts = [c for c in class_counts.values() if c > 0]
            if nonzero_counts:
                imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)

        logger.info("Computed dataset statistics (imbalance ratio=%.2f)", imbalance_ratio)
        return DatasetStatistics(
            band_mean=band_mean, band_std=band_std, band_min=band_min, band_max=band_max,
            class_counts=class_counts, class_imbalance_ratio=imbalance_ratio,
        )
