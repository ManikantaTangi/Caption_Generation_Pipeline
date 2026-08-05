"""
Stage 1 - Official Split Loader
===================================
Purpose
    Build train/val/test `PatchDataset` splits from WHU-Hi's official,
    published benchmark mask files (`Train<N>.mat` / `Test<N>.mat` under
    "Training samples and test samples/") instead of PatchGenerator's own
    ad-hoc stratified random split. Using the official split makes
    reported accuracy/kappa numbers directly comparable to published
    WHU-Hi benchmark results in the literature -- important for an IEEE
    write-up.

Input
    A normalized `HSICube`, plus paths to a `Train<N>.mat` and matching
    `Test<N>.mat` mask file. Each mask is a `(H, W)` array holding the
    1-indexed class id at pixels selected for that split, and 0
    elsewhere (WHU-Hi's own convention -- verified against
    `data/WHU-Hi-LongKou/Training samples and test samples/Train100.mat`
    during integration testing: exactly `N` pixels per class 1..C).

Algorithm
    Read each mask, take `np.where(mask > 0)` as the centre-pixel list
    for that split, and reuse `PatchGenerator.build_patches_from_centers`
    (the *same* patch-construction code path used by the random-split
    generator) so both splitting strategies produce identical `Patch`
    objects. The official files do not define a validation split, so a
    small validation set is carved out of the training centres
    (stratified, `val_fraction` of each class) purely for Stage 7's
    calibration and early stopping -- it never overlaps the official
    test set. Complexity: O(H*W) to read masks + O(N_patches * P^2 * B)
    for extraction (same as PatchGenerator).
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat

from hsi_caption.datatypes import HSICube, PatchDataset
from hsi_caption.stage1_preprocessing.patch_generator import PatchGenerationError, PatchGenerator

logger = logging.getLogger(__name__)


class OfficialSplitLoadError(Exception):
    """Raised when an official split mask file cannot be parsed or is inconsistent."""


class OfficialSplitLoader:
    """Loads WHU-Hi's official Train<N>/Test<N> mask files as a PatchDataset."""

    def __init__(self, patch_generator: PatchGenerator, val_fraction: float = 0.1) -> None:
        self.patch_generator = patch_generator
        self.val_fraction = val_fraction

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise OfficialSplitLoadError(f"Split mask file not found: {path}")
        mat = loadmat(path)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        if not arrays:
            raise OfficialSplitLoadError(f"No arrays found in {path}")
        key = max(arrays, key=lambda k: np.asarray(arrays[k]).size)
        return np.asarray(arrays[key])

    def load(self, cube: HSICube, train_mask_path: str, test_mask_path: str,
              normalization_stats: Optional[Dict[str, np.ndarray]] = None) -> PatchDataset:
        train_mask = self._load_mask(train_mask_path)
        test_mask = self._load_mask(test_mask_path)

        if train_mask.shape != cube.labels.shape or test_mask.shape != cube.labels.shape:
            raise OfficialSplitLoadError(
                f"Mask shape {train_mask.shape}/{test_mask.shape} does not match cube label shape "
                f"{cube.labels.shape}. Did you load the matching scene's masks?"
            )

        train_rows, train_cols = np.where(train_mask > 0)
        test_rows, test_cols = np.where(test_mask > 0)
        all_train_centers = list(zip(train_rows.tolist(), train_cols.tolist()))
        test_centers = list(zip(test_rows.tolist(), test_cols.tolist()))

        overlap = set(all_train_centers) & set(test_centers)
        if overlap:
            raise OfficialSplitLoadError(
                f"Train/test masks overlap at {len(overlap)} pixel(s) -- refusing to build a leaking split."
            )

        # carve a small stratified validation set out of the training centres only
        by_class: Dict[int, List[tuple]] = defaultdict(list)
        for r, c in all_train_centers:
            by_class[int(train_mask[r, c])].append((r, c))

        rng = self.patch_generator.rng
        train_centers, val_centers = [], []
        for cls_id, pts in by_class.items():
            pts = list(pts)
            rng.shuffle(pts)
            n_val = max(1, int(round(len(pts) * self.val_fraction))) if len(pts) > 1 else 0
            val_centers.extend(pts[:n_val])
            train_centers.extend(pts[n_val:])

        train = self.patch_generator.build_patches_from_centers(cube, train_centers)
        val = self.patch_generator.build_patches_from_centers(cube, val_centers)
        test = self.patch_generator.build_patches_from_centers(cube, test_centers)

        logger.info(
            "Loaded official split from %s / %s: train=%d (val carved out=%d) test=%d",
            os.path.basename(train_mask_path), os.path.basename(test_mask_path),
            len(train), len(val), len(test),
        )
        return PatchDataset(
            train=train, val=val, test=test,
            wavelengths_nm=cube.wavelengths_nm, class_names=cube.class_names,
            normalization_stats=normalization_stats or {},
        )
