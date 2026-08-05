"""
Stage 1 - Patch Generator
============================
Purpose
    Convert a full, normalized HSICube into a set of fixed-size
    spatial-spectral `Patch` objects, split into train/val/test, which
    is the concrete unit of work for every downstream stage (Stage 2
    encodes one patch at a time; Stage 9 captions one patch at a time).

Algorithm
    Sliding-window center sampling with configurable stride over labelled
    pixels only (background-heavy WHU-Hi scenes make full dense sampling
    wasteful). Each patch is extracted directly from the (unpadded)
    cube and *locally* reflection-padded only if its window would run
    past a scene border -- deliberately avoiding padding the entire
    cube up front (a real WHU-Hi-LongKou cube is ~550x400x270; padding
    the whole thing allocates another ~230MB copy purely to service a
    handful of border patches). Split is a stratified-by-class random
    shuffle of patch centres (train/val/test), which is standard practice
    for WHU-Hi benchmarking; we flag the option to switch to spatially
    disjoint block-splitting in `docs/ARCHITECTURE.md` to avoid spatial
    autocorrelation leakage in rigorous generalisation studies.
    Complexity: O(N_patches * patch_size^2 * B) for extraction, O(patch_size^2 * B)
    peak memory per patch (not O(H*W*B) for a full padded copy).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from hsi_caption.datatypes import HSICube, Patch, PatchDataset

logger = logging.getLogger(__name__)


class PatchGenerationError(Exception):
    """Raised when patch extraction parameters are invalid for the cube."""


class PatchGenerator:
    """Extracts fixed-size patches from a HSICube and splits them."""

    def __init__(self, patch_size: int, stride: int,
                 train_ratio: float, val_ratio: float, test_ratio: float,
                 random_seed: int = 42) -> None:
        if patch_size % 2 == 0:
            raise PatchGenerationError("patch_size must be odd so it has a well-defined centre pixel.")
        self.patch_size = patch_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = np.random.default_rng(random_seed)

    def _extract_centers(self, cube: HSICube) -> List[tuple]:
        if cube.labels is None:
            raise PatchGenerationError("Patch extraction requires labelled data.")
        rows, cols = np.where(cube.labels >= 0)  # every pixel incl. background is a valid centre
        centers = list(zip(rows[::self.stride], cols[::self.stride])) if self.stride > 1 else list(zip(rows, cols))
        return centers

    def _extract_single_patch_cube(self, cube: HSICube, r: int, c: int) -> np.ndarray:
        """Extracts one (patch_size, patch_size, B) window, reflection-padding
        only the border overhang (if any) rather than the whole cube."""
        half = self.patch_size // 2
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        vr0, vc0 = max(0, r0), max(0, c0)
        vr1, vc1 = min(cube.height, r1), min(cube.width, c1)

        sub = cube.data[vr0:vr1, vc0:vc1, :]
        pad_top, pad_bottom = vr0 - r0, r1 - vr1
        pad_left, pad_right = vc0 - c0, c1 - vc1
        if pad_top or pad_bottom or pad_left or pad_right:
            sub = np.pad(sub, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
        return sub

    def build_patches_from_centers(self, cube: HSICube, center_list: List[tuple],
                                     padded: Optional[np.ndarray] = None) -> List[Patch]:
        """Builds `Patch` objects for an explicit list of (row, col) centres.

        Exposed as a public method (rather than a private closure) so
        `OfficialSplitLoader` can reuse the exact same patch-construction
        logic when centres come from WHU-Hi's official Train*/Test*.mat
        masks instead of this class's own stratified random split.

        `padded` is accepted-but-ignored for backward compatibility with
        earlier call sites; patches are now extracted directly from
        `cube.data` with local border padding (see `_extract_single_patch_cube`)
        to avoid holding a full extra padded copy of the whole cube in memory.
        """
        half = self.patch_size // 2
        patches = []
        for r, c in center_list:
            cube_patch = self._extract_single_patch_cube(cube, r, c)
            label_patch = cube.labels[
                max(0, r - half):min(cube.height, r + half + 1),
                max(0, c - half):min(cube.width, c + half + 1),
            ]
            center_label = int(cube.labels[r, c])
            purity = float((label_patch == center_label).mean())
            metadata = {
                "row_norm": r / max(cube.height - 1, 1),
                "col_norm": c / max(cube.width - 1, 1),
                "local_class_purity": purity,
            }
            patches.append(Patch(
                patch_id=f"{cube.name}_r{r}_c{c}",
                cube_data=cube_patch.astype(np.float32, copy=False),
                center_label=center_label,
                label_patch=label_patch,
                center_row=r, center_col=c,
                metadata=metadata,
            ))
        return patches

    def generate(self, cube: HSICube, normalization_stats: Optional[Dict[str, np.ndarray]] = None) -> PatchDataset:
        centers = self._extract_centers(cube)

        by_class: Dict[int, List[tuple]] = defaultdict(list)
        for r, c in centers:
            by_class[int(cube.labels[r, c])].append((r, c))

        train_centers, val_centers, test_centers = [], [], []
        for cls_id, pts in by_class.items():
            pts = list(pts)
            self.rng.shuffle(pts)
            n = len(pts)
            n_train = int(round(n * self.train_ratio))
            n_val = int(round(n * self.val_ratio))
            train_centers.extend(pts[:n_train])
            val_centers.extend(pts[n_train:n_train + n_val])
            test_centers.extend(pts[n_train + n_val:])

        self.rng.shuffle(train_centers)
        self.rng.shuffle(val_centers)
        self.rng.shuffle(test_centers)

        train = self.build_patches_from_centers(cube, train_centers)
        val = self.build_patches_from_centers(cube, val_centers)
        test = self.build_patches_from_centers(cube, test_centers)
        logger.info("Generated patches: train=%d val=%d test=%d (patch_size=%d, stride=%d)",
                    len(train), len(val), len(test), self.patch_size, self.stride)

        return PatchDataset(
            train=train, val=val, test=test,
            wavelengths_nm=cube.wavelengths_nm, class_names=cube.class_names,
            normalization_stats=normalization_stats or {},
        )
