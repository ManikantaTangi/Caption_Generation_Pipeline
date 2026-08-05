"""
Stage 1 - Dataset Discovery
============================
Purpose
    Locate WHU-Hi dataset artefacts (image cube + ground-truth label file)
    on disk under `dataset.root_dir`, without assuming a single fixed
    filename convention (the public WHU-Hi releases ship as .mat, some
    mirrors as .npy). If nothing is found, falls back to declaring a
    synthetic-generation request so the pipeline still runs end-to-end
    (see dataset_loader.SyntheticWHUHiGenerator).

Algorithm
    Directory walk + extension whitelist matching, O(N) in the number of
    files under root_dir.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".mat", ".npy", ".tif", ".tiff")
_LABEL_HINTS = ("gt", "label", "groundtruth")


class DatasetDiscoveryError(Exception):
    """Raised when discovery finds an unusable / ambiguous file set."""


@dataclass
class DiscoveredDataset:
    root_dir: str
    image_files: List[str]
    label_files: List[str]
    use_synthetic: bool


class DatasetDiscovery:
    """Scans a directory for WHU-Hi image/label artefacts."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    def discover(self) -> DiscoveredDataset:
        if not os.path.isdir(self.root_dir):
            logger.warning("root_dir %s does not exist; will use synthetic data.", self.root_dir)
            return DiscoveredDataset(self.root_dir, [], [], use_synthetic=True)

        candidates = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.lower().endswith(_IMAGE_EXTENSIONS)
        ]
        label_files = [f for f in candidates if any(h in os.path.basename(f).lower() for h in _LABEL_HINTS)]
        image_files = [f for f in candidates if f not in label_files]

        use_synthetic = len(image_files) == 0
        if use_synthetic:
            logger.info("No dataset artefacts found under %s; synthetic generation will be used.", self.root_dir)
        else:
            logger.info("Discovered %d image file(s), %d label file(s) under %s",
                        len(image_files), len(label_files), self.root_dir)
        return DiscoveredDataset(self.root_dir, image_files, label_files, use_synthetic)
