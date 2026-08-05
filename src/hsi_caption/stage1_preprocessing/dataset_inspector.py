"""
Stage 1 - Dataset Inspector
=============================
Purpose
    Produce a structured, human- and machine-readable summary of a
    validated HSICube (shape, per-class pixel counts, dynamic range).
    Used both for interactive debugging and for the automatic Stage
    Module Report.

Algorithm
    Single-pass aggregation over labels (O(H*W)) plus vectorised
    per-band summary stats (O(H*W*B)).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from hsi_caption.datatypes import HSICube

logger = logging.getLogger(__name__)


@dataclass
class InspectionSummary:
    name: str
    height: int
    width: int
    num_bands: int
    dtype: str
    value_min: float
    value_max: float
    value_mean: float
    class_pixel_counts: Dict[str, int] = field(default_factory=dict)
    class_pixel_fraction: Dict[str, float] = field(default_factory=dict)

    def as_text(self) -> str:
        lines = [
            f"Cube: {self.name}",
            f"  Shape: {self.height} x {self.width} x {self.num_bands} bands ({self.dtype})",
            f"  Value range: [{self.value_min:.4f}, {self.value_max:.4f}], mean={self.value_mean:.4f}",
            "  Class distribution:",
        ]
        for cls, frac in self.class_pixel_fraction.items():
            lines.append(f"    {cls:<22s}: {self.class_pixel_counts[cls]:>7d} px ({frac*100:5.2f}%)")
        return "\n".join(lines)


class DatasetInspector:
    """Computes a structural + class-distribution summary of a HSICube."""

    def inspect(self, cube: HSICube) -> InspectionSummary:
        h, w, b = cube.data.shape
        summary = InspectionSummary(
            name=cube.name, height=h, width=w, num_bands=b, dtype=str(cube.data.dtype),
            value_min=float(cube.data.min()), value_max=float(cube.data.max()),
            value_mean=float(cube.data.mean()),
        )
        if cube.labels is not None:
            total = cube.labels.size
            for idx, cls_name in enumerate(cube.class_names):
                count = int((cube.labels == idx).sum())
                summary.class_pixel_counts[cls_name] = count
                summary.class_pixel_fraction[cls_name] = count / total
        logger.info("Inspected cube %s: %dx%dx%d", cube.name, h, w, b)
        return summary
