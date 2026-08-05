"""
Stage 1 - Dataset Validator
=============================
Purpose
    Fail fast if the loaded HSICube is structurally unsound: shape
    mismatches, NaN/Inf contamination, out-of-range labels, or a band
    count that disagrees with configuration. Prevents silent propagation
    of corrupt data through 8 downstream stages.

Algorithm
    A fixed battery of O(H*W*B) or better checks; short-circuits on the
    first violation category but reports *all* violations found within
    that pass for actionable debugging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from hsi_caption.datatypes import HSICube

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when a HSICube fails structural validation."""


@dataclass
class ValidationReport:
    is_valid: bool
    issues: List[str] = field(default_factory=list)


class DatasetValidator:
    """Validates structural integrity of a HSICube before use."""

    def __init__(self, expected_num_bands: int, num_classes: int) -> None:
        self.expected_num_bands = expected_num_bands
        self.num_classes = num_classes

    def validate(self, cube: HSICube) -> ValidationReport:
        issues: List[str] = []

        if cube.data.ndim != 3:
            issues.append(f"Expected 3D cube (H,W,B), got shape {cube.data.shape}")
        else:
            if cube.data.shape[2] != self.expected_num_bands:
                issues.append(
                    f"Band count mismatch: cube has {cube.data.shape[2]}, "
                    f"config expects {self.expected_num_bands}"
                )
            if np.isnan(cube.data).any():
                issues.append("Cube contains NaN values.")
            if np.isinf(cube.data).any():
                issues.append("Cube contains Inf values.")
            if cube.data.min() < -1e-3:
                issues.append(f"Cube contains negative reflectance values (min={cube.data.min():.4f}).")

        if cube.labels is not None:
            if cube.labels.shape != cube.data.shape[:2]:
                issues.append(f"Label shape {cube.labels.shape} != image spatial shape {cube.data.shape[:2]}")
            else:
                max_label = int(cube.labels.max())
                if max_label >= self.num_classes:
                    issues.append(f"Label id {max_label} exceeds num_classes={self.num_classes}")
                if cube.labels.min() < 0:
                    issues.append("Negative label id found.")

        if cube.wavelengths_nm.shape[0] != cube.data.shape[2]:
            issues.append("wavelengths_nm length does not match number of bands.")

        report = ValidationReport(is_valid=len(issues) == 0, issues=issues)
        if not report.is_valid:
            logger.error("Dataset validation failed for %s: %s", cube.name, issues)
        else:
            logger.info("Dataset validation passed for %s.", cube.name)
        return report

    def validate_or_raise(self, cube: HSICube) -> None:
        report = self.validate(cube)
        if not report.is_valid:
            raise DatasetValidationError("; ".join(report.issues))
