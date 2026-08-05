"""
Stage 9 - Template Selector
==============================
Purpose
    Choose the caption template bank appropriate to the patch's
    confidence band and load the raw template strings from
    configs/caption_templates.yaml -- kept separate from CaptionGenerator
    so the template *vocabulary* can be edited/extended by a domain
    expert without touching generation logic.

Input
    A `confidence_score` in [0, 1] (Stage 7, post Stage-8 adjustment).

Algorithm
    Direct range lookup against `caption.confidence_bands`
    (`high`/`medium`/`low`), each mapped to a small template list;
    one template is chosen deterministically (round-robin by patch
    index) for reproducibility across pipeline re-runs -- avoids
    non-reproducible randomness in the final, citable output the IEEE
    write-up would quote. O(1).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)


class TemplateSelector:
    """Selects a caption template string for a given confidence band."""

    def __init__(self, template_bank_path: str, confidence_bands: Dict[str, List[float]]) -> None:
        with open(template_bank_path, "r", encoding="utf-8") as fh:
            self.templates: Dict[str, List[str]] = yaml.safe_load(fh)["templates"]
        self.confidence_bands = confidence_bands
        self._round_robin_counters: Dict[str, int] = {band: 0 for band in self.confidence_bands}

    def band_for(self, confidence_score: float) -> str:
        for band, (lo, hi) in self.confidence_bands.items():
            if lo <= confidence_score < hi:
                return band
        return "low"

    def select(self, confidence_score: float) -> "tuple[str, str]":
        band = self.band_for(confidence_score)
        options = self.templates.get(band, self.templates.get("low", ["{dominant_material}."]))
        idx = self._round_robin_counters[band] % len(options)
        self._round_robin_counters[band] += 1
        return band, options[idx]
