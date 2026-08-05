"""
Stage 9 - Caption Generator
==============================
Purpose
    Instantiate the selected template with patch-specific values drawn
    from Stage 8's `VerifiedFacts` -- and *only* verified facts, never
    rejected ones -- producing the raw (pre-refinement) caption text.

Input
    Stage 8's `VerifiedFacts.verified` triplets, Stage 7's
    `confidence_score`, the selected template string (this stage).

Algorithm
    Slot-filling: `dominant_material` = highest-score verified
    `hasMaterial` triplet's object (falling back to the highest-score
    verified triplet of any predicate if no `hasMaterial` triplet
    survived verification); `secondary_material` = second-highest. This
    is a template-slot-filling NLG approach (deliberately not a
    free-generation LLM call) so every word in the caption is directly
    traceable to a verified fact -- required for the pipeline's
    auditability guarantee. Contextual detail (which rule fired, which
    spectral match, why) is intentionally left out of the caption itself
    and surfaced instead in `ExplanationGenerator`'s companion text, so
    the caption stays concise while the reasoning stays fully auditable.
    O(F).
"""
from __future__ import annotations

import logging
from typing import List

from hsi_caption.datatypes import Triplet

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Fills a template string using Stage-8 verified facts."""

    def _dominant_and_secondary(self, verified: List[Triplet]) -> "tuple[str, str]":
        materials = sorted(
            [t for t in verified if t.predicate == "hasMaterial"], key=lambda t: t.score, reverse=True
        )
        if not materials:
            materials = sorted(verified, key=lambda t: t.score, reverse=True)
        dominant = materials[0].object if materials else "an unidentified material"
        secondary = materials[1].object if len(materials) > 1 else "no clearly secondary material"
        return dominant, secondary

    def generate(self, template: str, verified: List[Triplet], confidence_score: float) -> str:
        dominant, secondary = self._dominant_and_secondary(verified)
        caption = template.format(
            dominant_material=dominant, secondary_material=secondary,
            confidence_pct=round(confidence_score * 100),
        )
        return caption
