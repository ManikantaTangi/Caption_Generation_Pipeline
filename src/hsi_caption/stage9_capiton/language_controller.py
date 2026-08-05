"""
Stage 9 - Confidence-aware Language Controller
===================================================
Purpose
    Adjust the *hedging language* of the raw caption to match its
    confidence band beyond what the template alone guarantees -- e.g.
    ensuring low-confidence captions never contain unhedged assertive
    verbs ("is") introduced accidentally by a filled-in fact string, and
    that high-confidence captions aren't needlessly hedgy.

Input
    Raw caption string (this stage), `confidence_band`.

Algorithm
    A small deterministic lexical substitution table per band (e.g. for
    "low", replace bare "is" -> "may be", "shows" -> "appears to show").
    Rule-based lexical control is used instead of a learned rewriting
    model because the substitution set is small, fully auditable, and
    must never invert factual polarity -- a property much harder to
    guarantee with a learned paraphraser. Complexity: O(len(caption)).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_HEDGE_SUBSTITUTIONS = {
    "low": [(r"\bis\b", "may be"), (r"\bshows\b", "appears to show"), (r"\bconfirms\b", "suggests")],
    "medium": [(r"\bconfirms\b", "indicates")],
    "high": [],  # no softening needed; template language already reflects high confidence
}


class LanguageController:
    """Applies band-appropriate hedging substitutions to a raw caption."""

    def control(self, caption: str, confidence_band: str) -> str:
        for pattern, replacement in _HEDGE_SUBSTITUTIONS.get(confidence_band, []):
            caption = re.sub(pattern, replacement, caption)
        return caption
