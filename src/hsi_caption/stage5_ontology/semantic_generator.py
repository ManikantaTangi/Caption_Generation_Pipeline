"""
Stage 5 - Semantic Generator
===============================
Purpose
    Turn the OntologyMatcher's raw class matches into a candidate list of
    "refined concepts" -- the set of ontology classes (leaf + informative
    ancestors) worth mentioning in the final caption -- balancing
    specificity (leaf class) against generality (useful when confidence
    is low, e.g. falling back to "Crop" instead of a specific cultivar).

Input
    `List[OntologyMatch]` from the OntologyMatcher (this stage).

Algorithm
    For the top-weighted match, walk its hierarchy path and keep the
    leaf class plus its immediate parent (two levels of specificity);
    for lower-weighted matches, keep only the leaf. This mirrors how a
    domain expert captions imagery: name the dominant material
    specifically, mention its category, and list secondary materials
    tersely. Complexity: O(M).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from hsi_caption.stage5_ontology.ontology_matcher import OntologyMatch

logger = logging.getLogger(__name__)


class SemanticGenerator:
    """Produces the refined-concept list from ontology matches."""

    def generate(self, matches: List[OntologyMatch]) -> List[str]:
        if not matches:
            return []
        concepts: List[str] = []
        top = matches[0]
        concepts.append(top.matched_class)
        if len(top.hierarchy_path) > 1:
            concepts.append(top.hierarchy_path[1])  # immediate parent
        for m in matches[1:]:
            concepts.append(m.matched_class)
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for c in concepts:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        return ordered
