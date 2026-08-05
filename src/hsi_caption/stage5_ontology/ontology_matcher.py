"""
Stage 5 - Ontology Matcher
=============================
Purpose
    Match Stage 3's identified material names to formal ontology classes
    and resolve their full `subClassOf` chain, so a caption can later say
    "a cereal crop (rice)" instead of only the leaf label. Also scores
    how well the patch's material composition aligns with a single
    ontological branch (a mixed-category patch scores lower -- useful as
    a semantic-coherence signal).

Input
    `KnowledgeEmbedding.material_fractions` (Stage 3) and `OntologyLoader`
    (this stage).

Algorithm
    Exact string matching from material name to ontology class name
    (both vocabularies are curated to align 1:1; a production system
    with open-vocabulary materials would replace this with embedding
    similarity against class labels, e.g. Sentence-Transformers cosine
    similarity thresholded at `match_threshold` -- the interface below
    already returns a similarity score so that swap is drop-in).
    Coherence score = fraction of total probability mass whose matched
    classes share the *same* top-level branch (Vegetation/Water/BuiltUp).
    Complexity: O(M) per patch, M = materials with nonzero fraction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader

logger = logging.getLogger(__name__)


@dataclass
class OntologyMatch:
    material_name: str
    matched_class: str
    similarity: float
    hierarchy_path: List[str]


class OntologyMatcher:
    """Matches material fractions to ontology classes and scores coherence."""

    def __init__(self, ontology: OntologyLoader, match_threshold: float) -> None:
        self.ontology = ontology
        self.match_threshold = match_threshold

    def match(self, material_fractions: Dict[str, float]) -> List[OntologyMatch]:
        matches = []
        for name, frac in material_fractions.items():
            if frac <= 0.0:
                continue
            if name in self.ontology.hierarchy:
                similarity = 1.0  # exact vocabulary match
                path = [name] + self.ontology.superclasses(name)
            else:
                similarity = 0.0
                path = []
            if similarity >= self.match_threshold:
                matches.append(OntologyMatch(name, name, similarity, path))
        matches.sort(key=lambda m: material_fractions[m.material_name], reverse=True)
        return matches

    @staticmethod
    def _top_level_branch(path: List[str]) -> str:
        # top-level branch = the ancestor directly under "LandCover" (or "Entity" if shallow)
        for node in reversed(path):
            if node in ("Vegetation", "Water", "BuiltUp"):
                return node
        return path[-1] if path else "unknown"

    def coherence_score(self, matches: List[OntologyMatch], material_fractions: Dict[str, float]) -> float:
        if not matches:
            return 0.0
        branches = [self._top_level_branch(m.hierarchy_path) for m in matches]
        weights = [material_fractions[m.material_name] for m in matches]
        total = sum(weights)
        if total == 0:
            return 0.0
        from collections import Counter
        branch_weight = Counter()
        for b, w in zip(branches, weights):
            branch_weight[b] += w
        return max(branch_weight.values()) / total
