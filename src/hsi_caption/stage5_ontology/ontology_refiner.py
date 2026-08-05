"""
Stage 5 - Ontology Refiner
=============================
Purpose
    Iteratively tighten the refined-concept list against the ontology's
    formal axioms and coherence score, dropping concepts that create
    contradictions (e.g. a concept from a branch the ReasoningEngine
    explicitly suppressed in Stage 4) and producing the final
    `ontology_score` used by Stage 8's OntologyMatching verification.

Input
    Stage 4's `ReasonedKnowledge.suppressed_categories`, this stage's
    `List[str]` refined concepts (from SemanticGenerator) and
    `coherence_score` (from OntologyMatcher).

Algorithm
    Fixed-point-style iterative refinement, bounded by
    `refine_iterations`: each pass removes any concept whose ontology
    branch is in `suppressed_categories`; stops early if nothing changes.
    `ontology_score` = coherence_score, discounted 10% per concept
    removed (removed concepts indicate the initial match was noisier).
    Complexity: O(refine_iterations * M).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from hsi_caption.datatypes import OntologySemanticRepresentation
from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage5_ontology.ontology_matcher import OntologyMatch

logger = logging.getLogger(__name__)


class OntologyRefiner:
    """Iteratively refines concepts against suppressed categories."""

    def __init__(self, ontology: OntologyLoader, refine_iterations: int) -> None:
        self.ontology = ontology
        self.refine_iterations = refine_iterations

    def _branch_of(self, concept: str) -> str:
        if concept in ("Vegetation", "Water", "BuiltUp"):
            return concept
        path = [concept] + self.ontology.superclasses(concept)
        for node in path:
            if node in ("Vegetation", "Water", "BuiltUp"):
                return node
        return concept

    def refine(self, patch_id: str, concepts: List[str], suppressed_categories: List[str],
               coherence_score: float, matches: List[OntologyMatch]) -> OntologySemanticRepresentation:
        suppressed_branches = {
            "vegetation": "Vegetation", "water": "Water", "built-up": "BuiltUp",
        }
        suppressed_ontology_branches = {suppressed_branches.get(c, c) for c in suppressed_categories}

        refined = list(concepts)
        removed_count = 0
        for _ in range(self.refine_iterations):
            new_refined = [c for c in refined if self._branch_of(c) not in suppressed_ontology_branches]
            if len(new_refined) == len(refined):
                break
            removed_count += len(refined) - len(new_refined)
            refined = new_refined

        ontology_score = max(0.0, coherence_score * (0.9 ** removed_count))
        hierarchy_paths: Dict[str, List[str]] = {
            m.matched_class: m.hierarchy_path for m in matches if m.matched_class in refined
        }
        matched_classes = [m.matched_class for m in matches]

        return OntologySemanticRepresentation(
            patch_id=patch_id, matched_classes=matched_classes,
            class_hierarchy_paths=hierarchy_paths, refined_concepts=refined,
            ontology_score=float(ontology_score),
        )
