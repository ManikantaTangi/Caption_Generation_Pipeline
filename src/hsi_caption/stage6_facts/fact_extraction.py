"""
Stage 6 - Fact Extraction
============================
Purpose
    Convert the numeric/symbolic outputs of Stages 3-5 (material
    fractions, reasoning contexts, refined ontology concepts) into a
    flat list of atomic candidate facts -- short (subject, predicate,
    object)-shaped propositions -- before they are formally packaged
    into `Triplet` objects by the TripletGenerator.

Input
    Stage 3's `material_fractions`, Stage 4's `ReasonedKnowledge`, Stage
    5's `OntologySemanticRepresentation`.

Algorithm
    Deterministic template instantiation, one candidate fact per: (a)
    each material with fraction above `min_fact_score`->
    "patch hasMaterial <material>"; (b) each fired context ->
    "patch hasContext <context>"; (c) each refined ontology concept ->
    "patch hasConcept <concept>". This is intentionally simple/auditable
    (vs. an open-ended LLM extractor) since every fact must be
    traceable to a specific upstream stage for the Explainability
    requirement (Stage 9). Complexity: O(M + K + C).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from hsi_caption.datatypes import OntologySemanticRepresentation, ReasonedKnowledge

logger = logging.getLogger(__name__)


@dataclass
class CandidateFact:
    subject: str
    predicate: str
    object: str
    raw_score: float


class FactExtractor:
    """Extracts candidate (subject, predicate, object) facts from Stages 3-5."""

    def __init__(self, min_fact_score: float) -> None:
        self.min_fact_score = min_fact_score

    def extract(self, patch_id: str, material_fractions: Dict[str, float],
                reasoned: ReasonedKnowledge, ontology_rep: OntologySemanticRepresentation) -> List[CandidateFact]:
        facts: List[CandidateFact] = []
        for material, frac in material_fractions.items():
            if frac >= self.min_fact_score:
                facts.append(CandidateFact(patch_id, "hasMaterial", material, frac))
        for context in reasoned.contexts:
            facts.append(CandidateFact(patch_id, "hasContext", context, reasoned.reasoning_confidence))
        for concept in ontology_rep.refined_concepts:
            facts.append(CandidateFact(patch_id, "hasConcept", concept, ontology_rep.ontology_score))
        logger.debug("Extracted %d candidate facts for %s", len(facts), patch_id)
        return facts
