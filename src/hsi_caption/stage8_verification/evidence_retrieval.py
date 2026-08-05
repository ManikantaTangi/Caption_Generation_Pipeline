"""
Stage 8 - Evidence Retrieval
===============================
Purpose
    For each candidate triplet from Stage 6, gather the concrete pieces
    of upstream evidence that could support or contradict it -- the
    material's spectral-library similarity score (Stage 3), whether it
    appears in the ontology (Stage 5), and whether a rule explicitly
    asserted it (Stage 4). This bundles verification's raw inputs before
    the four independent checks (KG, ontology, rule, semantic-similarity)
    each consume it.

Input
    Stage 6's `Triplet`, Stage 3's `KnowledgeEmbedding`, Stage 4's
    `ReasonedKnowledge`, Stage 5's `OntologySemanticRepresentation`.

Algorithm
    Direct dict/list lookups tying each triplet's object back to its
    originating stage's evidence store. O(1) amortised per triplet given
    small fixed-size upstream structures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from hsi_caption.datatypes import KnowledgeEmbedding, OntologySemanticRepresentation, ReasonedKnowledge, Triplet

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    triplet: Triplet
    spectral_similarity: Optional[float]
    in_ontology: bool
    rule_asserted: bool
    in_kg: bool


class EvidenceRetriever:
    """Gathers cross-stage evidence for a single candidate triplet."""

    def retrieve(self, triplet: Triplet, ke: KnowledgeEmbedding, rk: ReasonedKnowledge,
                 osr: OntologySemanticRepresentation) -> Evidence:
        spectral_similarity = next(
            (m.similarity for m in ke.material_matches if m.material_name == triplet.object), None
        )
        in_ontology = triplet.object in osr.matched_classes or triplet.object in osr.refined_concepts
        rule_asserted = triplet.object in rk.contexts or triplet.object in rk.related_materials
        in_kg = any(triplet.object in node_id for node_id in ke.kg_node_ids)
        return Evidence(
            triplet=triplet, spectral_similarity=spectral_similarity,
            in_ontology=in_ontology, rule_asserted=rule_asserted, in_kg=in_kg,
        )
