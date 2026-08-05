"""
Stage 6 - Relationship Generator
===================================
Purpose
    Add *inter-entity* relational triples that FactExtraction's per-patch
    template instantiation cannot produce alone -- specifically,
    material-to-material relations surfaced by Stage 3's knowledge graph
    (`similarTo`) and Stage 4's rule-derived `related_materials`
    (e.g. Rice `coOccursWith` Water). These give Stage 9 the raw
    material for relational language ("alongside", "adjacent to").

Input
    Stage 3's `KnowledgeEmbedding.kg_node_ids` / material matches, Stage
    4's `ReasonedKnowledge.related_materials`.

Algorithm
    For every fired rule's `related_material`, emit
    "<dominant_material> coOccursWith <related_material>" if the related
    material's fraction is also above the extraction floor (avoids
    asserting co-occurrence with an absent material). Complexity: O(K).
"""
from __future__ import annotations

import logging
from typing import Dict, List

from hsi_caption.datatypes import ReasonedKnowledge, Triplet

logger = logging.getLogger(__name__)


class RelationshipGenerator:
    """Generates material-to-material relational triples."""

    def __init__(self, relation_confidence_floor: float) -> None:
        self.relation_confidence_floor = relation_confidence_floor

    def generate(self, dominant_material: str, material_fractions: Dict[str, float],
                 reasoned: ReasonedKnowledge) -> List[Triplet]:
        triplets: List[Triplet] = []
        for related in reasoned.related_materials:
            frac = material_fractions.get(related, 0.0)
            if frac >= self.relation_confidence_floor and related != dominant_material:
                triplets.append(Triplet(
                    subject=dominant_material, predicate="coOccursWith", object=related,
                    score=float(min(frac, reasoned.reasoning_confidence)),
                ))
        return triplets
