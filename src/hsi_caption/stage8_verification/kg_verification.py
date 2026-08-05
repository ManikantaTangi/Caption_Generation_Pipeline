"""
Stage 8 - Knowledge Graph Verification
==========================================
Purpose
    Score a triplet's plausibility by checking whether its object node
    actually exists in Stage 3's static knowledge graph and, for
    `coOccursWith` relational triplets, whether the two materials are
    graph-connected via a `similarTo` edge -- catching hallucinated
    material names that never appeared in retrieval.

Input
    `Evidence.in_kg` (this module's primary signal) and the
    `KnowledgeGraphBuilder` (Stage 3) for relational lookups.

Algorithm
    Binary-plus-graph-distance score: 1.0 if the object node is present
    in the graph and (for relational triplets) directly connected to the
    subject by `similarTo`; 0.5 if present but unconnected; 0.0 if
    entirely absent. O(1) using NetworkX's adjacency lookup.
"""
from __future__ import annotations

import logging

from hsi_caption.stage3_knowledge_engine.knowledge_graph import KnowledgeGraphBuilder
from hsi_caption.stage8_verification.evidence_retrieval import Evidence

logger = logging.getLogger(__name__)


class KnowledgeGraphVerifier:
    """Verifies a triplet against the static Stage-3 knowledge graph."""

    def __init__(self, kg: KnowledgeGraphBuilder) -> None:
        self.kg = kg

    def verify(self, evidence: Evidence) -> float:
        if not evidence.in_kg:
            return 0.0
        triplet = evidence.triplet
        if triplet.predicate == "coOccursWith":
            related = self.kg.related_materials(triplet.subject)
            return 1.0 if triplet.object in related else 0.5
        return 1.0
