"""
Stage 6 - Structured Fact Generator (facade)
================================================
Wires FactExtraction + TripletGenerator + RelationshipGenerator +
FactRanking into one callable consuming Stage 3/4/5 outputs and
producing the Stage 6 `StructuredFacts`.

Note: Stage 7's trained-classifier prediction is deliberately NOT
injected here. It enters the pipeline in Stage 8 instead (see
`FactVerificationEngine` / `ClassifierAgreementVerifier`), which keeps
a single, auditable injection point for "candidates Stage 3 didn't
propose on its own" and lets the classifier's full probability
distribution (not just its top-1 guess) score every candidate material,
not only the one Stage 6 would have hard-coded in.
"""
from __future__ import annotations

import logging
from typing import Dict

from hsi_caption.config import FactConfig
from hsi_caption.datatypes import OntologySemanticRepresentation, ReasonedKnowledge, StructuredFacts
from hsi_caption.stage6_facts.fact_extraction import FactExtractor
from hsi_caption.stage6_facts.fact_ranking import FactRanking
from hsi_caption.stage6_facts.relationship_generator import RelationshipGenerator
from hsi_caption.stage6_facts.triplet_generator import TripletGenerator

logger = logging.getLogger(__name__)


class StructuredFactGenerator:
    """Facade combining all four Stage 6 modules."""

    def __init__(self, cfg: FactConfig) -> None:
        self.extractor = FactExtractor(cfg.min_fact_score)
        self.triplet_generator = TripletGenerator()
        self.relationship_generator = RelationshipGenerator(cfg.relation_confidence_floor)
        self.ranking = FactRanking(cfg.min_fact_score, cfg.max_facts_per_patch)

    def process(self, patch_id: str, material_fractions: Dict[str, float],
                reasoned: ReasonedKnowledge, ontology_rep: OntologySemanticRepresentation) -> StructuredFacts:
        candidates = self.extractor.extract(patch_id, material_fractions, reasoned, ontology_rep)
        base_triplets = self.triplet_generator.generate(candidates)

        dominant_material = max(material_fractions, key=material_fractions.get) if material_fractions else "unknown"
        relation_triplets = self.relationship_generator.generate(dominant_material, material_fractions, reasoned)

        return self.ranking.rank(patch_id, base_triplets + relation_triplets)
