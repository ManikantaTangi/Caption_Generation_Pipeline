"""
Stage 5 - Ontology-Enriched Semantic Generator (facade)
===========================================================
Wires OntologyLoader + OntologyMatcher + SemanticGenerator +
OntologyRefiner into one callable consuming Stage 3's material
fractions and Stage 4's ReasonedKnowledge, producing the Stage 5
`OntologySemanticRepresentation`.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from hsi_caption.config import OntologyConfig
from hsi_caption.datatypes import OntologySemanticRepresentation, ReasonedKnowledge
from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage5_ontology.ontology_matcher import OntologyMatcher
from hsi_caption.stage5_ontology.ontology_refiner import OntologyRefiner
from hsi_caption.stage5_ontology.semantic_generator import SemanticGenerator

logger = logging.getLogger(__name__)


class OntologyEngine:
    """Facade combining all four Stage 5 modules."""

    def __init__(self, cfg: OntologyConfig) -> None:
        self.ontology = OntologyLoader(cfg.ontology_path)
        self.matcher = OntologyMatcher(self.ontology, cfg.match_threshold)
        self.generator = SemanticGenerator()
        self.refiner = OntologyRefiner(self.ontology, cfg.refine_iterations)

    def process(self, patch_id: str, material_fractions: Dict[str, float],
                reasoned: ReasonedKnowledge) -> OntologySemanticRepresentation:
        matches = self.matcher.match(material_fractions)
        concepts = self.generator.generate(matches)
        coherence = self.matcher.coherence_score(matches, material_fractions)
        return self.refiner.refine(patch_id, concepts, reasoned.suppressed_categories, coherence, matches)
