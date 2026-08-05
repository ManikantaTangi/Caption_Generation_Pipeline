"""
Stage 8 - Fact Verification (facade)
========================================
Wires EvidenceRetrieval + KnowledgeGraphVerifier + OntologyVerification
Matcher + RuleVerifier + SemanticSimilarityVerifier +
ClassifierAgreementVerifier + VerificationConfidenceFusion into one
callable consuming Stage 3/4/5/6/7 outputs and producing Stage 8's
`VerifiedFacts`.

Stage 7's trained classifier is folded in here (not silently applied in
Stage 9) for two reasons: (1) it keeps Stage 8 the single place where
every material candidate is cross-checked, auditable, and explainable;
(2) it lets the classifier's own top prediction compete on equal footing
with Stage 3's spectral-fraction-derived candidates, rather than one
silently overriding the other.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from hsi_caption.config import VerificationConfig
from hsi_caption.datatypes import KnowledgeEmbedding, OntologySemanticRepresentation, ReasonedKnowledge, \
    StructuredFacts, UncertaintyEstimate, VerifiedFacts
from hsi_caption.stage3_knowledge_engine.knowledge_graph import KnowledgeGraphBuilder
from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage8_verification.classifier_agreement import ClassifierAgreementVerifier
from hsi_caption.stage8_verification.confidence_fusion import VerificationConfidenceFusion
from hsi_caption.stage8_verification.evidence_retrieval import EvidenceRetriever
from hsi_caption.stage8_verification.kg_verification import KnowledgeGraphVerifier
from hsi_caption.stage8_verification.ontology_matching import OntologyVerificationMatcher
from hsi_caption.stage8_verification.rule_verification import RuleVerifier
from hsi_caption.stage8_verification.semantic_similarity import SemanticSimilarityVerifier

logger = logging.getLogger(__name__)


class FactVerificationEngine:
    """Facade combining all seven Stage 8 modules."""

    def __init__(self, kg: KnowledgeGraphBuilder, ontology: OntologyLoader, cfg: VerificationConfig,
                 class_names: Optional[List[str]] = None) -> None:
        self.evidence_retriever = EvidenceRetriever()
        self.kg_verifier = KnowledgeGraphVerifier(kg)
        self.ontology_verifier = OntologyVerificationMatcher(ontology)
        self.rule_verifier = RuleVerifier()
        self.semantic_verifier = SemanticSimilarityVerifier()
        self.classifier_verifier = ClassifierAgreementVerifier(class_names or [])
        self.fusion = VerificationConfidenceFusion(
            cfg.semantic_similarity_weight, cfg.kg_weight, cfg.ontology_weight, cfg.rule_weight,
            cfg.verification_threshold, classifier_weight=cfg.classifier_weight,
        )

    def process(self, structured_facts: StructuredFacts, ke: KnowledgeEmbedding,
                rk: ReasonedKnowledge, osr: OntologySemanticRepresentation,
                uncertainty: Optional[UncertaintyEstimate] = None) -> VerifiedFacts:
        candidate_triplets = list(structured_facts.ranked_facts)

        # Ensure the trained classifier's own top prediction enters the
        # evidence pool even if Stage 3's spectral voting never proposed it.
        if uncertainty is not None and self.classifier_verifier.class_names:
            predicted = self.classifier_verifier.predicted_material_triplet(structured_facts.patch_id, uncertainty)
            already_present = any(
                t.predicate == predicted.predicate and t.object == predicted.object for t in candidate_triplets
            )
            if not already_present:
                candidate_triplets.append(predicted)

        scored = []
        for triplet in candidate_triplets:
            evidence = self.evidence_retriever.retrieve(triplet, ke, rk, osr)
            clf_score = self.classifier_verifier.verify(triplet, uncertainty) if uncertainty is not None else 0.5
            scored.append({
                "triplet": triplet,
                "sem": self.semantic_verifier.verify(evidence),
                "kg": self.kg_verifier.verify(evidence),
                "onto": self.ontology_verifier.verify(evidence),
                "rule": self.rule_verifier.verify(evidence, rk),
                "clf": clf_score,
            })
        return self.fusion.fuse(structured_facts.patch_id, scored)
