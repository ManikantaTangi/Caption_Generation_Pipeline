"""
Stage 8 - Confidence Fusion
==============================
Purpose
    Combine five independent verification scores (KG, ontology, rule,
    semantic-similarity, and Stage 7's trained classifier agreement)
    into one `verification_score` per triplet, and split all candidate
    triplets into `verified` vs. `rejected` sets -- Stage 8's
    contractual output, `VerifiedFacts`, which is the *only* fact
    source Stage 9 is allowed to caption from.

Input
    Per-triplet scores from KnowledgeGraphVerifier, OntologyVerification
    Matcher, RuleVerifier, SemanticSimilarityVerifier, and
    ClassifierAgreementVerifier (this stage).

Algorithm
    Weighted linear fusion (weights configurable, sum to 1 by default,
    validated at startup):
        score = w_sem*sem + w_kg*kg + w_onto*onto + w_rule*rule + w_clf*clf
    A linear combination of five independently-meaningful [0,1] scores
    is preferred over a learned re-ranker here for the same
    interpretability reason as Stage 6's ranking: every verification
    decision must be explainable ("this material was accepted mainly on
    classifier agreement, weakly on ontology") for the Explainable-AI
    requirement. `classifier_weight` is the largest single weight by
    default (0.30) since it is the pipeline's only signal validated
    against held-out labelled accuracy, but it never has sole authority
    -- a confidently-agreed-with-nothing-else fact can still be rejected
    if the other four signals strongly disagree.
    A triplet is `verified` iff score >= `verification_threshold`.
    Complexity: O(F) for F candidate triplets.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from hsi_caption.datatypes import Triplet, VerifiedFacts

logger = logging.getLogger(__name__)


class VerificationConfidenceFusion:
    """Fuses the five Stage-8 verification signals into a final score."""

    def __init__(self, sem_weight: float, kg_weight: float, onto_weight: float, rule_weight: float,
                 threshold: float, classifier_weight: float = 0.0) -> None:
        self.sem_weight = sem_weight
        self.kg_weight = kg_weight
        self.onto_weight = onto_weight
        self.rule_weight = rule_weight
        self.classifier_weight = classifier_weight
        self.threshold = threshold

    def fuse(self, patch_id: str, scored: List[Dict]) -> VerifiedFacts:
        """scored: list of dicts with keys {triplet, sem, kg, onto, rule, clf}.
        `clf` is optional per-entry (defaults to a neutral 0.5) so this
        stays backward compatible with call sites that don't run the
        classifier-agreement check (e.g. no trained model available)."""
        verified: List[Triplet] = []
        rejected: List[Triplet] = []
        fact_confidence: Dict[str, float] = {}

        for entry in scored:
            score = (
                self.sem_weight * entry["sem"] + self.kg_weight * entry["kg"]
                + self.onto_weight * entry["onto"] + self.rule_weight * entry["rule"]
                + self.classifier_weight * entry.get("clf", 0.5)
            )
            triplet: Triplet = entry["triplet"]
            key = f"{triplet.subject}|{triplet.predicate}|{triplet.object}"
            # if the same (s,p,o) appears twice (e.g. Stage 3 and the classifier
            # both proposed the same material), keep the higher-scoring entry
            if key in fact_confidence and fact_confidence[key] >= score:
                continue
            fact_confidence[key] = float(score)
            if score >= self.threshold:
                verified = [t for t in verified if f"{t.subject}|{t.predicate}|{t.object}" != key]
                verified.append(triplet)
            else:
                rejected = [t for t in rejected if f"{t.subject}|{t.predicate}|{t.object}" != key]
                rejected.append(triplet)

        verification_score = float(sum(fact_confidence.values()) / len(fact_confidence)) if fact_confidence else 0.0
        logger.debug("Verified %d/%d facts for %s (overall score=%.3f)",
                     len(verified), len(scored), patch_id, verification_score)
        return VerifiedFacts(
            patch_id=patch_id, verified=verified, rejected=rejected,
            verification_score=verification_score, fact_confidence=fact_confidence,
        )
