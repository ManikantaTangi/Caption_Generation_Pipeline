"""
Stage 9 - Explanation Generator
==================================
Purpose
    Produce a natural-language justification for the caption -- Stage
    9's third contractual output alongside `caption` and
    `confidence_score` -- that describes the actual evidence observed
    at each stage (spectral similarity, material composition, trained
    model confidence, why a rule fired) rather than citing internal
    identifiers (rule IDs, raw fact counts). This fulfils the
    Explainable-AI requirement in a form a non-technical reader can
    actually use: a domain expert or thesis reviewer can trace exactly
    which observation led to which conclusion, in plain English.

Input
    `VerifiedFacts` (Stage 8), `UncertaintyEstimate` (Stage 7),
    `ReasonedKnowledge` (Stage 4), and optionally Stage 3's
    `KnowledgeEmbedding` (for the underlying spectral evidence) and the
    dataset's `class_names` (to name the trained classifier's top
    prediction from its raw probability vector).

Algorithm
    Ordered natural-language composition, each sentence grounded in a
    specific upstream value (never a fabricated claim):
      1. What the trained classifier (Stage 7) concluded, by name and
         confidence -- the pipeline's strongest, empirically-validated
         signal.
      2. What the spectral evidence (Stage 3) showed: the closest-matching
         reference material, its similarity score, and what fraction of
         the patch's pixels shared that match -- i.e. the literal
         features observed, not an internal rule ID.
      3. Any domain-reasoning context (Stage 4), phrased using the rule's
         own human-readable description (`fired_rule_descriptions`) --
         never a bare rule ID like "R3".
      4. A plain-language caveat if a hard domain constraint was violated.
      5. How many candidate facts were independently verified, and the
         overall verification score.
      6. Whether the remaining uncertainty is mostly model-driven
         (epistemic) or data-driven (aleatoric), and what that implies.
    Kept as templated natural-language composition (not a free-generation
    LLM call) so every sentence remains directly traceable to a specific
    upstream value -- the same auditability guarantee CaptionGenerator
    upholds. Complexity: O(F + R) for F facts, R fired rules.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from hsi_caption.datatypes import KnowledgeEmbedding, ReasonedKnowledge, UncertaintyEstimate, VerifiedFacts

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Builds a natural-language, evidence-grounded explanation for a caption."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names or []

    def _classifier_sentence(self, uncertainty: UncertaintyEstimate) -> Optional[str]:
        if not self.class_names or uncertainty.class_probs_mean.shape[0] != len(self.class_names):
            return None
        idx = int(np.argmax(uncertainty.class_probs_mean))
        predicted = self.class_names[idx]
        return (f"The trained classifier identified this patch as most consistent with {predicted} "
                f"({uncertainty.confidence_score*100:.0f}% confidence).")

    def _spectral_sentence(self, ke: KnowledgeEmbedding) -> Optional[str]:
        if not ke.material_matches:
            return None
        top = ke.material_matches[0]
        fraction = ke.material_fractions.get(top.material_name, 0.0)
        return (f"Its spectral signature most closely matched the {top.material_name} reference spectrum "
                f"({top.similarity*100:.0f}% similarity), and this material accounted for "
                f"{fraction*100:.0f}% of the patch's pixels.")

    def _reasoning_sentences(self, reasoned: ReasonedKnowledge) -> List[str]:
        sentences = []
        for desc in reasoned.fired_rule_descriptions:
            text = desc.strip()
            if text and not text.endswith((".", "!", "?")):
                text += "."
            if text:
                sentences.append(text)
        return sentences

    def _constraint_sentence(self, reasoned: ReasonedKnowledge) -> Optional[str]:
        if not reasoned.constraint_violations:
            return None
        details = []
        for v in reasoned.constraint_violations:
            details.append(v.split(":", 1)[1].strip() if ":" in v else v)
        return ("This conclusion should be treated cautiously: " + "; ".join(details) + ".")

    def _verification_sentence(self, verified_facts: VerifiedFacts) -> str:
        total = len(verified_facts.verified) + len(verified_facts.rejected)
        if total == 0:
            return f"No candidate facts were available to verify for this patch."
        return (f"{len(verified_facts.verified)} of {total} candidate facts were independently verified "
                f"against the knowledge graph, ontology, and reasoning rules "
                f"(verification score {verified_facts.verification_score:.2f}).")

    def _uncertainty_sentence(self, uncertainty: UncertaintyEstimate) -> str:
        if uncertainty.epistemic_uncertainty > uncertainty.aleatoric_uncertainty:
            return ("The remaining uncertainty is mostly model-driven, suggesting more training examples "
                    "like this one could sharpen future predictions.")
        return ("The remaining uncertainty is mostly data-driven, meaning the spectral and spatial signal "
                "itself carries some natural ambiguity for this patch.")

    def generate(self, verified_facts: VerifiedFacts, uncertainty: UncertaintyEstimate,
                 reasoned: ReasonedKnowledge, knowledge_embedding: Optional[KnowledgeEmbedding] = None) -> str:
        sentences: List[str] = []

        classifier_sentence = self._classifier_sentence(uncertainty)
        if classifier_sentence:
            sentences.append(classifier_sentence)

        if knowledge_embedding is not None:
            spectral_sentence = self._spectral_sentence(knowledge_embedding)
            if spectral_sentence:
                sentences.append(spectral_sentence)

        sentences.extend(self._reasoning_sentences(reasoned))

        constraint_sentence = self._constraint_sentence(reasoned)
        if constraint_sentence:
            sentences.append(constraint_sentence)

        sentences.append(self._verification_sentence(verified_facts))
        sentences.append(self._uncertainty_sentence(uncertainty))

        return " ".join(sentences)
