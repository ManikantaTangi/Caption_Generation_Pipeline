"""
Stage 8 - Classifier Agreement
=================================
Purpose
    Score a `hasMaterial` triplet by how strongly Stage 7's *trained*
    classifier agrees with it -- the pipeline's most direct empirical
    check, since it reflects a model actually validated against held-out
    labelled data (see `demos/train_uncertainty_head.py`), rather than
    only the untrained symbolic/retrieval evidence Stages 3-5 provide.

    Before this module existed, Stage 9 chose which material to caption
    purely from Stage 3's raw per-pixel spectral-fraction vote, which
    never consulted the trained classifier at all -- a real architecture
    gap (a patch could be captioned with a material the trained model
    confidently disagreed with). This module closes that gap by folding
    Stage 7's prediction into Stage 8's fusion as a proper, weighted,
    independently-auditable signal alongside KG/ontology/rule/semantic
    evidence, rather than silently overriding anything.

Input
    A candidate `Triplet` and Stage 7's `UncertaintyEstimate.class_probs_mean`
    (the calibrated, MC-Dropout-averaged class probability vector).

Algorithm
    For a `hasMaterial` triplet whose object names a known class, the
    score is simply that class's calibrated predicted probability:
        score = class_probs_mean[class_names.index(triplet.object)]
    This is a full probabilistic agreement score (not just a 0/1 match
    against the argmax prediction), so it naturally rewards *any*
    candidate material the trained classifier considers plausible, and
    penalises ones it considers unlikely -- including candidates Stage 3
    proposed on its own. Non-material predicates (hasContext, hasConcept,
    coOccursWith) get a neutral score, since the classifier has no
    opinion on those. Complexity: O(1) per triplet (array index).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from hsi_caption.datatypes import Triplet, UncertaintyEstimate

logger = logging.getLogger(__name__)


class ClassifierAgreementVerifier:
    """Scores hasMaterial triplets by the trained classifier's own probability for that class."""

    NEUTRAL_SCORE = 0.5

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names

    def verify(self, triplet: Triplet, uncertainty: UncertaintyEstimate) -> float:
        if triplet.predicate != "hasMaterial" or triplet.object not in self.class_names:
            return self.NEUTRAL_SCORE
        idx = self.class_names.index(triplet.object)
        if idx >= uncertainty.class_probs_mean.shape[0]:
            return self.NEUTRAL_SCORE
        return float(np.clip(uncertainty.class_probs_mean[idx], 0.0, 1.0))

    def predicted_material_triplet(self, patch_id: str, uncertainty: UncertaintyEstimate) -> Triplet:
        """Builds a hasMaterial candidate from the classifier's own top prediction,
        so it enters Stage 8's evidence pool even if Stage 3's spectral voting
        never proposed that material on its own."""
        idx = int(np.argmax(uncertainty.class_probs_mean))
        name = self.class_names[idx] if idx < len(self.class_names) else "unknown"
        return Triplet(subject=patch_id, predicate="hasMaterial", object=name,
                        score=uncertainty.confidence_score)
