"""
Stage 7 - Confidence Fusion
==============================
Purpose
    Combine the calibrated class-probability mean, predictive entropy,
    and the epistemic/aleatoric split into the single scalar
    `confidence_score` that every downstream stage (Stage 8 verification
    weighting, Stage 9 confidence-banded language) treats as *the*
    number to trust. This is Stage 7's contractual output,
    `UncertaintyEstimate`.

Input
    Calibrated `class_probs_mean`, `EntropyDecomposition` (this stage).

Algorithm
    confidence_score = max_class_prob * (1 - predictive_entropy_normalized)
    Multiplying the top-class probability by (1 - normalized entropy)
    penalises cases where the top class is only narrowly ahead of
    competitors (high entropy despite a locally-large max probability),
    which a max-probability-only confidence score would miss. Bounded
    to [0, 1] by construction (both factors are in [0, 1] given
    `entropy_normalize=True` upstream). Complexity: O(C).
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.datatypes import UncertaintyEstimate
from hsi_caption.stage7_uncertainty.entropy_calculator import EntropyDecomposition

logger = logging.getLogger(__name__)


class ConfidenceFusion:
    """Fuses calibrated probabilities and entropy decomposition into one score."""

    def fuse(self, patch_id: str, decomposition: EntropyDecomposition, calibrated: bool) -> UncertaintyEstimate:
        max_prob = float(decomposition.class_probs_mean.max())
        confidence_score = float(np.clip(max_prob * (1.0 - decomposition.predictive_entropy), 0.0, 1.0))

        return UncertaintyEstimate(
            patch_id=patch_id,
            class_probs_mean=decomposition.class_probs_mean,
            epistemic_uncertainty=decomposition.epistemic_uncertainty,
            aleatoric_uncertainty=decomposition.aleatoric_uncertainty,
            predictive_entropy=decomposition.predictive_entropy,
            confidence_score=confidence_score,
            calibrated=calibrated,
        )
