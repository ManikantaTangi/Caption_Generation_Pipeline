"""
Stage 7 - Monte Carlo Dropout
================================
Purpose
    Drive the BayesianPredictor through `mc_dropout_passes` stochastic
    forward passes and collect the resulting distribution of class
    probability vectors -- the raw material from which epistemic
    uncertainty (model uncertainty, from disagreement *across* passes)
    is separated from aleatoric uncertainty (data uncertainty, from
    spread *within* each pass's distribution).

Input
    Stage 3's `KnowledgeEmbedding.semantic_vector` and the
    `BayesianPredictor` (this stage).

Algorithm
    Repeated stochastic forward passes with independent dropout masks
    (each call re-samples `Dropout`'s internal RNG). Returns the full
    (T, C) sample matrix for the EntropyCalculator / Calibration modules
    to consume, plus the sample mean as the point-estimate probability
    vector. Complexity: O(T * forward_pass_cost), T = mc_dropout_passes
    (default 30, matching common MC-Dropout practice of T in [20,100]).
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.stage7_uncertainty.bayesian_predictor import BayesianPredictor

logger = logging.getLogger(__name__)


class MCDropoutSampler:
    """Runs repeated stochastic forward passes for MC Dropout uncertainty."""

    def __init__(self, predictor: BayesianPredictor, num_passes: int) -> None:
        self.predictor = predictor
        self.num_passes = num_passes

    def sample(self, semantic_vector: np.ndarray) -> np.ndarray:
        """Returns (T, C) matrix of stochastic class-probability samples."""
        samples = np.stack([
            self.predictor.forward(semantic_vector, stochastic=True) for _ in range(self.num_passes)
        ], axis=0)
        return samples
