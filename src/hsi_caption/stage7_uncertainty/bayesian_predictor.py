"""
Stage 7 - Bayesian Predictor
===============================
Purpose
    Produce a class-probability distribution over `dataset.class_names`
    for a patch that supports *approximate Bayesian* uncertainty
    estimation via repeated stochastic forward passes (see MC Dropout,
    below), rather than a single point-estimate softmax.

Input
    Stage 3's `KnowledgeEmbedding.semantic_vector` (the knowledge-guided
    fused representation).

Algorithm
    A small MLP classifier head (Dense->Dropout->Dense->softmax) built
    on `nn_utils` primitives, with `Dropout.__call__(training=True)`
    left *active* at inference time. This operationalises MC Dropout
    (Gal & Ghahramani, 2016): dropout at test time turns a deterministic
    network into an approximation of a Bayesian neural network, where
    the *distribution* of predictions across stochastic passes
    approximates the posterior predictive distribution. Complexity:
    O(d_in * d_hidden + d_hidden * C) per forward pass.
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.nn_utils import Dense, Dropout, softmax

logger = logging.getLogger(__name__)


class BayesianPredictor:
    """MLP classifier head with test-time dropout for Bayesian approximation."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float, rng: np.random.Generator) -> None:
        hidden_dim = max(16, input_dim // 2)
        self.hidden = Dense(input_dim, hidden_dim, rng, activation=True)
        self.dropout = Dropout(dropout_rate, rng)
        self.output = Dense(hidden_dim, num_classes, rng, activation=False)

    def forward(self, semantic_vector: np.ndarray, stochastic: bool) -> np.ndarray:
        """Single forward pass -> class probability vector (C,)."""
        h = self.hidden(semantic_vector)
        h = self.dropout(h, training=stochastic)
        logits = self.output(h)
        return softmax(logits)
