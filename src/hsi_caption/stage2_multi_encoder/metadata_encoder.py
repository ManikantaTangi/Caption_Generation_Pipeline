"""
Stage 2 - Metadata Encoder
=============================
Purpose
    Embed non-spectral, non-raw-spatial side information (patch position
    within the scene, local label purity as a homogeneity proxy) so
    Stage 9 captions can reference context like "near a field boundary"
    or "well within a homogeneous crop region" and Stage 4's context
    rules can condition on it.

Input
    `Patch.metadata` dict (`row_norm`, `col_norm`, `local_class_purity`)
    from Stage 1's PatchGenerator output.

Algorithm
    Small 2-layer MLP (Dense -> ReLU -> Dense) over the fixed-order
    metadata feature vector. A full transformer/attention mechanism is
    unnecessary for 3 scalar features; an MLP is the minimum-complexity
    architecture that still lets the fusion stage learn non-linear
    interactions between position and purity.

Complexity
    O(1) per patch (fixed small input/hidden sizes).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from hsi_caption.nn_utils import Dense

logger = logging.getLogger(__name__)

METADATA_FEATURE_ORDER: List[str] = ["row_norm", "col_norm", "local_class_purity"]


class MetadataEncoder:
    """2-layer MLP encoder for scalar patch metadata."""

    def __init__(self, embed_dim: int, rng: np.random.Generator) -> None:
        hidden_dim = max(8, embed_dim // 2)
        self.hidden = Dense(len(METADATA_FEATURE_ORDER), hidden_dim, rng, activation=True)
        self.out = Dense(hidden_dim, embed_dim, rng, activation=False)

    def encode(self, metadata: Dict[str, float]) -> np.ndarray:
        vec = np.array([metadata.get(k, 0.0) for k in METADATA_FEATURE_ORDER], dtype=np.float32)
        return self.out(self.hidden(vec))
