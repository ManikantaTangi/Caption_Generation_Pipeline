"""
Stage 2 - Spectral Encoder
============================
Purpose
    Extract a fixed-length embedding that captures the *spectral shape*
    (absorption features, red-edge slope, NIR plateau) of a patch's
    centre-pixel spectrum, independent of spatial context. This is the
    signal Stage 3's material identification and spectral-library
    retrieval depend on most directly.

Input
    `Patch.cube_data[center, center, :]` — the (B,) spectrum at the
    patch centre — from Stage 1's PatchGenerator output.

Algorithm
    Multi-scale 1D-CNN: three parallel Conv1D branches with kernel sizes
    {7, 5, 3} (configurable) applied to the raw spectrum, each followed
    by global average pooling, concatenated, then projected by a Dense
    layer to `spectral_embed_dim`. Multi-scale kernels let the network
    respond to both broad absorption bands (large kernel) and narrow
    diagnostic features (small kernel) simultaneously — standard
    practice in HSI spectral-CNN literature (e.g. Hu et al. 2015-style
    1D-CNN spectral classifiers), reused here as a general-purpose
    feature extractor rather than a classifier head.

Complexity
    O(B * K_max * C) per branch; O(B) overall for typical B~270, small K.
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.nn_utils import Conv1D, Dense

logger = logging.getLogger(__name__)


class SpectralEncoder:
    """Multi-scale 1D-CNN spectral feature extractor."""

    def __init__(self, num_bands: int, embed_dim: int, kernels: list, rng: np.random.Generator) -> None:
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        hidden_channels = 8
        self.branches = [Conv1D(1, hidden_channels, k, rng, stride=2) for k in kernels]
        self.projector = Dense(hidden_channels * len(kernels), embed_dim, rng, activation=False)

    def encode(self, spectrum: np.ndarray) -> np.ndarray:
        """spectrum: (num_bands,) -> (embed_dim,)."""
        if spectrum.shape[0] != self.num_bands:
            raise ValueError(f"Expected spectrum length {self.num_bands}, got {spectrum.shape[0]}")
        x = spectrum.reshape(-1, 1).astype(np.float32)  # (B, 1) treat as single-channel signal
        pooled_features = []
        for branch in self.branches:
            feat_map = branch(x)               # (L', hidden_channels)
            pooled_features.append(feat_map.mean(axis=0))  # global average pool -> (hidden_channels,)
        concat = np.concatenate(pooled_features, axis=0)
        embedding = self.projector(concat)
        return embedding
