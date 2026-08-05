"""
Stage 2 - Feature Fusion
===========================
Purpose
    Combine the three modality embeddings (spectral, spatial, metadata)
    into a single `fused_vector` — the Unified Feature Embedding that is
    Stage 2's contractual output and the primary numeric input consumed
    by Stage 3 (material retrieval) and Stage 7 (uncertainty estimation).

Input
    Three vectors of possibly different dimensionality, produced in the
    same forward pass by SpectralEncoder, SpatialEncoder, MetadataEncoder.

Algorithm
    Three interchangeable strategies (selected via `fusion_strategy`):
      - "concat": simple concatenation + Dense projection. Cheapest,
        no cross-modal weighting; a solid baseline.
      - "sum": project each modality to the common `fused_embed_dim`
        then element-wise sum. Assumes modalities are equally salient.
      - "attention" (default): each modality vector is treated as a
        token; SingleHeadAttention lets the model learn *which*
        modality to weight more per-patch (e.g. down-weight spatial
        texture for a spectrally-unambiguous water patch), then the
        attended tokens are mean-pooled and projected. This is the
        method used in the reference results because per-patch
        modality salience genuinely varies in HSI scenes (a homogeneous
        water body needs little spatial evidence; a field-boundary
        patch needs a lot).

Output
    `FusedEmbedding` (Stage 2's contractual output) -> feeds directly
    into Stage 3 (KnowledgeRetrieval uses `fused_vector` for similarity
    search) and Stage 7 (BayesianPredictor uses it as classifier input).

Complexity
    O(d^2) for the attention/projection steps (d = embed dim, small).
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.datatypes import FusedEmbedding
from hsi_caption.nn_utils import Dense, SingleHeadAttention

logger = logging.getLogger(__name__)


class FeatureFusionError(Exception):
    """Raised for an unsupported fusion strategy."""


class FeatureFusion:
    """Fuses spectral, spatial, and metadata embeddings into one vector."""

    _SUPPORTED = ("concat", "sum", "attention")

    def __init__(self, spectral_dim: int, spatial_dim: int, metadata_dim: int,
                 fused_dim: int, strategy: str, rng: np.random.Generator) -> None:
        if strategy not in self._SUPPORTED:
            raise FeatureFusionError(f"Unsupported fusion strategy: {strategy}")
        self.strategy = strategy
        self.fused_dim = fused_dim

        if strategy == "concat":
            self.projector = Dense(spectral_dim + spatial_dim + metadata_dim, fused_dim, rng, activation=False)
        elif strategy == "sum":
            self.spec_proj = Dense(spectral_dim, fused_dim, rng, activation=False)
            self.spat_proj = Dense(spatial_dim, fused_dim, rng, activation=False)
            self.meta_proj = Dense(metadata_dim, fused_dim, rng, activation=False)
        else:  # attention
            self.spec_proj = Dense(spectral_dim, fused_dim, rng, activation=False)
            self.spat_proj = Dense(spatial_dim, fused_dim, rng, activation=False)
            self.meta_proj = Dense(metadata_dim, fused_dim, rng, activation=False)
            self.attention = SingleHeadAttention(fused_dim, rng)

    def fuse(self, patch_id: str, spectral_vec: np.ndarray, spatial_vec: np.ndarray,
              metadata_vec: np.ndarray) -> FusedEmbedding:
        attn_weights = None
        if self.strategy == "concat":
            fused = self.projector(np.concatenate([spectral_vec, spatial_vec, metadata_vec]))
        elif self.strategy == "sum":
            fused = self.spec_proj(spectral_vec) + self.spat_proj(spatial_vec) + self.meta_proj(metadata_vec)
        else:  # attention
            tokens = np.stack([
                self.spec_proj(spectral_vec), self.spat_proj(spatial_vec), self.meta_proj(metadata_vec),
            ], axis=0)  # (3, fused_dim)
            context, attn_weights = self.attention(tokens)
            fused = context.mean(axis=0)

        return FusedEmbedding(
            patch_id=patch_id, spectral_vector=spectral_vec, spatial_vector=spatial_vec,
            metadata_vector=metadata_vec, fused_vector=fused, attention_weights=attn_weights,
        )
