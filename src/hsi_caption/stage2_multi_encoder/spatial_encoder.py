"""
Stage 2 - Spatial Encoder
============================
Purpose
    Extract a fixed-length embedding of the patch's *spatial texture and
    homogeneity* (field-boundary vs. homogeneous crop interior, road
    edges, etc.), which the spectral-only encoder cannot see.

Input
    `Patch.cube_data` — the full (patch_size, patch_size, B) cube — from
    Stage 1's PatchGenerator output.

Algorithm
    Vision-Transformer-style tokenization: the patch is split into
    `spatial_patch_tokens` (default 9, i.e. a 3x3 grid of sub-blocks);
    each sub-block's mean band-averaged reflectance vector (reduced to a
    small per-token descriptor) is treated as a token. A single-head
    self-attention block lets tokens exchange context (a homogeneous
    interior token vs. a boundary token attend differently), then tokens
    are mean-pooled and projected to `spatial_embed_dim`. This mirrors
    the ViT patch-embedding + self-attention pattern (Dosovitskiy et al.
    2021) at a scale appropriate for small (15x15) HSI patches, where a
    full multi-layer ViT would overfit/be unnecessary.

Complexity
    O(T^2 * d) for self-attention over T tokens of dim d (T is small,
    9 by default, so this is negligible compared to the spectral branch).
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.nn_utils import Dense, SingleHeadAttention

logger = logging.getLogger(__name__)


class SpatialEncoder:
    """ViT-style tokenized self-attention spatial feature extractor."""

    def __init__(self, num_bands: int, embed_dim: int, num_tokens: int, rng: np.random.Generator) -> None:
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.grid = int(round(np.sqrt(num_tokens)))
        if self.grid * self.grid != num_tokens:
            raise ValueError("spatial_patch_tokens must be a perfect square (e.g. 9, 16).")
        token_dim = 16
        self.token_projector = Dense(num_bands, token_dim, rng, activation=True)
        self.attention = SingleHeadAttention(token_dim, rng)
        self.out_projector = Dense(token_dim, embed_dim, rng, activation=False)

    def _tokenize(self, cube_patch: np.ndarray) -> np.ndarray:
        """cube_patch: (P, P, B) -> tokens: (grid*grid, B) via block-mean pooling."""
        p = cube_patch.shape[0]
        edges = np.linspace(0, p, self.grid + 1).astype(int)
        tokens = []
        for i in range(self.grid):
            for j in range(self.grid):
                block = cube_patch[edges[i]:edges[i + 1], edges[j]:edges[j + 1], :]
                if block.size == 0:
                    block = cube_patch[max(0, edges[i] - 1):edges[i] + 1, max(0, edges[j] - 1):edges[j] + 1, :]
                tokens.append(block.reshape(-1, cube_patch.shape[-1]).mean(axis=0))
        return np.stack(tokens, axis=0)  # (grid*grid, B)

    def encode(self, cube_patch: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """cube_patch: (P, P, B) -> (embedding: (embed_dim,), attn_weights: (T, T))."""
        tokens = self._tokenize(cube_patch)                  # (T, B)
        token_embeds = self.token_projector(tokens)           # (T, token_dim)
        context, attn_weights = self.attention(token_embeds)  # (T, token_dim)
        pooled = context.mean(axis=0)                          # (token_dim,)
        embedding = self.out_projector(pooled)
        return embedding, attn_weights
