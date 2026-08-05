"""
Stage 2 - Multi Encoder (facade)
===================================
Wires SpectralEncoder + SpatialEncoder + MetadataEncoder + FeatureFusion
into a single callable that takes one Stage-1 `Patch` and returns the
Stage-2 `FusedEmbedding`. This is the object `pipeline.py` and Stage 3
actually depend on; the four sub-modules above remain independently
unit-testable.
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.config import EncoderConfig
from hsi_caption.datatypes import FusedEmbedding, Patch
from hsi_caption.stage2_multi_encoder.feature_fusion import FeatureFusion
from hsi_caption.stage2_multi_encoder.metadata_encoder import MetadataEncoder
from hsi_caption.stage2_multi_encoder.spatial_encoder import SpatialEncoder
from hsi_caption.stage2_multi_encoder.spectral_encoder import SpectralEncoder

logger = logging.getLogger(__name__)


class MultiEncoder:
    """Facade combining all four Stage 2 modules."""

    def __init__(self, num_bands: int, cfg: EncoderConfig, random_seed: int = 42) -> None:
        rng = np.random.default_rng(random_seed)
        self.spectral_encoder = SpectralEncoder(num_bands, cfg.spectral_embed_dim, cfg.spectral_conv_kernels, rng)
        self.spatial_encoder = SpatialEncoder(num_bands, cfg.spatial_embed_dim, cfg.spatial_patch_tokens, rng)
        self.metadata_encoder = MetadataEncoder(cfg.metadata_embed_dim, rng)
        self.feature_fusion = FeatureFusion(
            cfg.spectral_embed_dim, cfg.spatial_embed_dim, cfg.metadata_embed_dim,
            cfg.fused_embed_dim, cfg.fusion_strategy, rng,
        )

    def encode(self, patch: Patch) -> FusedEmbedding:
        p = patch.cube_data.shape[0] // 2
        center_spectrum = patch.cube_data[p, p, :]
        spectral_vec = self.spectral_encoder.encode(center_spectrum)
        spatial_vec, _ = self.spatial_encoder.encode(patch.cube_data)
        metadata_vec = self.metadata_encoder.encode(patch.metadata)
        return self.feature_fusion.fuse(patch.patch_id, spectral_vec, spatial_vec, metadata_vec)
