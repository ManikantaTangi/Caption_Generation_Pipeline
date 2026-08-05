import numpy as np
import pytest

from hsi_caption.stage2_multi_encoder.multi_encoder import MultiEncoder
from hsi_caption.stage2_multi_encoder.feature_fusion import FeatureFusion, FeatureFusionError
from hsi_caption.stage2_multi_encoder.metadata_encoder import MetadataEncoder
from hsi_caption.stage2_multi_encoder.spatial_encoder import SpatialEncoder
from hsi_caption.stage2_multi_encoder.spectral_encoder import SpectralEncoder


class TestSpectralEncoder:
    def test_output_dim(self, cfg):
        rng = np.random.default_rng(0)
        enc = SpectralEncoder(cfg.dataset.num_bands, cfg.encoder.spectral_embed_dim,
                               cfg.encoder.spectral_conv_kernels, rng)
        out = enc.encode(np.random.rand(cfg.dataset.num_bands).astype(np.float32))
        assert out.shape == (cfg.encoder.spectral_embed_dim,)

    def test_wrong_length_raises(self, cfg):
        rng = np.random.default_rng(0)
        enc = SpectralEncoder(cfg.dataset.num_bands, cfg.encoder.spectral_embed_dim,
                               cfg.encoder.spectral_conv_kernels, rng)
        with pytest.raises(ValueError):
            enc.encode(np.random.rand(5).astype(np.float32))


class TestSpatialEncoder:
    def test_output_dim_and_attention_shape(self, cfg):
        rng = np.random.default_rng(0)
        enc = SpatialEncoder(cfg.dataset.num_bands, cfg.encoder.spatial_embed_dim,
                              cfg.encoder.spatial_patch_tokens, rng)
        patch = np.random.rand(cfg.patch.patch_size, cfg.patch.patch_size, cfg.dataset.num_bands).astype(np.float32)
        embedding, attn = enc.encode(patch)
        assert embedding.shape == (cfg.encoder.spatial_embed_dim,)
        assert attn.shape == (cfg.encoder.spatial_patch_tokens, cfg.encoder.spatial_patch_tokens)

    def test_non_square_tokens_raises(self, cfg):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            SpatialEncoder(cfg.dataset.num_bands, cfg.encoder.spatial_embed_dim, 7, rng)


class TestMetadataEncoder:
    def test_output_dim(self, cfg):
        rng = np.random.default_rng(0)
        enc = MetadataEncoder(cfg.encoder.metadata_embed_dim, rng)
        out = enc.encode({"row_norm": 0.5, "col_norm": 0.5, "local_class_purity": 1.0})
        assert out.shape == (cfg.encoder.metadata_embed_dim,)

    def test_missing_keys_defaults_to_zero(self, cfg):
        rng = np.random.default_rng(0)
        enc = MetadataEncoder(cfg.encoder.metadata_embed_dim, rng)
        out = enc.encode({})
        assert out.shape == (cfg.encoder.metadata_embed_dim,)


class TestFeatureFusion:
    @pytest.mark.parametrize("strategy", ["concat", "sum", "attention"])
    def test_all_strategies_produce_correct_shape(self, cfg, strategy):
        rng = np.random.default_rng(0)
        fusion = FeatureFusion(cfg.encoder.spectral_embed_dim, cfg.encoder.spatial_embed_dim,
                                cfg.encoder.metadata_embed_dim, cfg.encoder.fused_embed_dim, strategy, rng)
        fused = fusion.fuse(
            "p1", np.random.rand(cfg.encoder.spectral_embed_dim).astype(np.float32),
            np.random.rand(cfg.encoder.spatial_embed_dim).astype(np.float32),
            np.random.rand(cfg.encoder.metadata_embed_dim).astype(np.float32),
        )
        assert fused.fused_vector.shape == (cfg.encoder.fused_embed_dim,)

    def test_unsupported_strategy_raises(self, cfg):
        rng = np.random.default_rng(0)
        with pytest.raises(FeatureFusionError):
            FeatureFusion(4, 4, 4, 4, "unsupported", rng)


class TestMultiEncoderFacade:
    def test_encode_patch_end_to_end(self, cfg, sample_patch):
        encoder = MultiEncoder(cfg.dataset.num_bands, cfg.encoder, cfg.patch.random_seed)
        emb = encoder.encode(sample_patch)
        assert emb.fused_vector.shape == (cfg.encoder.fused_embed_dim,)
        assert emb.patch_id == sample_patch.patch_id
