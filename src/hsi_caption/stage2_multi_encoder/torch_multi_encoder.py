"""
Stage 2 - Multi Encoder (trained PyTorch backend)
=====================================================
Drop-in replacement for `MultiEncoder` (same `.encode(patch) ->
FusedEmbedding` interface) backed by a `torch_modules.EncoderNet`
trained by `demos/train_encoder.py`, run in inference mode
(`model.eval()`, no dropout, no gradient tracking). This is what
`pipeline.py` uses instead of the numpy reference encoder once trained
weights are available -- see `HSICaptionPipeline.build_engines(...,
encoder_weights_path=...)`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from hsi_caption.datatypes import FusedEmbedding, Patch
from hsi_caption.torch_modules import EncoderNet

logger = logging.getLogger(__name__)


class TorchMultiEncoder:
    """Loads trained EncoderNet weights and exposes the MultiEncoder interface."""

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        self.model = EncoderNet(
            num_bands=cfg["num_bands"], num_classes=cfg["num_classes"], spectral_dim=cfg["spectral_dim"],
            spatial_dim=cfg["spatial_dim"], metadata_dim=cfg["metadata_dim"], fused_dim=cfg["fused_dim"],
            kernels=cfg["kernels"], spatial_tokens=cfg["spatial_tokens"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        logger.info("Loaded trained encoder weights from %s", weights_path)

    def encode(self, patch: Patch) -> FusedEmbedding:
        p = patch.cube_data.shape[0] // 2
        spectrum = torch.from_numpy(patch.cube_data[p, p, :].astype(np.float32)).unsqueeze(0).to(self.device)
        full_patch = torch.from_numpy(patch.cube_data.astype(np.float32)).unsqueeze(0).to(self.device)
        metadata = torch.tensor([[
            patch.metadata.get("row_norm", 0.0), patch.metadata.get("col_norm", 0.0),
            patch.metadata.get("local_class_purity", 0.0),
        ]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            fused, _ = self.model(spectrum, full_patch, metadata)

        fused_np = fused.squeeze(0).cpu().numpy()
        return FusedEmbedding(
            patch_id=patch.patch_id, spectral_vector=fused_np, spatial_vector=fused_np,
            metadata_vector=fused_np, fused_vector=fused_np,
        )
