"""
Trainable PyTorch modules for Stage 2 (encoder) and Stage 7 (Bayesian
classifier head).

This module is the production upgrade path documented in
`docs/stage_reports/stage2_report.md` / `stage7_report.md`: the exact
same architectural shapes as `nn_utils.py` (multi-scale spectral CNN,
tokenized spatial self-attention, MLP metadata branch, attention
fusion, dropout-based Bayesian classifier head), now with real
autograd/backprop via `torch.nn`.

Two training phases use these modules:
  Phase A (`demos/train_encoder.py`):    EncoderNet trained end-to-end
                                          on raw patches -> class logits.
  Phase B (`demos/train_uncertainty_head.py`): BayesianHead trained on
                                          Stage 3's knowledge-guided
                                          semantic_vector (built from the
                                          frozen, Phase-A-trained encoder).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBranch(nn.Module):
    """Multi-scale 1D-CNN over the spectral axis. Mirrors nn_utils.SpectralEncoder."""

    def __init__(self, num_bands: int, embed_dim: int, kernels: List[int]) -> None:
        super().__init__()
        hidden_channels = 8
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=k, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            for k in kernels
        ])
        self.proj = nn.Linear(hidden_channels * len(kernels), embed_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """spectrum: (batch, num_bands) -> (batch, embed_dim)."""
        x = spectrum.unsqueeze(1)  # (batch, 1, num_bands)
        feats = [branch(x).squeeze(-1) for branch in self.branches]  # each (batch, hidden_channels)
        return self.proj(torch.cat(feats, dim=1))


class SpatialBranch(nn.Module):
    """Tokenized self-attention over patch sub-blocks. Mirrors nn_utils.SpatialEncoder."""

    def __init__(self, num_bands: int, embed_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.grid = int(round(num_tokens ** 0.5))
        if self.grid * self.grid != num_tokens:
            raise ValueError("num_tokens must be a perfect square.")
        token_dim = 16
        self.token_proj = nn.Linear(num_bands, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=1, batch_first=True)
        self.out_proj = nn.Linear(token_dim, embed_dim)

    def _tokenize(self, patch: torch.Tensor) -> torch.Tensor:
        """patch: (batch, P, P, B) -> tokens: (batch, grid*grid, B) via block-mean pooling."""
        batch, p, _, b = patch.shape
        edges = torch.linspace(0, p, self.grid + 1).long()
        tokens = []
        for i in range(self.grid):
            for j in range(self.grid):
                r0, r1 = edges[i].item(), max(edges[i + 1].item(), edges[i].item() + 1)
                c0, c1 = edges[j].item(), max(edges[j + 1].item(), edges[j].item() + 1)
                block = patch[:, r0:r1, c0:c1, :].reshape(batch, -1, b).mean(dim=1)
                tokens.append(block)
        return torch.stack(tokens, dim=1)  # (batch, grid*grid, B)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        tokens = self._tokenize(patch)                 # (batch, T, B)
        token_embeds = F.relu(self.token_proj(tokens))  # (batch, T, token_dim)
        context, _ = self.attn(token_embeds, token_embeds, token_embeds)
        pooled = context.mean(dim=1)                     # (batch, token_dim)
        return self.out_proj(pooled)


class MetadataBranch(nn.Module):
    """2-layer MLP over scalar metadata. Mirrors nn_utils.MetadataEncoder."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, embed_dim // 2)
        self.net = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class AttentionFusion(nn.Module):
    """Attention-based fusion of the three modality embeddings. Mirrors nn_utils fusion."""

    def __init__(self, spectral_dim: int, spatial_dim: int, metadata_dim: int, fused_dim: int) -> None:
        super().__init__()
        self.spec_proj = nn.Linear(spectral_dim, fused_dim)
        self.spat_proj = nn.Linear(spatial_dim, fused_dim)
        self.meta_proj = nn.Linear(metadata_dim, fused_dim)
        self.attn = nn.MultiheadAttention(fused_dim, num_heads=1, batch_first=True)

    def forward(self, spec: torch.Tensor, spat: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.spec_proj(spec), self.spat_proj(spat), self.meta_proj(meta)], dim=1)
        context, _ = self.attn(tokens, tokens, tokens)
        return context.mean(dim=1)


class EncoderNet(nn.Module):
    """Phase A: full Stage-2 encoder + a classification head for supervised training.

    `forward()` returns (fused_vector, logits); Phase A trains against
    `logits`, and after training `fused_vector` is the real (learned)
    Stage-2 output every downstream stage consumes.
    """

    def __init__(self, num_bands: int, num_classes: int, spectral_dim: int = 64, spatial_dim: int = 64,
                 metadata_dim: int = 16, fused_dim: int = 128, kernels: List[int] = (7, 5, 3),
                 spatial_tokens: int = 9) -> None:
        super().__init__()
        self.spectral = SpectralBranch(num_bands, spectral_dim, list(kernels))
        self.spatial = SpatialBranch(num_bands, spatial_dim, spatial_tokens)
        self.metadata = MetadataBranch(metadata_dim)
        self.fusion = AttentionFusion(spectral_dim, spatial_dim, metadata_dim, fused_dim)
        self.classifier = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Dropout(0.2),
                                         nn.Linear(fused_dim, num_classes))

    def forward(self, center_spectrum: torch.Tensor, full_patch: torch.Tensor, metadata: torch.Tensor):
        spec_vec = self.spectral(center_spectrum)
        spat_vec = self.spatial(full_patch)
        meta_vec = self.metadata(metadata)
        fused = self.fusion(spec_vec, spat_vec, meta_vec)
        logits = self.classifier(fused)
        return fused, logits


class BayesianHead(nn.Module):
    """Phase B: Stage-7's trainable classifier head, matching nn_utils.BayesianPredictor's
    architecture (Dense -> Dropout -> Dense) but with real backprop. Dropout is kept
    active at inference time (`model.train()`) for MC-Dropout sampling in Stage 7."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3) -> None:
        super().__init__()
        hidden_dim = max(16, input_dim // 2)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.hidden(x))
        h = self.dropout(h)
        return self.output(h)  # logits; caller applies softmax
