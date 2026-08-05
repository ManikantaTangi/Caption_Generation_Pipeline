"""
Stage 7 - Uncertainty Estimation (trained PyTorch backend)
===============================================================
Drop-in replacement for `UncertaintyEngine` (same `.process(patch_id,
semantic_vector) -> UncertaintyEstimate` interface) backed by a
`torch_modules.BayesianHead` trained by `demos/train_uncertainty_head.py`.
MC-Dropout sampling keeps the model in `train()` mode (dropout active)
for the T stochastic forward passes, exactly mirroring the numpy
reference's `Dropout(training=True)` behaviour, but now with real
learned weights.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from hsi_caption.config import UncertaintyConfig
from hsi_caption.datatypes import UncertaintyEstimate
from hsi_caption.stage7_uncertainty.calibration import Calibration
from hsi_caption.stage7_uncertainty.confidence_fusion import ConfidenceFusion
from hsi_caption.stage7_uncertainty.entropy_calculator import EntropyCalculator
from hsi_caption.torch_modules import BayesianHead

logger = logging.getLogger(__name__)


class TorchUncertaintyEngine:
    """Loads a trained BayesianHead and runs MC-Dropout uncertainty estimation."""

    def __init__(self, weights_path: str, cfg: UncertaintyConfig, device: str = "cpu",
                 calibration_temperature: float = None) -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model = BayesianHead(checkpoint["semantic_dim"], checkpoint["num_classes"],
                                   checkpoint["dropout"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.num_passes = cfg.mc_dropout_passes
        self.entropy_calc = EntropyCalculator(cfg.entropy_normalize)
        temp = calibration_temperature if calibration_temperature is not None else cfg.temperature
        self.calibration = Calibration(temp)
        self.confidence_fusion = ConfidenceFusion()
        logger.info("Loaded trained Bayesian head from %s (semantic_dim=%d)",
                    weights_path, checkpoint["semantic_dim"])

    def process(self, patch_id: str, semantic_vector: np.ndarray, calibrated: bool = True) -> UncertaintyEstimate:
        x = torch.from_numpy(semantic_vector.astype(np.float32)).unsqueeze(0).to(self.device)

        self.model.train()  # keep dropout ACTIVE for MC sampling (not a training step -- no backward/optimizer)
        samples = []
        with torch.no_grad():
            for _ in range(self.num_passes):
                probs = F.softmax(self.model(x), dim=1).squeeze(0).cpu().numpy()
                samples.append(probs)
        self.model.eval()
        samples = np.stack(samples, axis=0)

        decomposition = self.entropy_calc.decompose(samples)
        if calibrated:
            decomposition.class_probs_mean = self.calibration.apply(decomposition.class_probs_mean)
        return self.confidence_fusion.fuse(patch_id, decomposition, calibrated)
