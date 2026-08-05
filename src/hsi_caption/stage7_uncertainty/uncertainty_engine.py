"""
Stage 7 - Uncertainty Estimation (facade)
=============================================
Wires BayesianPredictor + MCDropoutSampler + EntropyCalculator +
Calibration + ConfidenceFusion into one callable consuming Stage 3's
semantic vector and producing Stage 7's `UncertaintyEstimate`.
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.config import UncertaintyConfig
from hsi_caption.datatypes import UncertaintyEstimate
from hsi_caption.stage7_uncertainty.bayesian_predictor import BayesianPredictor
from hsi_caption.stage7_uncertainty.calibration import Calibration
from hsi_caption.stage7_uncertainty.confidence_fusion import ConfidenceFusion
from hsi_caption.stage7_uncertainty.entropy_calculator import EntropyCalculator
from hsi_caption.stage7_uncertainty.mc_dropout import MCDropoutSampler

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """Facade combining all five Stage 7 modules."""

    def __init__(self, semantic_vector_dim: int, num_classes: int, cfg: UncertaintyConfig,
                 dropout_rate: float, random_seed: int = 42) -> None:
        rng = np.random.default_rng(random_seed)
        self.predictor = BayesianPredictor(semantic_vector_dim, num_classes, dropout_rate, rng)
        self.sampler = MCDropoutSampler(self.predictor, cfg.mc_dropout_passes)
        self.entropy_calc = EntropyCalculator(cfg.entropy_normalize)
        self.calibration = Calibration(cfg.temperature)
        self.confidence_fusion = ConfidenceFusion()

    def fit_calibration(self, validation_pairs) -> float:
        """validation_pairs: List[(semantic_vector, true_label_idx)]."""
        probs_labels = []
        for vec, label in validation_pairs:
            samples = self.sampler.sample(vec)
            decomposition = self.entropy_calc.decompose(samples)
            probs_labels.append((decomposition.class_probs_mean, label))
        return self.calibration.fit(probs_labels)

    def process(self, patch_id: str, semantic_vector: np.ndarray, calibrated: bool = True) -> UncertaintyEstimate:
        samples = self.sampler.sample(semantic_vector)
        decomposition = self.entropy_calc.decompose(samples)
        if calibrated:
            decomposition.class_probs_mean = self.calibration.apply(decomposition.class_probs_mean)
        return self.confidence_fusion.fuse(patch_id, decomposition, calibrated)
