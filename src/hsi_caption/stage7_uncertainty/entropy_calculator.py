"""
Stage 7 - Entropy Calculator
===============================
Purpose
    Decompose the MC-Dropout sample matrix into predictive entropy and
    its two canonical sub-components -- epistemic (model) and aleatoric
    (data) uncertainty -- giving Stage 9's explanation generator a
    principled reason to say "the model is unsure which class" (high
    epistemic) vs. "the input itself is ambiguous" (high aleatoric).

Input
    `(T, C)` MC-Dropout sample matrix from MCDropoutSampler (this stage).

Algorithm
    Standard entropy decomposition (Depeweg et al. 2018; Kwon et al.
    2020, used widely in Bayesian deep learning for HSI/medical imaging):
        p_bar        = mean over T samples                (C,)
        H_predictive = H[p_bar] = -sum(p_bar * log(p_bar))          (total uncertainty)
        H_aleatoric  = E_t[H[p_t]] = mean_t( -sum(p_t*log(p_t)) )   (expected data noise)
        H_epistemic  = H_predictive - H_aleatoric                    (= mutual information,
                                                                        the BALD acquisition score)
    Complexity: O(T*C).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntropyDecomposition:
    class_probs_mean: np.ndarray
    predictive_entropy: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float


class EntropyCalculator:
    """Decomposes MC-Dropout samples into predictive/aleatoric/epistemic entropy."""

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    @staticmethod
    def _entropy(p: np.ndarray, axis: int) -> np.ndarray:
        p = np.clip(p, 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=axis)

    def decompose(self, samples: np.ndarray) -> EntropyDecomposition:
        p_bar = samples.mean(axis=0)  # (C,)
        h_predictive = float(self._entropy(p_bar, axis=0))
        per_sample_entropy = self._entropy(samples, axis=1)  # (T,)
        h_aleatoric = float(per_sample_entropy.mean())
        h_epistemic = max(0.0, h_predictive - h_aleatoric)

        if self.normalize:
            max_entropy = float(np.log(samples.shape[1]))  # log(C), uniform-distribution entropy
            h_predictive /= max_entropy
            h_aleatoric /= max_entropy
            h_epistemic /= max_entropy

        return EntropyDecomposition(
            class_probs_mean=p_bar, predictive_entropy=h_predictive,
            aleatoric_uncertainty=h_aleatoric, epistemic_uncertainty=h_epistemic,
        )
