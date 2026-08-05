"""
Stage 7 - Calibration
========================
Purpose
    Correct systematic over/under-confidence in the raw MC-Dropout mean
    probabilities before they are reported as a "confidence score" --
    an uncalibrated softmax is well known to be overconfident (Guo et
    al. 2017), which would make Stage 9's confidence-banded captions
    misleading.

Input
    `class_probs_mean` (C,) from EntropyCalculator (this stage); a
    (probs, true_label) validation set for `fit()`.

Algorithm
    Temperature scaling (Guo et al. 2017), adapted to operate on
    already-softmaxed probabilities (since `BayesianPredictor` returns
    probabilities, not logits, keeping the nn_utils primitives simple):
        p_calibrated = normalize(p ** (1 / T))
    T is fit by grid search over a small candidate set minimizing
    negative log-likelihood on a held-out (probs, label) set -- the
    simplest calibration method that is still provably a strict
    generalisation of the uncalibrated case at T=1, and standard
    practice for calibrating Bayesian-approximate classifiers.
    Complexity: fit O(|grid| * N); apply O(C).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Calibration:
    """Temperature-scaling calibration for class probability vectors."""

    _GRID = np.round(np.arange(0.5, 3.05, 0.1), 2)

    def __init__(self, initial_temperature: float = 1.5) -> None:
        self.temperature = initial_temperature

    @staticmethod
    def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
        powered = np.clip(probs, 1e-12, 1.0) ** (1.0 / temperature)
        return powered / powered.sum()

    def apply(self, probs: np.ndarray) -> np.ndarray:
        return self._apply_temperature(probs, self.temperature)

    def fit(self, validation_set: List[Tuple[np.ndarray, int]]) -> float:
        """Grid-search the temperature minimizing NLL on a validation set.
        Falls back to the configured initial temperature if no validation
        data is available (e.g. a cold-start run with no labelled data)."""
        if not validation_set:
            logger.warning("No validation data for calibration; keeping T=%.2f", self.temperature)
            return self.temperature

        best_t, best_nll = self.temperature, float("inf")
        for t in self._GRID:
            nll = 0.0
            for probs, label in validation_set:
                calibrated = self._apply_temperature(probs, t)
                nll -= np.log(max(calibrated[label], 1e-12))
            nll /= len(validation_set)
            if nll < best_nll:
                best_nll, best_t = nll, float(t)

        logger.info("Calibration fit: T=%.2f (NLL=%.4f) over %d validation samples", best_t, best_nll,
                    len(validation_set))
        self.temperature = best_t
        return best_t
