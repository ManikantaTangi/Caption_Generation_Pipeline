import numpy as np
import pytest

from hsi_caption.stage7_uncertainty.bayesian_predictor import BayesianPredictor
from hsi_caption.stage7_uncertainty.calibration import Calibration
from hsi_caption.stage7_uncertainty.confidence_fusion import ConfidenceFusion
from hsi_caption.stage7_uncertainty.entropy_calculator import EntropyCalculator
from hsi_caption.stage7_uncertainty.mc_dropout import MCDropoutSampler
from hsi_caption.stage7_uncertainty.uncertainty_engine import UncertaintyEngine


class TestBayesianPredictor:
    def test_output_is_valid_probability_distribution(self):
        rng = np.random.default_rng(0)
        predictor = BayesianPredictor(input_dim=32, num_classes=5, dropout_rate=0.2, rng=rng)
        probs = predictor.forward(np.random.rand(32).astype(np.float32), stochastic=False)
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all()


class TestMCDropoutSampler:
    def test_sample_shape(self):
        rng = np.random.default_rng(0)
        predictor = BayesianPredictor(input_dim=16, num_classes=4, dropout_rate=0.3, rng=rng)
        sampler = MCDropoutSampler(predictor, num_passes=10)
        samples = sampler.sample(np.random.rand(16).astype(np.float32))
        assert samples.shape == (10, 4)

    def test_stochastic_passes_vary(self):
        rng = np.random.default_rng(0)
        predictor = BayesianPredictor(input_dim=16, num_classes=4, dropout_rate=0.5, rng=rng)
        sampler = MCDropoutSampler(predictor, num_passes=20)
        samples = sampler.sample(np.random.rand(16).astype(np.float32))
        assert samples.std(axis=0).sum() > 0  # dropout should introduce variation


class TestEntropyCalculator:
    def test_zero_entropy_for_certain_prediction(self):
        calc = EntropyCalculator(normalize=False)
        samples = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
        decomposition = calc.decompose(samples)
        assert decomposition.predictive_entropy == pytest.approx(0.0, abs=1e-6)

    def test_epistemic_nonnegative(self):
        calc = EntropyCalculator(normalize=True)
        rng = np.random.default_rng(1)
        samples = rng.dirichlet(alpha=[1, 1, 1], size=15)
        decomposition = calc.decompose(samples)
        assert decomposition.epistemic_uncertainty >= 0.0


class TestCalibration:
    def test_temperature_one_is_identity(self):
        cal = Calibration(initial_temperature=1.0)
        probs = np.array([0.7, 0.2, 0.1])
        calibrated = cal.apply(probs)
        assert np.allclose(calibrated, probs, atol=1e-6)

    def test_fit_returns_temperature_from_grid(self):
        cal = Calibration()
        val_set = [(np.array([0.9, 0.1]), 0), (np.array([0.6, 0.4]), 1)]
        t = cal.fit(val_set)
        assert t in cal._GRID

    def test_fit_with_no_data_keeps_default(self):
        cal = Calibration(initial_temperature=1.5)
        t = cal.fit([])
        assert t == 1.5


class TestConfidenceFusion:
    def test_confidence_in_bounds(self):
        from hsi_caption.stage7_uncertainty.entropy_calculator import EntropyDecomposition
        decomposition = EntropyDecomposition(
            class_probs_mean=np.array([0.8, 0.2]), predictive_entropy=0.3,
            aleatoric_uncertainty=0.2, epistemic_uncertainty=0.1,
        )
        result = ConfidenceFusion().fuse("p1", decomposition, calibrated=True)
        assert 0.0 <= result.confidence_score <= 1.0


class TestUncertaintyEngineFacade:
    def test_process_end_to_end(self, cfg):
        engine = UncertaintyEngine(semantic_vector_dim=40, num_classes=cfg.dataset.num_classes,
                                    cfg=cfg.uncertainty, dropout_rate=0.1, random_seed=1)
        result = engine.process("p1", np.random.rand(40).astype(np.float32))
        assert result.class_probs_mean.shape == (cfg.dataset.num_classes,)
        assert 0.0 <= result.confidence_score <= 1.0
