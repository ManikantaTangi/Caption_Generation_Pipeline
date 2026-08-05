"""Shared fixtures: a small synthetic cube + patch dataset used across all stage tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hsi_caption.config import PipelineConfig
from hsi_caption.stage1_preprocessing.dataset_loader import SyntheticWHUHiGenerator
from hsi_caption.stage1_preprocessing.dataset_normalization import DatasetNormalizer
from hsi_caption.stage1_preprocessing.dataset_statistics import DatasetStatisticsComputer
from hsi_caption.stage1_preprocessing.patch_generator import PatchGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.test.yaml")


@pytest.fixture(scope="session")
def cfg():
    c = PipelineConfig.from_yaml(CONFIG_PATH)
    c.validate()
    return c


@pytest.fixture(scope="session")
def small_cube(cfg):
    gen = SyntheticWHUHiGenerator(
        cfg.knowledge.spectral_library_path, cfg.dataset.class_names, cfg.dataset.num_bands,
        tuple(cfg.dataset.wavelength_range_nm), seed=cfg.patch.random_seed,
    )
    return gen.generate(height=40, width=40, num_seeds=10)


@pytest.fixture(scope="session")
def normalized_cube_and_params(cfg, small_cube):
    stats = DatasetStatisticsComputer().compute(small_cube)
    norm_cube, params = DatasetNormalizer(cfg.patch.normalize).fit_transform(small_cube, stats)
    return norm_cube, params


@pytest.fixture(scope="session")
def patch_dataset(cfg, normalized_cube_and_params):
    norm_cube, params = normalized_cube_and_params
    pg = PatchGenerator(cfg.patch.patch_size, stride=6, train_ratio=0.7, val_ratio=0.15,
                         test_ratio=0.15, random_seed=cfg.patch.random_seed)
    return pg.generate(norm_cube, normalization_stats=params)


@pytest.fixture(scope="session")
def sample_patch(patch_dataset):
    return patch_dataset.train[0]
