"""
Unit Tests for Dataset Normalization
"""

import unittest
import numpy as np

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.normalization import DatasetNormalizer


class TestDatasetNormalization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        scene = discovery.discover()[0]

        cls.dataset = DatasetLoader.load(scene)

    def test_min_max(self):

        dataset = DatasetNormalizer.min_max(
            self.dataset
        )

        cube = dataset["cube"]

        self.assertGreaterEqual(cube.min(), 0)

        self.assertLessEqual(cube.max(), 1)

    def test_z_score(self):

        dataset = DatasetNormalizer.z_score(
            self.dataset
        )

        cube = dataset["cube"]

        self.assertAlmostEqual(
            cube.mean(),
            0,
            places=1
        )

    def test_bandwise(self):

        dataset = DatasetNormalizer.bandwise(
            self.dataset
        )

        cube = dataset["cube"]

        self.assertEqual(
            cube.shape,
            self.dataset["cube"].shape
        )

    def test_l2(self):

        dataset = DatasetNormalizer.l2(
            self.dataset
        )

        cube = dataset["cube"]

        self.assertEqual(
            cube.shape,
            self.dataset["cube"].shape
        )

        self.assertIsInstance(
            cube,
            np.ndarray
        )


if __name__ == "__main__":
    unittest.main()