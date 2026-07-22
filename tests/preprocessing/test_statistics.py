"""
Unit Tests for Dataset Statistics
"""

import unittest
import numpy as np

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.statistics import DatasetStatistics


class TestDatasetStatistics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        scene = discovery.discover()[0]

        dataset = DatasetLoader.load(scene)

        cls.stats = DatasetStatistics.compute(dataset)

    def test_keys_exist(self):

        required = {

            "scene",
            "height",
            "width",
            "bands",
            "pixels",
            "classes",
            "dtype",
            "minimum",
            "maximum",
            "mean",
            "median",
            "std",
            "variance",
            "nan_values",
            "infinite_values",
            "spectral_mean",
            "spectral_std",
            "class_distribution"

        }

        self.assertTrue(
            required.issubset(
                self.stats.keys()
            )
        )

    def test_dimensions(self):

        self.assertGreater(
            self.stats["height"],
            0
        )

        self.assertGreater(
            self.stats["width"],
            0
        )

        self.assertGreater(
            self.stats["bands"],
            0
        )

    def test_statistics(self):

        self.assertLessEqual(
            self.stats["minimum"],
            self.stats["maximum"]
        )

        self.assertGreaterEqual(
            self.stats["std"],
            0
        )

    def test_no_nan(self):

        self.assertEqual(
            self.stats["nan_values"],
            0
        )

    def test_no_inf(self):

        self.assertEqual(
            self.stats["infinite_values"],
            0
        )

    def test_spectral_vectors(self):

        self.assertIsInstance(
            self.stats["spectral_mean"],
            np.ndarray
        )

        self.assertIsInstance(
            self.stats["spectral_std"],
            np.ndarray
        )


if __name__ == "__main__":
    unittest.main()