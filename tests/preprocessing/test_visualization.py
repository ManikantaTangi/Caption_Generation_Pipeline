"""
Unit Tests for Dataset Visualization
"""

import unittest
import matplotlib.pyplot as plt

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.visualization import DatasetVisualizer


class TestDatasetVisualizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        scene = discovery.discover()[0]

        cls.dataset = DatasetLoader.load(scene)

    def tearDown(self):

        plt.close("all")

    def test_show_rgb(self):

        DatasetVisualizer.show_rgb(self.dataset)

        self.assertTrue(True)

    def test_show_ground_truth(self):

        DatasetVisualizer.show_ground_truth(self.dataset)

        self.assertTrue(True)

    def test_show_single_band(self):

        DatasetVisualizer.show_single_band(
            self.dataset,
            band=10
        )

        self.assertTrue(True)

    def test_histogram(self):

        DatasetVisualizer.show_band_histogram(
            self.dataset,
            band=10
        )

        self.assertTrue(True)

    def test_compare_bands(self):

        DatasetVisualizer.compare_bands(
            self.dataset,
            bands=(10, 30, 50)
        )

        self.assertTrue(True)

    def test_class_distribution(self):

        DatasetVisualizer.show_class_distribution(
            self.dataset
        )

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()