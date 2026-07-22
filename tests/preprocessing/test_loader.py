"""
Unit Tests for Dataset Loader
"""

import unittest
import numpy as np

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        cls.scenes = discovery.discover()

    def test_loading(self):

        for scene in self.scenes:

            dataset = DatasetLoader.load(scene)

            self.assertEqual(dataset["cube"].ndim, 3)

            self.assertEqual(
                dataset["ground_truth"].ndim,
                2
            )

            self.assertGreater(
                dataset["bands"],
                0
            )

            self.assertGreater(
                dataset["classes"],
                0
            )

            self.assertIsInstance(
                dataset["cube"],
                np.ndarray
            )


if __name__ == "__main__":
    unittest.main()
    