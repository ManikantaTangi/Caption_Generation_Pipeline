"""
Unit Tests for Dataset Inspector
"""

import unittest

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.inspector import DatasetInspector


class TestDatasetInspector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        cls.scene = discovery.discover()[0]

    def test_report_generation(self):

        dataset = DatasetLoader.load(self.scene)

        report = DatasetInspector.inspect(dataset)

        required_keys = {

            "Scene",
            "Height",
            "Width",
            "Bands",
            "Data Type",
            "Minimum",
            "Maximum",
            "Mean",
            "Std",
            "NaN Values",
            "Infinite Values",
            "Classes",
            "Pixels"

        }

        self.assertTrue(
            required_keys.issubset(
                report.keys()
            )
        )

        self.assertGreater(
            report["Bands"],
            0
        )

        self.assertGreater(
            report["Pixels"],
            0
        )


if __name__ == "__main__":
    unittest.main()
    