"""
Unit Tests for Dataset Validator
"""

import unittest

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.validator import DatasetValidator


class TestDatasetValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        cls.scenes = discovery.discover()

    def test_all_scenes_are_valid(self):

        for scene in self.scenes:
            self.assertTrue(
                DatasetValidator.validate(scene)
            )


if __name__ == "__main__":
    unittest.main()