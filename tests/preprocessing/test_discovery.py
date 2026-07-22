"""
Unit Tests for Dataset Discovery
"""

import unittest

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery


class TestDatasetDiscovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paths = config_loader.load("paths.yaml")
        dataset_root = paths["data"]["raw"] + "/WHU-Hi"
        cls.discovery = DatasetDiscovery(dataset_root)

    def test_scene_count(self):
        scenes = self.discovery.discover()
        self.assertEqual(len(scenes), 3)

    def test_required_keys(self):

        scenes = self.discovery.discover()

        required = {
            "scene",
            "cube",
            "ground_truth",
            "split_folder",
        }

        for scene in scenes:
            self.assertTrue(required.issubset(scene.keys()))

    def test_paths_exist(self):

        scenes = self.discovery.discover()

        for scene in scenes:
            self.assertTrue(scene["cube"].exists())
            self.assertTrue(scene["ground_truth"].exists())
            self.assertTrue(scene["split_folder"].exists())


if __name__ == "__main__":
    unittest.main()