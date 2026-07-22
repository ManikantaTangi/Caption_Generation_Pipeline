"""
Unit Tests for Patch Generator
"""

import unittest

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.patch_generator import PatchGenerator


class TestPatchGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        paths = config_loader.load("paths.yaml")

        discovery = DatasetDiscovery(
            paths["data"]["raw"] + "/WHU-Hi"
        )

        scene = discovery.discover()[0]

        dataset = DatasetLoader.load(scene)

        cls.patches = PatchGenerator.generate(
            dataset,
            patch_size=32,
            stride=32
        )

    def test_patch_count(self):

        self.assertGreater(
            len(self.patches),
            0
        )

    def test_patch_shape(self):

        patch = self.patches[0]

        self.assertEqual(
            patch["cube"].shape,
            (32, 32, patch["cube"].shape[2])
        )

        self.assertEqual(
            patch["ground_truth"].shape,
            (32, 32)
        )

    def test_patch_location(self):

        patch = self.patches[0]

        self.assertIn("row", patch)

        self.assertIn("col", patch)


if __name__ == "__main__":
    unittest.main()