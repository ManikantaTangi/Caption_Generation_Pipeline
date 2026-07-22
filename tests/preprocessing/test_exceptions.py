"""
Unit Tests for Custom Exceptions
"""

import unittest

from src.preprocessing.exceptions import *


class TestExceptions(unittest.TestCase):

    def test_dataset_not_found(self):

        with self.assertRaises(
            DatasetNotFoundError
        ):
            raise DatasetNotFoundError()

    def test_invalid_cube(self):

        with self.assertRaises(
            InvalidCubeShapeError
        ):
            raise InvalidCubeShapeError()

    def test_patch_generation(self):

        with self.assertRaises(
            PatchGenerationError
        ):
            raise PatchGenerationError()

    def test_visualization(self):

        with self.assertRaises(
            VisualizationError
        ):
            raise VisualizationError()

    def test_statistics(self):

        with self.assertRaises(
            StatisticsError
        ):
            raise StatisticsError()


if __name__ == "__main__":
    unittest.main()