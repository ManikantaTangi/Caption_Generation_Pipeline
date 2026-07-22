"""
Dataset Normalization
---------------------

Provides normalization methods for hyperspectral datasets.

Author: Manikanta Tangi
Project: HSI Caption Generation Pipeline
"""

import numpy as np


class DatasetNormalizer:
    """
    Normalize hyperspectral datasets.
    """

    @staticmethod
    def min_max(dataset):
        """
        Min-Max normalization to [0, 1].
        """

        cube = dataset["cube"].astype(np.float32)

        minimum = np.min(cube)
        maximum = np.max(cube)

        normalized = (cube - minimum) / (maximum - minimum)

        result = dataset.copy()
        result["cube"] = normalized

        return result

    @staticmethod
    def z_score(dataset):
        """
        Standard score normalization.
        """

        cube = dataset["cube"].astype(np.float32)

        mean = np.mean(cube)
        std = np.std(cube)

        normalized = (cube - mean) / std

        result = dataset.copy()
        result["cube"] = normalized

        return result

    @staticmethod
    def bandwise(dataset):
        """
        Normalize each spectral band independently.
        """

        cube = dataset["cube"].astype(np.float32)

        normalized = np.empty_like(cube)

        for band in range(cube.shape[2]):

            image = cube[:, :, band]

            minimum = np.min(image)
            maximum = np.max(image)

            if maximum == minimum:
                normalized[:, :, band] = 0
            else:
                normalized[:, :, band] = (
                    image - minimum
                ) / (
                    maximum - minimum
                )

        result = dataset.copy()
        result["cube"] = normalized

        return result

    @staticmethod
    def l2(dataset):
        """
        L2 normalization for each pixel spectrum.
        """

        cube = dataset["cube"].astype(np.float32)

        norm = np.linalg.norm(
            cube,
            axis=2,
            keepdims=True
        )

        norm[norm == 0] = 1

        normalized = cube / norm

        result = dataset.copy()
        result["cube"] = normalized

        return result