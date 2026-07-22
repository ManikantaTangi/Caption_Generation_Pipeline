"""
Dataset Statistics
------------------

Computes descriptive statistics for hyperspectral datasets.

Author: Manikanta Tangi
Project: HSI Caption Generation Pipeline
"""

import numpy as np


class DatasetStatistics:
    """
    Compute descriptive statistics for an HSI dataset.
    """

    @staticmethod
    def compute(dataset):
        """
        Compute statistics for the dataset.

        Parameters
        ----------
        dataset : dict
            Output from DatasetLoader.

        Returns
        -------
        dict
        """

        cube = dataset["cube"]
        gt = dataset["ground_truth"]

        stats = {

            "scene": dataset["scene"],

            "height": cube.shape[0],

            "width": cube.shape[1],

            "bands": cube.shape[2],

            "pixels": cube.shape[0] * cube.shape[1],

            "classes": len(np.unique(gt)) - 1,

            "dtype": str(cube.dtype),

            "minimum": float(np.min(cube)),

            "maximum": float(np.max(cube)),

            "mean": float(np.mean(cube)),

            "median": float(np.median(cube)),

            "std": float(np.std(cube)),

            "variance": float(np.var(cube)),

            "nan_values": int(np.isnan(cube).sum()),

            "infinite_values": int(np.isinf(cube).sum()),

            "spectral_mean": np.mean(cube, axis=(0, 1)),

            "spectral_std": np.std(cube, axis=(0, 1)),

            "class_distribution": {
                int(c): int(np.sum(gt == c))
                for c in np.unique(gt)
            }

        }

        return stats