"""
Dataset Inspector
-----------------

Analyzes an HSI dataset and reports useful statistics.
"""

import numpy as np


class DatasetInspector:

    @staticmethod
    def inspect(dataset):

        cube = dataset["cube"]
        gt = dataset["ground_truth"]

        report = {

            "Scene": dataset["scene"],

            "Height": cube.shape[0],

            "Width": cube.shape[1],

            "Bands": cube.shape[2],

            "Data Type": cube.dtype,

            "Minimum": float(np.min(cube)),

            "Maximum": float(np.max(cube)),

            "Mean": float(np.mean(cube)),

            "Std": float(np.std(cube)),

            "NaN Values": int(np.isnan(cube).sum()),

            "Infinite Values": int(np.isinf(cube).sum()),

            "Classes": len(np.unique(gt)) - 1,

            "Pixels": cube.shape[0] * cube.shape[1]
        }

        return report