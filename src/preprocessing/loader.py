"""
Dataset Loader
--------------

Loads validated hyperspectral datasets.
"""

from scipy.io import loadmat
import numpy as np


class DatasetLoader:
    """
    Load HSI datasets into a standard format.
    """

    @staticmethod
    def _extract_data(mat_dict):
        """
        Extract the first non-metadata variable from a MAT file.
        """
        for key, value in mat_dict.items():
            if not key.startswith("__"):
                return value

        raise ValueError("No valid data found in MAT file.")

    @classmethod
    def load(cls, scene_info):
        """
        Load HSI cube and ground truth.

        Parameters
        ----------
        scene_info : dict

        Returns
        -------
        dict
        """

        cube = cls._extract_data(
            loadmat(scene_info["cube"])
        )

        gt = cls._extract_data(
            loadmat(scene_info["ground_truth"])
        )

        H, W, B = cube.shape

        metadata = {
            "dataset": "WHU-Hi",
            "scene": scene_info["scene"],

            "cube": cube,
            "ground_truth": gt,

            "height": H,
            "width": W,
            "bands": B,

            "classes": len(np.unique(gt)) - 1,

            "dtype": str(cube.dtype)
        }

        return metadata