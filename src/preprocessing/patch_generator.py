"""
Patch Generator
---------------

Generates patches from hyperspectral datasets.

Author: Manikanta Tangi
Project: HSI Caption Generation Pipeline
"""

import numpy as np


class PatchGenerator:
    """
    Generate hyperspectral patches.
    """

    @staticmethod
    def generate(
        dataset,
        patch_size=32,
        stride=32,
        ignore_background=True
    ):
        """
        Generate patches.

        Parameters
        ----------
        dataset : dict

        patch_size : int

        stride : int

        ignore_background : bool

        Returns
        -------
        list
        """

        cube = dataset["cube"]
        gt = dataset["ground_truth"]

        H, W, _ = cube.shape

        patches = []

        for row in range(0, H - patch_size + 1, stride):

            for col in range(0, W - patch_size + 1, stride):

                cube_patch = cube[
                    row:row + patch_size,
                    col:col + patch_size,
                    :
                ]

                gt_patch = gt[
                    row:row + patch_size,
                    col:col + patch_size
                ]

                if ignore_background:

                    if np.all(gt_patch == 0):
                        continue

                patches.append({

                    "cube": cube_patch,

                    "ground_truth": gt_patch,

                    "row": row,

                    "col": col

                })

        return patches