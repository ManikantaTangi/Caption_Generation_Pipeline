"""
Dataset Validator
-----------------
Validates HSI dataset files before loading.
"""

from pathlib import Path
from scipy.io import loadmat


class DatasetValidator:
    """Validate HSI cube and ground truth files."""

    @staticmethod
    def validate(scene_info: dict) -> bool:
        """
        Validate dataset structure.

        Parameters
        ----------
        scene_info : dict
            Dictionary returned by DatasetDiscovery.

        Returns
        -------
        bool
            True if dataset is valid.
        """

        cube_path = scene_info["cube"]
        gt_path = scene_info["ground_truth"]

        # Check file existence
        if cube_path is None or not Path(cube_path).exists():
            raise FileNotFoundError(
                f"Cube file not found: {cube_path}"
            )

        if gt_path is None or not Path(gt_path).exists():
            raise FileNotFoundError(
                f"Ground truth file not found: {gt_path}"
            )

        # Load .mat files
        cube_data = loadmat(cube_path)
        gt_data = loadmat(gt_path)

        # Find the first non-metadata variable
        cube = next(
            value for key, value in cube_data.items()
            if not key.startswith("__")
        )

        gt = next(
            value for key, value in gt_data.items()
            if not key.startswith("__")
        )

        # Check dimensions
        if cube.ndim != 3:
            raise ValueError(
                f"Cube must be 3D, got {cube.ndim}D"
            )

        if gt.ndim != 2:
            raise ValueError(
                f"Ground truth must be 2D, got {gt.ndim}D"
            )

        # Spatial consistency
        if cube.shape[:2] != gt.shape:
            raise ValueError(
                "Cube and ground truth dimensions do not match."
            )

        return True