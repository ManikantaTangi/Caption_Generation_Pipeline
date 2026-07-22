"""
Dataset Visualization
---------------------

Visualizes hyperspectral datasets using RGB composites,
ground truth maps, spectral bands, histograms, and more.

Author: Manikanta Tangi
Project: HSI Caption Generation Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt


class DatasetVisualizer:
    """
    Visualization utilities for hyperspectral datasets.
    """

    @staticmethod
    def _normalize(image):
        """
        Normalize an image to [0, 1].

        Parameters
        ----------
        image : ndarray

        Returns
        -------
        ndarray
        """

        image = image.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum - minimum == 0:
            return np.zeros_like(image)

        return (image - minimum) / (maximum - minimum)

    @classmethod
    def show_rgb(
        cls,
        dataset,
        bands=(50, 30, 10),
        figsize=(8, 8),
        title=None,
        save_path=None
    ):
        """
        Display RGB composite image.

        Parameters
        ----------
        dataset : dict
            Output from DatasetLoader.

        bands : tuple
            RGB band indices.

        figsize : tuple

        title : str

        save_path : str
        """

        cube = dataset["cube"]

        if cube.ndim != 3:
            raise ValueError("Cube must be 3D.")

        if max(bands) >= cube.shape[2]:
            raise ValueError("Band index exceeds cube dimensions.")

        rgb = np.stack([
            cube[:, :, bands[0]],
            cube[:, :, bands[1]],
            cube[:, :, bands[2]]
        ], axis=-1)

        rgb = cls._normalize(rgb)

        plt.figure(figsize=figsize)

        plt.imshow(rgb)

        plt.title(
            title or
            f"{dataset['scene']} RGB Composite"
        )

        plt.axis("off")

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    @classmethod
    def show_ground_truth(
        cls,
        dataset,
        figsize=(8, 8),
        title=None,
        save_path=None
    ):
        """
        Display ground truth labels.
        """

        gt = dataset["ground_truth"]

        plt.figure(figsize=figsize)

        plt.imshow(gt, cmap="tab20")

        plt.colorbar()

        plt.title(
            title or
            f"{dataset['scene']} Ground Truth"
        )

        plt.axis("off")

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    @classmethod
    def show_single_band(
        cls,
        dataset,
        band,
        figsize=(8, 8),
        title=None,
        save_path=None
    ):
        """
        Display one spectral band.
        """

        cube = dataset["cube"]

        if band >= cube.shape[2]:
            raise ValueError("Invalid band index.")

        image = cube[:, :, band]

        plt.figure(figsize=figsize)

        plt.imshow(image, cmap="gray")

        plt.colorbar()

        plt.title(
            title or
            f"{dataset['scene']} Band {band}"
        )

        plt.axis("off")

        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    @classmethod
    def show_band_histogram(
        cls,
        dataset,
        band,
        bins=100,
        figsize=(8, 5)
    ):
        """
        Histogram of a spectral band.
        """

        cube = dataset["cube"]

        if band >= cube.shape[2]:
            raise ValueError("Invalid band index.")

        plt.figure(figsize=figsize)

        plt.hist(
            cube[:, :, band].ravel(),
            bins=bins
        )

        plt.title(
            f"{dataset['scene']} Band {band} Histogram"
        )

        plt.xlabel("Reflectance")

        plt.ylabel("Frequency")

        plt.grid(True)

        plt.show()

    @classmethod
    def compare_bands(
        cls,
        dataset,
        bands=(10, 30, 50),
        figsize=(15, 5)
    ):
        """
        Compare multiple spectral bands.
        """

        cube = dataset["cube"]

        fig, axes = plt.subplots(
            1,
            len(bands),
            figsize=figsize
        )

        for ax, band in zip(axes, bands):

            ax.imshow(
                cube[:, :, band],
                cmap="gray"
            )

            ax.set_title(f"Band {band}")

            ax.axis("off")

        plt.tight_layout()

        plt.show()

    @classmethod
    def show_class_distribution(
        cls,
        dataset,
        figsize=(8, 5)
    ):
        """
        Show class distribution.
        """

        gt = dataset["ground_truth"]

        classes, counts = np.unique(
            gt,
            return_counts=True
        )

        plt.figure(figsize=figsize)

        plt.bar(classes, counts)

        plt.title(
            f"{dataset['scene']} Class Distribution"
        )

        plt.xlabel("Class")

        plt.ylabel("Pixels")

        plt.grid(True)

        plt.show()