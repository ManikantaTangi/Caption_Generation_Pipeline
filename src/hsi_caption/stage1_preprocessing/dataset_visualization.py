"""
Stage 1 - Dataset Visualization
=================================
Purpose
    Render false-colour composites, ground-truth class maps, and mean
    spectral-signature plots to disk for IEEE-figure-ready inspection and
    for sanity-checking the synthetic/real cube before it enters the
    learned stages.

Algorithm
    Band selection for false colour: nearest bands to (650, 550, 450) nm
    (approx. R, G, B) -> per-channel percentile stretch (2nd/98th) for
    contrast. O(H*W) per plot.
"""
from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

from hsi_caption.datatypes import HSICube

logger = logging.getLogger(__name__)


class DatasetVisualizer:
    """Generates and saves standard HSI inspection figures."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _nearest_band(wavelengths_nm: np.ndarray, target_nm: float) -> int:
        return int(np.argmin(np.abs(wavelengths_nm - target_nm)))

    @staticmethod
    def _percentile_stretch(channel: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(channel, [2, 98])
        if hi <= lo:
            return np.zeros_like(channel)
        return np.clip((channel - lo) / (hi - lo), 0, 1)

    def false_color_composite(self, cube: HSICube) -> np.ndarray:
        r = self._nearest_band(cube.wavelengths_nm, 650)
        g = self._nearest_band(cube.wavelengths_nm, 550)
        b = self._nearest_band(cube.wavelengths_nm, 450)
        rgb = np.stack([
            self._percentile_stretch(cube.data[:, :, r]),
            self._percentile_stretch(cube.data[:, :, g]),
            self._percentile_stretch(cube.data[:, :, b]),
        ], axis=-1)
        return rgb

    def save_false_color(self, cube: HSICube, filename: str = "false_color.png") -> str:
        rgb = self.false_color_composite(cube)
        path = os.path.join(self.output_dir, filename)
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title(f"False-colour composite: {cube.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Saved false-colour composite to %s", path)
        return path

    def save_class_map(self, cube: HSICube, filename: str = "class_map.png") -> str:
        if cube.labels is None:
            raise ValueError("Cube has no labels; cannot render class map.")
        path = os.path.join(self.output_dir, filename)
        cmap = plt.get_cmap("tab10", len(cube.class_names))
        plt.figure(figsize=(6, 6))
        im = plt.imshow(cube.labels, cmap=cmap, vmin=0, vmax=len(cube.class_names) - 1)
        cbar = plt.colorbar(im, ticks=range(len(cube.class_names)))
        cbar.ax.set_yticklabels(cube.class_names, fontsize=7)
        plt.title(f"Ground-truth class map: {cube.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Saved class map to %s", path)
        return path

    def save_mean_spectra(self, cube: HSICube, filename: str = "mean_spectra.png") -> str:
        if cube.labels is None:
            raise ValueError("Cube has no labels; cannot compute per-class mean spectra.")
        path = os.path.join(self.output_dir, filename)
        plt.figure(figsize=(7, 4))
        for idx, cls_name in enumerate(cube.class_names):
            mask = cube.labels == idx
            if mask.sum() == 0:
                continue
            mean_spec = cube.data[mask].mean(axis=0)
            plt.plot(cube.wavelengths_nm, mean_spec, label=cls_name, linewidth=1.2)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title(f"Mean per-class spectral signatures: {cube.name}")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Saved mean spectra plot to %s", path)
        return path
