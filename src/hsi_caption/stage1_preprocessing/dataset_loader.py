"""
Stage 1 - Dataset Loader
=========================
Purpose
    Materialise a `HSICube` from whatever `DatasetDiscovery` found. Supports
    real WHU-Hi `.mat` files (via scipy.io, MATLAB v5) and `.npy` arrays.
    When no real data is available, `SyntheticWHUHiGenerator` produces a
    cube with the same structure as WHU-Hi-LongKou (270 bands, 400-1000nm,
    9 land-cover classes) using class-conditioned Gaussian spectral
    signatures drawn from the same spectral library used in Stage 3 --
    this keeps every downstream stage exercised realistically without
    requiring network access to the dataset host.

Algorithm
    Real path: scipy.io.loadmat / np.load -> O(H*W*B) memory copy.
    Synthetic path: procedurally paint contiguous class regions (via
    Voronoi-like seeding) then sample each pixel's spectrum from its
    class's reference curve + Gaussian sensor noise. O(H*W*B).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import yaml
from scipy.io import loadmat

from hsi_caption.datatypes import HSICube

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when a dataset file cannot be parsed into an HSICube."""


class DatasetLoader:
    """Loads a real WHU-Hi cube from disk (.mat / .npy)."""

    def __init__(self, class_names: list, num_bands: int, wavelength_range_nm: tuple) -> None:
        self.class_names = class_names
        self.num_bands = num_bands
        self.wavelength_range_nm = wavelength_range_nm

    def load(self, image_path: str, label_path: Optional[str] = None) -> HSICube:
        ext = os.path.splitext(image_path)[1].lower()
        try:
            if ext == ".mat":
                data = self._load_mat_array(image_path)
            elif ext == ".npy":
                data = np.load(image_path)
            else:
                raise DatasetLoadError(f"Unsupported image extension: {ext}")
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise DatasetLoadError(f"Failed to load image {image_path}: {exc}") from exc

        labels = None
        if label_path is not None:
            try:
                labels = self._load_mat_array(label_path) if label_path.endswith(".mat") else np.load(label_path)
            except Exception as exc:  # noqa: BLE001
                raise DatasetLoadError(f"Failed to load labels {label_path}: {exc}") from exc

        wavelengths = np.linspace(self.wavelength_range_nm[0], self.wavelength_range_nm[1], data.shape[-1])
        data32 = np.asarray(data, dtype=np.float32)
        if data32 is not data:
            del data  # release the float64 original as soon as the float32 copy exists
        return HSICube(data=data32, labels=labels, wavelengths_nm=wavelengths,
                        class_names=self.class_names, name=os.path.basename(image_path))

    @staticmethod
    def _load_mat_array(path: str) -> np.ndarray:
        mat = loadmat(path)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        if not arrays:
            raise DatasetLoadError(f"No arrays found in .mat file {path}")
        # WHU-Hi releases typically store a single variable; take the largest.
        key = max(arrays, key=lambda k: np.asarray(arrays[k]).size)
        return np.asarray(arrays[key])


class SyntheticWHUHiGenerator:
    """Generates a WHU-Hi-LongKou-shaped synthetic cube for pipeline
    development/testing when real data is unavailable.

    This is clearly labelled synthetic (`HSICube.name` prefix) so it is
    never confused with real experimental results in reports.
    """

    def __init__(self, spectral_library_path: str, class_names: list,
                 num_bands: int, wavelength_range_nm: tuple, seed: int = 42) -> None:
        self.class_names = class_names
        self.num_bands = num_bands
        self.wavelength_range_nm = wavelength_range_nm
        self.rng = np.random.default_rng(seed)
        with open(spectral_library_path, "r", encoding="utf-8") as fh:
            self.library = yaml.safe_load(fh)["materials"]

    def _reference_spectrum(self, material: str, wavelengths: np.ndarray) -> np.ndarray:
        cps = self.library[material]["control_points"]
        xs = np.array(sorted(cps.keys()), dtype=float)
        ys = np.array([cps[int(x)] for x in xs], dtype=float)
        return np.interp(wavelengths, xs, ys)

    def generate(self, height: int = 120, width: int = 120, num_seeds: int = 24) -> HSICube:
        wavelengths = np.linspace(self.wavelength_range_nm[0], self.wavelength_range_nm[1], self.num_bands)
        num_classes = len(self.class_names)

        # 1. Voronoi-style region painting for spatially-contiguous classes.
        seed_rows = self.rng.integers(0, height, size=num_seeds)
        seed_cols = self.rng.integers(0, width, size=num_seeds)
        seed_classes = self.rng.integers(1, num_classes, size=num_seeds)  # skip 0=Background as a seed
        rr, cc = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        labels = np.zeros((height, width), dtype=np.int32)
        dist_stack = np.stack([
            (rr - seed_rows[i]) ** 2 + (cc - seed_cols[i]) ** 2 for i in range(num_seeds)
        ], axis=-1)
        nearest = np.argmin(dist_stack, axis=-1)
        for i in range(num_seeds):
            labels[nearest == i] = seed_classes[i]

        # 2. Reference spectra per class, resampled onto the sensor grid.
        ref_spectra = np.stack([self._reference_spectrum(name, wavelengths) for name in self.class_names], axis=0)

        # 3. Paint pixel spectra = class reference + multiplicative texture
        #    noise + additive sensor noise (mimics NIR canopy heterogeneity).
        cube = np.zeros((height, width, self.num_bands), dtype=np.float32)
        for cls_id in range(num_classes):
            mask = labels == cls_id
            n_pix = int(mask.sum())
            if n_pix == 0:
                continue
            texture = self.rng.normal(1.0, 0.05, size=(n_pix, 1))
            sensor_noise = self.rng.normal(0.0, 0.01, size=(n_pix, self.num_bands))
            cube[mask] = np.clip(ref_spectra[cls_id][None, :] * texture + sensor_noise, 0.0, 1.0)

        logger.info("Generated synthetic WHU-Hi-LongKou-shaped cube: %dx%dx%d, %d classes",
                    height, width, self.num_bands, num_classes)
        return HSICube(data=cube, labels=labels, wavelengths_nm=wavelengths,
                        class_names=self.class_names, name="synthetic_WHU-Hi-LongKou")
