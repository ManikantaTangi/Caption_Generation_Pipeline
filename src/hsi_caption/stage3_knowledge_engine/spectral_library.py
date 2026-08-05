"""
Stage 3 - Spectral Library
=============================
Purpose
    Load and index the reference spectral signatures (configs/spectral_library.yaml)
    that ground Stage 3's material identification in domain knowledge rather
    than purely learned embeddings — the "knowledge-guided" part of the
    pipeline's name.

Algorithm
    YAML parse -> per-material piecewise-linear interpolation of control
    points onto the sensor's actual wavelength grid (so the library is
    reusable across sensors with different band counts). O(M*B) to build
    the full library matrix, M = number of materials.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LibraryEntry:
    name: str
    category: str
    description: str
    spectrum: np.ndarray  # resampled onto sensor wavelength grid


class SpectralLibrary:
    """Indexed collection of reference material spectra."""

    def __init__(self, library_path: str, wavelengths_nm: np.ndarray) -> None:
        with open(library_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)["materials"]
        self.wavelengths_nm = wavelengths_nm
        self.entries: Dict[str, LibraryEntry] = {}
        for name, info in raw.items():
            cps = info["control_points"]
            xs = np.array(sorted(cps.keys()), dtype=float)
            ys = np.array([cps[int(x)] for x in xs], dtype=float)
            spectrum = np.interp(wavelengths_nm, xs, ys)
            self.entries[name] = LibraryEntry(
                name=name, category=info["category"], description=info.get("description", ""),
                spectrum=spectrum.astype(np.float32),
            )
        logger.info("Loaded spectral library with %d materials.", len(self.entries))

    def names(self) -> List[str]:
        return list(self.entries.keys())

    def matrix(self) -> "tuple[np.ndarray, List[str]]":
        names = self.names()
        mat = np.stack([self.entries[n].spectrum for n in names], axis=0)
        return mat, names

    def category_of(self, name: str) -> str:
        return self.entries[name].category

    def apply_normalization(self, strategy: str, params: Dict[str, np.ndarray]) -> None:
        """Re-express every library spectrum in the *same* normalized space
        as the patches it will be matched against.

        Rationale: Stage 1 normalizes the cube per-band before patches are
        cut (so the learned encoders in Stage 2 see well-scaled inputs).
        If the reference library is left in raw reflectance units, Spectral
        Angle Mapper / cosine similarity against normalized patch spectra
        becomes systematically wrong (the two vector spaces no longer
        share the same relative band-to-band shape). This method applies
        the identical min-max/z-score transform (fit on the same cube) to
        the library so both sides of the similarity computation live in
        one consistent space. Must be called once, right after the
        library is constructed and before any retrieval.
        """
        for entry in self.entries.values():
            if strategy == "minmax":
                denom = np.where((params["band_max"] - params["band_min"]) < 1e-8, 1.0,
                                  params["band_max"] - params["band_min"])
                entry.spectrum = ((entry.spectrum - params["band_min"]) / denom).astype(np.float32)
            elif strategy == "zscore":
                entry.spectrum = ((entry.spectrum - params["band_mean"]) / params["band_std"]).astype(np.float32)
            else:
                raise ValueError(f"Unsupported normalization strategy: {strategy}")
        logger.info("Applied %s normalization to spectral library (now consistent with patch space).", strategy)

    def refit_from_patches(self, patches, class_names: List[str]) -> None:
        """Replaces the hand-authored control-point spectra with EMPIRICAL
        per-class mean spectra computed from real labelled training patches.

        Rationale: `configs/spectral_library.yaml`'s control points are
        illustrative approximations of typical vegetation/water/built-up
        reflectance behaviour, hand-authored for the synthetic-data demo.
        A real sensor's raw values live on whatever scale that sensor's
        calibration produces (WHU-Hi-LongKou's raw `.mat` values range up
        to ~28, not the [0,1] reflectance the hand-authored curves assume).
        Forcing the illustrative library through the *cube's* min-max
        normalization (see `apply_normalization`) rescales it, but does
        not fix the underlying shape mismatch -- a library authored on a
        vastly different assumed dynamic range gets crushed nearly flat,
        and every material becomes spectrally indistinguishable.

        The correct fix -- standard practice for building a real spectral
        library -- is to derive each material's reference curve directly
        from real, labelled example pixels of that material, in the
        *same* normalized space the query patches live in. This method
        does exactly that: for every class present in `patches`, it
        averages the centre-pixel spectrum across all patches whose
        ground-truth label matches that class. Categories/descriptions
        from the YAML file are kept (only the numeric spectrum is
        replaced), and any class *not* present in `patches` (e.g.
        "Background", which WHU-Hi's official Train<N>.mat splits never
        include) keeps its original hand-authored fallback curve.

        Call this INSTEAD of `apply_normalization` -- patches are already
        in the correct normalized space, so no further rescaling is
        needed. Complexity: O(N_patches * B).
        """
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for patch in patches:
            p = patch.cube_data.shape[0] // 2
            spectrum = patch.cube_data[p, p, :].astype(np.float64)
            if patch.center_label >= len(class_names):
                continue
            name = class_names[patch.center_label]
            if name not in self.entries:
                continue
            sums[name] = sums.get(name, np.zeros_like(spectrum)) + spectrum
            counts[name] = counts.get(name, 0) + 1

        refit_names = []
        for name, total in sums.items():
            if counts[name] > 0:
                self.entries[name].spectrum = (total / counts[name]).astype(np.float32)
                refit_names.append(f"{name}(n={counts[name]})")

        fallback_names = [n for n in self.entries if n not in sums]
        logger.info("Refit %d/%d library spectra from real labelled patches: %s. "
                    "Kept hand-authored fallback for: %s.",
                    len(refit_names), len(self.entries), refit_names, fallback_names)
