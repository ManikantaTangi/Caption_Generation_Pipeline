"""
Stage 3 - Knowledge Retrieval
================================
Purpose
    Retrieve the top-k spectral-library materials most similar to a
    query spectrum. This is the mechanism that grounds every downstream
    symbolic stage (rules, ontology, facts) in physical measurements
    rather than opaque learned features.

Input
    A query spectrum (B,) — typically the patch-centre pixel from Stage
    1's Patch.cube_data — and the SpectralLibrary built in this stage.

Algorithm
    Three interchangeable similarity metrics (configurable):
      - Spectral Angle Mapper (SAM, default): angle between two spectra
        as vectors, invariant to multiplicative illumination scaling —
        the standard, physically-motivated metric in the HSI literature
        for material matching (unlike Euclidean distance, SAM is not
        confused by brightness differences from illumination/shadow).
        SAM(x, r) = arccos( (x . r) / (||x|| ||r||) )
      - Cosine similarity: equivalent ranking to SAM but returns a
        similarity in [-1, 1] instead of an angle in radians; used when
        a similarity score is more convenient than a distance.
      - Euclidean: raw distance; sensitive to brightness, included as a
        baseline for ablation comparison in the IEEE write-up.
    Complexity: O(M*B) per query (M = library size, B = bands) — trivial
    for M~9, B~270.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from hsi_caption.datatypes import MaterialMatch
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary

logger = logging.getLogger(__name__)


class KnowledgeRetrievalError(Exception):
    """Raised for an unsupported similarity metric."""


class KnowledgeRetrieval:
    """Top-k spectral similarity search against the SpectralLibrary."""

    _SUPPORTED = ("spectral_angle_mapper", "cosine", "euclidean")

    def __init__(self, library: SpectralLibrary, metric: str, top_k: int) -> None:
        if metric not in self._SUPPORTED:
            raise KnowledgeRetrievalError(f"Unsupported similarity metric: {metric}")
        self.library = library
        self.metric = metric
        self.top_k = top_k
        self._matrix, self._names = library.matrix()  # (M, B)

    @staticmethod
    def _sam_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q = query / (np.linalg.norm(query) + 1e-8)
        r = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        cos = np.clip(r @ q, -1.0, 1.0)
        angle = np.arccos(cos)
        return 1.0 - (angle / np.pi)  # map to a [0,1] similarity, higher = closer

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q = query / (np.linalg.norm(query) + 1e-8)
        r = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return r @ q

    @staticmethod
    def _euclidean_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(matrix - query[None, :], axis=1)
        return 1.0 / (1.0 + dist)  # convert distance to a bounded similarity

    def retrieve(self, query_spectrum: np.ndarray) -> List[MaterialMatch]:
        if self.metric == "spectral_angle_mapper":
            sims = self._sam_similarity(query_spectrum, self._matrix)
        elif self.metric == "cosine":
            sims = self._cosine_similarity(query_spectrum, self._matrix)
        else:
            sims = self._euclidean_similarity(query_spectrum, self._matrix)

        order = np.argsort(-sims)[: self.top_k]
        matches = [
            MaterialMatch(
                material_name=self._names[i],
                category=self.library.category_of(self._names[i]),
                similarity=float(sims[i]),
            )
            for i in order
        ]
        return matches
