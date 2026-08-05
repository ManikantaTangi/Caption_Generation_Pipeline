"""
Stage 3 - Material Identification
=====================================
Purpose
    Turn per-pixel spectral retrieval into a patch-level material
    *composition* (fraction of the patch best-matched to each library
    material). This is the quantitative summary Stage 4's rule engine
    and Stage 6's fact generator both consume ("this patch is ~72% Rice,
    ~20% Water, ~8% other").

Input
    `Patch.cube_data` (P, P, B) — the full patch cube from Stage 1 — and
    the `KnowledgeRetrieval` instance built on the SpectralLibrary.

Algorithm
    For every pixel in the patch, retrieve its single best-matching
    library material (top_k=1 SAM/cosine/euclidean lookup), then tally
    fractions across all P*P pixels. This is deliberately
    measurement-driven (it never looks at Stage 1's ground-truth label),
    which is what makes it a genuine "identification" step rather than a
    label lookup — the fractions it produces are an independent estimate
    that Stage 7/8 later cross-check against.
    Complexity: O(P^2 * M * B) — dominated by the per-pixel retrieval.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict

import numpy as np

from hsi_caption.stage3_knowledge_engine.knowledge_retrieval import KnowledgeRetrieval

logger = logging.getLogger(__name__)


class MaterialIdentifier:
    """Computes per-patch material composition via pixel-wise spectral matching."""

    def __init__(self, retrieval: KnowledgeRetrieval) -> None:
        self.retrieval = retrieval

    def identify(self, cube_patch: np.ndarray) -> Dict[str, float]:
        p = cube_patch.shape[0]
        pixels = cube_patch.reshape(-1, cube_patch.shape[-1])
        best_materials = []
        for px in pixels:
            top1 = self.retrieval.retrieve(px)[0]  # retrieve() returns sorted desc by similarity
            best_materials.append(top1.material_name)
        counts = Counter(best_materials)
        total = len(best_materials)
        fractions = {name: count / total for name, count in counts.items()}
        # ensure every library material has an entry (0.0 if absent) for downstream determinism
        for name in self.retrieval.library.names():
            fractions.setdefault(name, 0.0)
        return fractions
