"""
Stage 3 - Semantic Representation
====================================
Purpose
    Fuse (a) the top-k material-match similarity scores, (b) the
    material's category as a one-hot signal, and (c) Stage 2's learned
    `fused_vector` into one `semantic_vector` — the Knowledge Embedding
    that is Stage 3's contractual output. This is what "knowledge-guided"
    means concretely: the learned embedding is not used in isolation but
    concatenated with an explicit, library-grounded knowledge signal.

Input
    Stage 2's `FusedEmbedding.fused_vector`, this stage's
    `List[MaterialMatch]` (from KnowledgeRetrieval) and material
    fractions (from MaterialIdentifier).

Algorithm
    semantic_vector = concat(
        fused_vector,                                  # learned signal
        top_k similarity scores (padded/truncated to top_k),
        category one-hot of the top-1 match,
        material fraction vector (ordered by library.names())
    )
    A simple, interpretable concatenation is deliberately preferred over
    a learned fusion here (unlike Stage 2's attention fusion): this
    vector is consumed by Stage 7's Bayesian predictor and Stage 8's
    similarity checks, where interpretability of which sub-block
    contributed what is valuable for the Explainability requirement.
    Complexity: O(fused_dim + k + C + M).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from hsi_caption.datatypes import KnowledgeEmbedding, MaterialMatch
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary

logger = logging.getLogger(__name__)


class SemanticRepresentationBuilder:
    """Builds the fused knowledge-guided semantic vector for one patch."""

    def __init__(self, library: SpectralLibrary, top_k: int) -> None:
        self.library = library
        self.top_k = top_k
        self.categories = sorted({e.category for e in library.entries.values()})
        self.material_order = library.names()

    def _category_one_hot(self, category: str) -> np.ndarray:
        vec = np.zeros(len(self.categories), dtype=np.float32)
        if category in self.categories:
            vec[self.categories.index(category)] = 1.0
        return vec

    def build(self, patch_id: str, fused_vector: np.ndarray, matches: List[MaterialMatch],
              kg_node_ids: List[str], material_fractions: Dict[str, float]) -> KnowledgeEmbedding:
        sims = np.array([m.similarity for m in matches[: self.top_k]], dtype=np.float32)
        if sims.shape[0] < self.top_k:
            sims = np.pad(sims, (0, self.top_k - sims.shape[0]))

        top_category = matches[0].category if matches else "unknown"
        cat_vec = self._category_one_hot(top_category)
        fraction_vec = np.array([material_fractions.get(n, 0.0) for n in self.material_order], dtype=np.float32)

        semantic_vector = np.concatenate([fused_vector, sims, cat_vec, fraction_vec])
        return KnowledgeEmbedding(
            patch_id=patch_id, material_matches=matches, kg_node_ids=kg_node_ids,
            semantic_vector=semantic_vector, material_fractions=material_fractions,
        )
