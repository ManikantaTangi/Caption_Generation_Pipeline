"""
Stage 3 - Spectral Knowledge Engine (facade)
================================================
Wires SpectralLibrary + KnowledgeRetrieval + MaterialIdentifier +
KnowledgeGraphBuilder + SemanticRepresentationBuilder into one callable
that takes a Stage-1 `Patch` + Stage-2 `FusedEmbedding` and returns the
Stage-3 `KnowledgeEmbedding`.
"""
from __future__ import annotations

import logging

import numpy as np

from typing import Dict, Optional

from hsi_caption.config import KnowledgeConfig
from hsi_caption.datatypes import FusedEmbedding, KnowledgeEmbedding, Patch
from hsi_caption.stage3_knowledge_engine.knowledge_graph import KnowledgeGraphBuilder
from hsi_caption.stage3_knowledge_engine.knowledge_retrieval import KnowledgeRetrieval
from hsi_caption.stage3_knowledge_engine.material_identification import MaterialIdentifier
from hsi_caption.stage3_knowledge_engine.semantic_representation import SemanticRepresentationBuilder
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary

logger = logging.getLogger(__name__)


class SpectralKnowledgeEngine:
    """Facade combining all five Stage 3 modules."""

    def __init__(self, wavelengths_nm: np.ndarray, cfg: KnowledgeConfig,
                 normalization_strategy: Optional[str] = None,
                 normalization_params: Optional[Dict[str, np.ndarray]] = None,
                 fit_patches: Optional[list] = None, fit_class_names: Optional[list] = None) -> None:
        self.library = SpectralLibrary(cfg.spectral_library_path, wavelengths_nm)
        if fit_patches and fit_class_names:
            # Preferred path: derive reference spectra empirically from real
            # labelled training patches (see SpectralLibrary.refit_from_patches
            # for why this is necessary, not just nicer, on real sensor data).
            self.library.refit_from_patches(fit_patches, fit_class_names)
        elif normalization_strategy and normalization_params:
            # Fallback path (no labelled patches available, e.g. a cold-start
            # / unlabelled-scene run): best-effort rescale of the illustrative
            # hand-authored library into the cube's normalized space.
            self.library.apply_normalization(normalization_strategy, normalization_params)
        self.retrieval = KnowledgeRetrieval(self.library, cfg.similarity_metric, cfg.top_k_materials)
        self.identifier = MaterialIdentifier(self.retrieval)
        self.kg = KnowledgeGraphBuilder(self.library, cfg.kg_similarity_threshold)
        self.semantic_builder = SemanticRepresentationBuilder(self.library, cfg.top_k_materials)

    def process(self, patch: Patch, fused_embedding: FusedEmbedding) -> KnowledgeEmbedding:
        p = patch.cube_data.shape[0] // 2
        center_spectrum = patch.cube_data[p, p, :]
        matches = self.retrieval.retrieve(center_spectrum)
        material_fractions = self.identifier.identify(patch.cube_data)
        kg_node_ids = self.kg.query_matched_node_ids(matches)
        return self.semantic_builder.build(
            patch.patch_id, fused_embedding.fused_vector, matches, kg_node_ids, material_fractions,
        )
