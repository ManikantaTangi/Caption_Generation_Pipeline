"""
Stage 8 - Semantic Similarity
================================
Purpose
    Score a triplet's plausibility using the raw spectral-similarity
    evidence retrieved in Stage 3 -- the most direct, measurement-level
    check available: does the physical spectrum actually resemble the
    material this fact claims?

Input
    `Evidence.spectral_similarity` (already computed in Stage 3's
    KnowledgeRetrieval, carried through EvidenceRetrieval).

Algorithm
    Direct pass-through of the SAM/cosine/euclidean-derived similarity
    in [0, 1] when available (i.e. the triplet's object was one of the
    top-k retrieved materials); a conservative 0.4 default when the
    object wasn't in the top-k (not necessarily false, just unconfirmed
    by direct spectral evidence -- e.g. a `hasContext` triplet has no
    material spectrum to compare). O(1).
"""
from __future__ import annotations

import logging

from hsi_caption.stage8_verification.evidence_retrieval import Evidence

logger = logging.getLogger(__name__)


class SemanticSimilarityVerifier:
    """Verifies a triplet using Stage-3's direct spectral similarity evidence."""

    def verify(self, evidence: Evidence) -> float:
        if evidence.spectral_similarity is not None:
            return float(evidence.spectral_similarity)
        return 0.4
