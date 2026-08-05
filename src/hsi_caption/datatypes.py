"""
Pipeline-wide data contracts.

Every stage in the pipeline communicates through these explicit,
strongly-typed dataclasses rather than loose dicts/tuples. This is the
backbone that makes "output of stage N == input of stage N+1" a checkable
property (see tests/test_integration.py) instead of an informal convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Stage 1 outputs
# --------------------------------------------------------------------------- #
@dataclass
class HSICube:
    """A raw or normalized hyperspectral image cube."""
    data: np.ndarray                      # (H, W, B)
    labels: Optional[np.ndarray]          # (H, W) integer class ids, or None
    wavelengths_nm: np.ndarray            # (B,)
    class_names: List[str]
    name: str = "unnamed_cube"

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def num_bands(self) -> int:
        return self.data.shape[2]


@dataclass
class Patch:
    """A single spatial-spectral patch extracted from an HSICube."""
    patch_id: str
    cube_data: np.ndarray                 # (patch_size, patch_size, B)
    center_label: int
    label_patch: np.ndarray               # (patch_size, patch_size)
    center_row: int
    center_col: int
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class PatchDataset:
    """Train/val/test split of patches produced by Stage 1."""
    train: List[Patch]
    val: List[Patch]
    test: List[Patch]
    wavelengths_nm: np.ndarray
    class_names: List[str]
    normalization_stats: Dict[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Stage 2 outputs
# --------------------------------------------------------------------------- #
@dataclass
class FusedEmbedding:
    """Unified feature embedding produced by Stage 2 for one patch."""
    patch_id: str
    spectral_vector: np.ndarray           # (spectral_embed_dim,)
    spatial_vector: np.ndarray            # (spatial_embed_dim,)
    metadata_vector: np.ndarray           # (metadata_embed_dim,)
    fused_vector: np.ndarray              # (fused_embed_dim,)
    attention_weights: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Stage 3 outputs
# --------------------------------------------------------------------------- #
@dataclass
class MaterialMatch:
    material_name: str
    category: str
    similarity: float


@dataclass
class KnowledgeEmbedding:
    patch_id: str
    material_matches: List[MaterialMatch]
    kg_node_ids: List[str]
    semantic_vector: np.ndarray
    material_fractions: Dict[str, float]  # ground-truth-derived, for rule engine


# --------------------------------------------------------------------------- #
# Stage 4 outputs
# --------------------------------------------------------------------------- #
@dataclass
class ReasonedKnowledge:
    patch_id: str
    fired_rules: List[str]
    contexts: List[str]
    suppressed_categories: List[str]
    constraint_violations: List[str]
    reasoning_confidence: float
    related_materials: List[str] = field(default_factory=list)
    fired_rule_descriptions: List[str] = field(default_factory=list)
    fired_rule_descriptions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Stage 5 outputs
# --------------------------------------------------------------------------- #
@dataclass
class OntologySemanticRepresentation:
    patch_id: str
    matched_classes: List[str]
    class_hierarchy_paths: Dict[str, List[str]]
    refined_concepts: List[str]
    ontology_score: float


# --------------------------------------------------------------------------- #
# Stage 6 outputs
# --------------------------------------------------------------------------- #
@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str
    score: float


@dataclass
class StructuredFacts:
    patch_id: str
    triplets: List[Triplet]
    ranked_facts: List[Triplet]


# --------------------------------------------------------------------------- #
# Stage 7 outputs
# --------------------------------------------------------------------------- #
@dataclass
class UncertaintyEstimate:
    patch_id: str
    class_probs_mean: np.ndarray
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    predictive_entropy: float
    confidence_score: float
    calibrated: bool = False


# --------------------------------------------------------------------------- #
# Stage 8 outputs
# --------------------------------------------------------------------------- #
@dataclass
class VerifiedFacts:
    patch_id: str
    verified: List[Triplet]
    rejected: List[Triplet]
    verification_score: float
    fact_confidence: Dict[str, float]


# --------------------------------------------------------------------------- #
# Stage 9 outputs
# --------------------------------------------------------------------------- #
@dataclass
class CaptionResult:
    patch_id: str
    caption: str
    confidence_score: float
    confidence_band: str
    explanation: str
