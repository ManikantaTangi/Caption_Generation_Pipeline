"""
Stage 8 - Ontology Matching (verification)
==============================================
Purpose
    Score a triplet's plausibility against the formal ontology's class
    hierarchy and axioms (Stage 5) -- e.g. reject a `requiresIrrigationFrom`
    fact for a material that is not, ontologically, a Crop, or confirm a
    fact that matches a declared axiom exactly.

Input
    `Evidence.in_ontology` and the `OntologyLoader` (Stage 5) for axiom
    and subclass checks.

Algorithm
    1.0 if the triplet exactly matches a declared axiom; 0.8 if the
    object is a valid ontology class *and* the predicate's domain/range
    constraints (where defined) are satisfied by the classes involved;
    0.3 if the object is a known class but the relation is semantically
    unsupported; 0.0 if the object is not a recognised ontology class at
    all. Complexity: O(1)-O(depth of hierarchy) per check.
"""
from __future__ import annotations

import logging

from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage8_verification.evidence_retrieval import Evidence

logger = logging.getLogger(__name__)


class OntologyVerificationMatcher:
    """Verifies a triplet against the Stage-5 ontology's axioms/hierarchy."""

    def __init__(self, ontology: OntologyLoader) -> None:
        self.ontology = ontology

    def verify(self, evidence: Evidence) -> float:
        triplet = evidence.triplet
        if self.ontology.has_axiom(triplet.subject, triplet.predicate, triplet.object):
            return 1.0
        if not evidence.in_ontology:
            return 0.0
        if triplet.predicate in self.ontology.object_properties:
            prop = self.ontology.object_properties[triplet.predicate]
            domain_ok = self.ontology.is_subclass_of(triplet.subject, prop.domain) or triplet.subject not in self.ontology.hierarchy
            range_ok = self.ontology.is_subclass_of(triplet.object, prop.range)
            return 0.8 if (domain_ok and range_ok) else 0.3
        return 0.8  # e.g. hasConcept/hasMaterial: informal predicates, object recognised = good signal
