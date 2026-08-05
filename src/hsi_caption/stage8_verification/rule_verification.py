"""
Stage 8 - Rule Verification
==============================
Purpose
    Re-check a triplet against Stage 4's fired rules and hard-constraint
    violations -- distinct from KG/ontology verification because it
    catches *logical* inconsistency (e.g. a fact asserted despite a
    fired constraint violation) rather than missing/incorrect entities.

Input
    `Evidence.rule_asserted` and Stage 4's `ReasonedKnowledge`
    (`constraint_violations`, `reasoning_confidence`).

Algorithm
    Score = `reasoning_confidence` if the triplet's object was directly
    rule-asserted; a neutral 0.5 baseline if not rule-asserted but no
    violations exist (rules simply didn't cover this fact, which isn't
    evidence against it); a heavily discounted score if any hard
    constraint was violated for this patch (regardless of which triplet,
    since a violated patch-level constraint casts doubt on all of that
    patch's symbolic facts). O(1) per triplet.
"""
from __future__ import annotations

import logging

from hsi_caption.datatypes import ReasonedKnowledge
from hsi_caption.stage8_verification.evidence_retrieval import Evidence

logger = logging.getLogger(__name__)


class RuleVerifier:
    """Verifies a triplet against Stage-4 rule firing and constraints."""

    def verify(self, evidence: Evidence, reasoned: ReasonedKnowledge) -> float:
        if reasoned.constraint_violations:
            return 0.1
        if evidence.rule_asserted:
            return reasoned.reasoning_confidence
        return 0.5
