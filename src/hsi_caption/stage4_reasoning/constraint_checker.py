"""
Stage 4 - Constraint Checker
===============================
Purpose
    Enforce domain invariants (configs/rules.yaml `constraints`) that
    must *never* be violated regardless of what the rule engine or
    upstream statistics say -- e.g. two mutually-exclusive land-cover
    types cannot both dominate a homogeneous patch. This is the
    trustworthy-AI safety net: a purely learned or rule-fired conclusion
    that breaks a hard constraint is flagged, not silently trusted.

Input
    `KnowledgeEmbedding.material_fractions` (Stage 3 output).

Algorithm
    Direct evaluation of each typed constraint (`mutual_exclusion`,
    `sum_to_one`) against the fraction dict. O(C) per patch, C =
    number of constraints (small, fixed).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """Validates material fractions against hard domain constraints."""

    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self.constraints = raw.get("constraints", [])
        logger.info("Loaded %d hard constraints.", len(self.constraints))

    def check(self, material_fractions: Dict[str, float]) -> List[str]:
        violations: List[str] = []
        for c in self.constraints:
            if c["type"] == "mutual_exclusion":
                fracs = [material_fractions.get(m, 0.0) for m in c["materials"]]
                if all(f > c["max_joint_fraction"] for f in fracs):
                    violations.append(f"{c['id']}: {c['description']}")
            elif c["type"] == "sum_to_one":
                total = sum(material_fractions.values())
                if abs(total - 1.0) > c["tolerance"]:
                    violations.append(f"{c['id']}: fractions sum to {total:.4f}, expected 1.0")
        if violations:
            logger.warning("Constraint violations: %s", violations)
        return violations
