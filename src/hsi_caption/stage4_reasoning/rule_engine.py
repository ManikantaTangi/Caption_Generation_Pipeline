"""
Stage 4 - Rule Engine
========================
Purpose
    Apply hand-authored domain rules (configs/rules.yaml) to a patch's
    Stage-3 material composition to derive symbolic *context* facts that
    no purely statistical model would produce reliably from a single
    patch (e.g. "paddy-field context" from a Rice+Water co-occurrence
    pattern). This is the neuro-symbolic bridge: numeric fractions in,
    symbolic propositions out.

Input
    `KnowledgeEmbedding.material_fractions` (Stage 3 output) and the
    patch's `label_patch` (for the entropy-based rule).

Algorithm
    Forward-chaining rule evaluation: for each rule, evaluate its
    `condition` against the material-fraction dict (and patch entropy);
    if satisfied, assert `conclusion` with the rule's static confidence.
    Forward chaining (data -> conclusions) is preferred over backward
    chaining here because we have no specific goal to prove -- we want
    *all* applicable context, which is exactly what forward chaining
    naturally enumerates in one pass. Complexity: O(R) per patch, R =
    number of rules (small, fixed).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class FiredRule:
    rule_id: str
    conclusion: Dict
    confidence: float
    description: str = ""
    description: str = ""


class RuleEngine:
    """Forward-chaining rule evaluator over material-fraction facts."""

    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self.rules = raw.get("rules", [])
        logger.info("Loaded %d reasoning rules.", len(self.rules))

    @staticmethod
    def _patch_entropy(label_patch: np.ndarray) -> float:
        values, counts = np.unique(label_patch, return_counts=True)
        probs = counts / counts.sum()
        return float(-(probs * np.log2(probs + 1e-12)).sum())

    def evaluate(self, material_fractions: Dict[str, float], label_patch: np.ndarray) -> List[FiredRule]:
        entropy = self._patch_entropy(label_patch)
        fired: List[FiredRule] = []
        for rule in self.rules:
            cond = rule["condition"]
            satisfied = True

            if "material" in cond:
                satisfied &= material_fractions.get(cond["material"], 0.0) >= cond.get("min_fraction", 0.0)
            if "materials_any" in cond:
                satisfied &= any(
                    material_fractions.get(m, 0.0) >= cond.get("min_fraction", 0.0)
                    for m in cond["materials_any"]
                )
            if "patch_entropy_gt" in cond:
                satisfied &= entropy > cond["patch_entropy_gt"]

            if satisfied:
                fired.append(FiredRule(rule_id=rule["id"], conclusion=rule["conclusion"],
                                        confidence=float(rule["confidence"]),
                                        description=rule.get("description", "")))
        logger.debug("Rule engine fired %d/%d rules (patch entropy=%.3f)", len(fired), len(self.rules), entropy)
        return fired
