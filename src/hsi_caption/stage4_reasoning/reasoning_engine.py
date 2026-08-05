"""
Stage 4 - Reasoning Engine
=============================
Purpose
    Combine the RuleEngine's fired rules, the ConstraintChecker's
    violations, and the ContextGenerator's aggregated context into the
    single `ReasonedKnowledge` object that is Stage 4's contractual
    output -- the "Reasoned Knowledge" consumed by Stage 5 (ontology
    matching uses `contexts`), Stage 6 (fact generation uses
    `related_materials`/`contexts`), and Stage 8 (rule verification
    re-checks `fired_rules`).

Input
    Stage 3's `KnowledgeEmbedding.material_fractions` and the patch's
    `label_patch`.

Algorithm
    reasoning_confidence = mean(fired rule confidences) if any rules
    fired, discounted multiplicatively by `contradiction_penalty` for
    every hard-constraint violation:
        conf = mean(rule_confidences) * (1 - penalty)^(#violations)
    This keeps the score in [0, 1] while making violations strictly
    costly regardless of how confident the individual rules were --
    a deliberately conservative (trustworthy-AI) design choice: a
    hard-constraint violation should never be masked by high rule
    confidence elsewhere. Complexity: O(R_fired + C).
"""
from __future__ import annotations

import logging

import numpy as np

from hsi_caption.datatypes import ReasonedKnowledge
from hsi_caption.stage4_reasoning.constraint_checker import ConstraintChecker
from hsi_caption.stage4_reasoning.context_generator import ContextGenerator
from hsi_caption.stage4_reasoning.rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Facade: fires rules, checks constraints, and scores overall confidence."""

    def __init__(self, rules_path: str, contradiction_penalty: float) -> None:
        self.rule_engine = RuleEngine(rules_path)
        self.constraint_checker = ConstraintChecker(rules_path)
        self.context_generator = ContextGenerator()
        self.contradiction_penalty = contradiction_penalty

    def reason(self, patch_id: str, material_fractions: dict, label_patch: np.ndarray) -> ReasonedKnowledge:
        fired = self.rule_engine.evaluate(material_fractions, label_patch)
        violations = self.constraint_checker.check(material_fractions)
        ctx = self.context_generator.generate(fired)

        base_conf = float(np.mean([r.confidence for r in fired])) if fired else 0.5
        penalty_factor = (1.0 - self.contradiction_penalty) ** len(violations)
        reasoning_confidence = float(np.clip(base_conf * penalty_factor, 0.0, 1.0))

        return ReasonedKnowledge(
            patch_id=patch_id,
            fired_rules=[r.rule_id for r in fired],
            contexts=ctx.contexts,
            suppressed_categories=ctx.suppressed_categories,
            constraint_violations=violations,
            reasoning_confidence=reasoning_confidence,
            related_materials=ctx.related_materials,
            fired_rule_descriptions=[r.description for r in fired if r.description],
        )
