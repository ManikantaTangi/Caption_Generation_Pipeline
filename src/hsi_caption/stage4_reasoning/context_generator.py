"""
Stage 4 - Context Generator
==============================
Purpose
    Aggregate the (possibly several) fired rules' conclusions into two
    clean lists -- symbolic `contexts` (for Stage 9's caption/explanation
    text) and `suppressed_categories` (for Stage 8's verification, which
    should down-weight facts about a suppressed category). Also carries
    forward each conclusion's `related_material`, used by Stage 6's
    RelationshipGenerator to seed candidate triples.

Input
    `List[FiredRule]` from the RuleEngine (this stage).

Algorithm
    Single pass over fired rules, dict-key dispatch on each conclusion's
    fields (`context`, `suppress_category`, `related_material`).
    O(R_fired).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from hsi_caption.stage4_reasoning.rule_engine import FiredRule

logger = logging.getLogger(__name__)


@dataclass
class GeneratedContext:
    contexts: List[str] = field(default_factory=list)
    suppressed_categories: List[str] = field(default_factory=list)
    related_materials: List[str] = field(default_factory=list)


class ContextGenerator:
    """Extracts symbolic context tags from fired rule conclusions."""

    def generate(self, fired_rules: List[FiredRule]) -> GeneratedContext:
        result = GeneratedContext()
        for rule in fired_rules:
            c = rule.conclusion
            if "context" in c:
                result.contexts.append(c["context"])
            if "suppress_category" in c:
                result.suppressed_categories.append(c["suppress_category"])
            if "related_material" in c:
                result.related_materials.append(c["related_material"])
        return result
