import numpy as np
import pytest

from hsi_caption.stage4_reasoning.constraint_checker import ConstraintChecker
from hsi_caption.stage4_reasoning.context_generator import ContextGenerator
from hsi_caption.stage4_reasoning.reasoning_engine import ReasoningEngine
from hsi_caption.stage4_reasoning.rule_engine import RuleEngine


class TestRuleEngine:
    def test_rice_water_rule_fires(self, cfg):
        engine = RuleEngine(cfg.reasoning.rules_path)
        fractions = {"Rice": 0.5, "Water": 0.3, "Corn": 0.2}
        label_patch = np.zeros((5, 5), dtype=np.int32)
        fired = engine.evaluate(fractions, label_patch)
        assert any(r.rule_id == "R1" for r in fired)

    def test_no_rules_fire_for_flat_uniform_fractions_below_thresholds(self, cfg):
        engine = RuleEngine(cfg.reasoning.rules_path)
        fractions = {name: 0.0 for name in cfg.dataset.class_names}
        fractions["Background"] = 1.0
        label_patch = np.zeros((3, 3), dtype=np.int32)  # zero entropy, single class
        fired = engine.evaluate(fractions, label_patch)
        assert isinstance(fired, list)

    def test_entropy_rule_fires_on_heterogeneous_patch(self, cfg):
        engine = RuleEngine(cfg.reasoning.rules_path)
        fractions = {name: 1.0 / len(cfg.dataset.class_names) for name in cfg.dataset.class_names}
        label_patch = np.arange(9).reshape(3, 3) % len(cfg.dataset.class_names)
        fired = engine.evaluate(fractions, label_patch)
        assert any(r.rule_id == "R5" for r in fired)


class TestConstraintChecker:
    def test_mutual_exclusion_violation_detected(self, cfg):
        checker = ConstraintChecker(cfg.reasoning.rules_path)
        fractions = {"Water": 0.5, "Roads and houses": 0.5}
        violations = checker.check(fractions)
        assert any("C1" in v for v in violations)

    def test_sum_to_one_violation_detected(self, cfg):
        checker = ConstraintChecker(cfg.reasoning.rules_path)
        fractions = {"Water": 0.5, "Corn": 0.2}  # sums to 0.7
        violations = checker.check(fractions)
        assert any("C2" in v for v in violations)

    def test_valid_fractions_pass(self, cfg):
        checker = ConstraintChecker(cfg.reasoning.rules_path)
        fractions = {name: 0.0 for name in cfg.dataset.class_names}
        fractions["Corn"] = 1.0
        violations = checker.check(fractions)
        assert violations == []


class TestContextGenerator:
    def test_generate_extracts_context_and_suppression(self):
        from hsi_caption.stage4_reasoning.rule_engine import FiredRule
        fired = [
            FiredRule("R1", {"context": "paddy_field_context", "related_material": "Water"}, 0.8),
            FiredRule("R2", {"context": "urban_context", "suppress_category": "vegetation"}, 0.7),
        ]
        result = ContextGenerator().generate(fired)
        assert "paddy_field_context" in result.contexts
        assert "urban_context" in result.contexts
        assert "vegetation" in result.suppressed_categories
        assert "Water" in result.related_materials


class TestReasoningEngine:
    def test_confidence_in_bounds(self, cfg):
        engine = ReasoningEngine(cfg.reasoning.rules_path, cfg.reasoning.contradiction_penalty)
        fractions = {name: 0.0 for name in cfg.dataset.class_names}
        fractions["Rice"] = 0.5
        fractions["Water"] = 0.5
        label_patch = np.zeros((5, 5), dtype=np.int32)
        result = engine.reason("p1", fractions, label_patch)
        assert 0.0 <= result.reasoning_confidence <= 1.0

    def test_violation_penalizes_confidence(self, cfg):
        engine = ReasoningEngine(cfg.reasoning.rules_path, cfg.reasoning.contradiction_penalty)
        good_fractions = {name: 0.0 for name in cfg.dataset.class_names}
        good_fractions["Rice"] = 1.0
        bad_fractions = {name: 0.0 for name in cfg.dataset.class_names}
        bad_fractions["Water"] = 0.5
        bad_fractions["Roads and houses"] = 0.5
        label_patch = np.zeros((5, 5), dtype=np.int32)
        good = engine.reason("p1", good_fractions, label_patch)
        bad = engine.reason("p2", bad_fractions, label_patch)
        assert bad.constraint_violations
        assert bad.reasoning_confidence < 0.5
