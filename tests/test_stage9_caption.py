import pytest

from hsi_caption.datatypes import ReasonedKnowledge, Triplet, UncertaintyEstimate, VerifiedFacts
from hsi_caption.stage9_caption.caption_engine import CaptionEngine
from hsi_caption.stage9_caption.caption_generator import CaptionGenerator
from hsi_caption.stage9_caption.caption_refiner import CaptionRefiner
from hsi_caption.stage9_caption.explanation_generator import ExplanationGenerator
from hsi_caption.stage9_caption.language_controller import LanguageController
from hsi_caption.stage9_caption.template_selector import TemplateSelector


def _uncertainty(score):
    import numpy as np
    return UncertaintyEstimate(
        patch_id="p1", class_probs_mean=np.array([score, 1 - score]),
        epistemic_uncertainty=0.1, aleatoric_uncertainty=0.4, predictive_entropy=0.3,
        confidence_score=score, calibrated=True,
    )


def _reasoned(fired_rule_descriptions=None, constraint_violations=None):
    return ReasonedKnowledge(
        patch_id="p1", fired_rules=["R1"], contexts=[], suppressed_categories=[],
        constraint_violations=constraint_violations or [], reasoning_confidence=0.8,
        fired_rule_descriptions=fired_rule_descriptions or [],
    )


class TestTemplateSelector:
    def test_band_boundaries(self, cfg):
        selector = TemplateSelector(cfg.caption.template_bank_path, cfg.caption.confidence_bands)
        assert selector.band_for(0.9) == "high"
        assert selector.band_for(0.5) == "medium"
        assert selector.band_for(0.1) == "low"

    def test_select_returns_valid_template(self, cfg):
        selector = TemplateSelector(cfg.caption.template_bank_path, cfg.caption.confidence_bands)
        band, template = selector.select(0.9)
        assert band == "high"
        assert "{dominant_material}" in template


class TestCaptionGenerator:
    def test_fills_dominant_material(self):
        gen = CaptionGenerator()
        verified = [Triplet("p1", "hasMaterial", "Rice", 0.9), Triplet("p1", "hasMaterial", "Water", 0.4)]
        caption = gen.generate("Dominant: {dominant_material}, secondary: {secondary_material}", verified, 0.8)
        assert "Rice" in caption
        assert "Water" in caption

    def test_handles_empty_facts_gracefully(self):
        gen = CaptionGenerator()
        caption = gen.generate("Material: {dominant_material}", [], 0.1)
        assert "unidentified" in caption


class TestLanguageController:
    def test_low_band_hedges_language(self):
        controller = LanguageController()
        result = controller.control("This is Rice and confirms paddy context.", "low")
        assert "may be" in result
        assert "suggests" in result

    def test_high_band_unchanged(self):
        controller = LanguageController()
        text = "This is Rice."
        assert controller.control(text, "high") == text


class TestCaptionRefiner:
    def test_truncates_long_caption(self):
        refiner = CaptionRefiner(max_caption_length=5)
        long_caption = " ".join(["word"] * 20)
        result = refiner.refine(long_caption)
        assert result.endswith("...")
        assert len(result.split(" ")) <= 6

    def test_adds_terminal_punctuation(self):
        refiner = CaptionRefiner(max_caption_length=50)
        result = refiner.refine("this has no period")
        assert result.endswith(".")

    def test_capitalises_first_letter(self):
        refiner = CaptionRefiner(max_caption_length=50)
        result = refiner.refine("lowercase start.")
        assert result[0] == "L"


class TestExplanationGenerator:
    def test_generate_mentions_verification_counts_naturally(self):
        verified_facts = VerifiedFacts(
            patch_id="p1", verified=[Triplet("p1", "hasMaterial", "Rice", 0.9)],
            rejected=[Triplet("p1", "hasMaterial", "X", 0.1)],
            verification_score=0.6, fact_confidence={},
        )
        explanation = ExplanationGenerator().generate(verified_facts, _uncertainty(0.7), _reasoned())
        assert "1 of 2 candidate facts were independently verified" in explanation
        assert "verification score 0.60" in explanation

    def test_never_leaks_raw_rule_ids(self):
        """Regression test: explanations must describe observed evidence in
        plain language, never cite an internal rule ID like 'R3'."""
        verified_facts = VerifiedFacts(patch_id="p1", verified=[], rejected=[], verification_score=0.5,
                                        fact_confidence={})
        reasoned = _reasoned(fired_rule_descriptions=[
            "Rice is frequently adjacent to standing water in paddy fields."
        ])
        explanation = ExplanationGenerator().generate(verified_facts, _uncertainty(0.7), reasoned)
        assert "R1" not in explanation
        assert "Rice is frequently adjacent to standing water in paddy fields." in explanation

    def test_includes_classifier_prediction_when_class_names_given(self):
        verified_facts = VerifiedFacts(patch_id="p1", verified=[], rejected=[], verification_score=0.5,
                                        fact_confidence={})
        explanation = ExplanationGenerator(class_names=["Rice", "Water"]).generate(
            verified_facts, _uncertainty(0.9), _reasoned())
        assert "trained classifier identified this patch as most consistent with Rice" in explanation
        assert "90%" in explanation

    def test_includes_spectral_evidence_when_knowledge_embedding_given(self):
        import numpy as np
        from hsi_caption.datatypes import KnowledgeEmbedding, MaterialMatch
        verified_facts = VerifiedFacts(patch_id="p1", verified=[], rejected=[], verification_score=0.5,
                                        fact_confidence={})
        ke = KnowledgeEmbedding(
            patch_id="p1", material_matches=[MaterialMatch("Rice", "vegetation", 0.95)],
            kg_node_ids=[], semantic_vector=np.zeros(1), material_fractions={"Rice": 0.8},
        )
        explanation = ExplanationGenerator().generate(verified_facts, _uncertainty(0.7), _reasoned(),
                                                        knowledge_embedding=ke)
        assert "Rice reference spectrum" in explanation
        assert "95% similarity" in explanation
        assert "80% of the patch's pixels" in explanation

    def test_constraint_violation_produces_plain_caution_sentence(self):
        verified_facts = VerifiedFacts(patch_id="p1", verified=[], rejected=[], verification_score=0.5,
                                        fact_confidence={})
        reasoned = _reasoned(constraint_violations=["C1: Water and Roads/houses cannot jointly dominate"])
        explanation = ExplanationGenerator().generate(verified_facts, _uncertainty(0.7), reasoned)
        assert "treated cautiously" in explanation
        assert "Water and Roads/houses cannot jointly dominate" in explanation


class TestCaptionEngineFacade:
    def test_process_end_to_end(self, cfg):
        engine = CaptionEngine(cfg.caption)
        verified_facts = VerifiedFacts(
            patch_id="p1", verified=[Triplet("p1", "hasMaterial", "Rice", 0.9)],
            rejected=[], verification_score=0.9, fact_confidence={},
        )
        result = engine.process("p1", verified_facts, _uncertainty(0.85), _reasoned())
        assert result.confidence_band == "high"
        assert "Rice" in result.caption
        assert result.caption.endswith(".")

    def test_process_with_class_names_produces_natural_explanation(self, cfg):
        engine = CaptionEngine(cfg.caption, class_names=["Rice", "Water"])
        verified_facts = VerifiedFacts(
            patch_id="p1", verified=[Triplet("p1", "hasMaterial", "Rice", 0.9)],
            rejected=[], verification_score=0.9, fact_confidence={},
        )
        reasoned = _reasoned(fired_rule_descriptions=[
            "Rice is frequently adjacent to standing water in paddy fields."
        ])
        result = engine.process("p1", verified_facts, _uncertainty(0.9), reasoned)
        assert "R1" not in result.explanation
        assert "trained classifier identified" in result.explanation
        assert "paddy fields" in result.explanation
