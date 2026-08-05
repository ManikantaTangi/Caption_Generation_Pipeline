import pytest

from hsi_caption.datatypes import ReasonedKnowledge
from hsi_caption.stage5_ontology.ontology_engine import OntologyEngine
from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage5_ontology.ontology_matcher import OntologyMatcher
from hsi_caption.stage5_ontology.ontology_refiner import OntologyRefiner
from hsi_caption.stage5_ontology.semantic_generator import SemanticGenerator


class TestOntologyLoader:
    def test_hierarchy_loaded(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        assert "Rice" in loader.hierarchy
        assert "CerealCrop" in loader.hierarchy

    def test_superclasses_chain(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        chain = loader.superclasses("Rice")
        assert "CerealCrop" in chain
        assert "Crop" in chain
        assert "Vegetation" in chain

    def test_is_subclass_of(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        assert loader.is_subclass_of("Rice", "Vegetation")
        assert not loader.is_subclass_of("Water", "Vegetation")

    def test_axiom_lookup(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        assert loader.has_axiom("Rice", "requiresIrrigationFrom", "Water")
        assert not loader.has_axiom("Corn", "requiresIrrigationFrom", "Water")


class TestOntologyMatcher:
    def test_match_returns_known_materials(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        matcher = OntologyMatcher(loader, cfg.ontology.match_threshold)
        fractions = {"Rice": 0.6, "Water": 0.4}
        matches = matcher.match(fractions)
        names = [m.material_name for m in matches]
        assert "Rice" in names and "Water" in names

    def test_coherence_score_high_for_single_branch(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        matcher = OntologyMatcher(loader, cfg.ontology.match_threshold)
        fractions = {"Rice": 0.5, "Corn": 0.5}  # both Vegetation branch
        matches = matcher.match(fractions)
        score = matcher.coherence_score(matches, fractions)
        assert score == pytest.approx(1.0)

    def test_coherence_score_lower_for_mixed_branch(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        matcher = OntologyMatcher(loader, cfg.ontology.match_threshold)
        fractions = {"Rice": 0.5, "Water": 0.5}  # different branches
        matches = matcher.match(fractions)
        score = matcher.coherence_score(matches, fractions)
        assert score < 1.0


class TestSemanticGenerator:
    def test_generate_includes_leaf_and_parent(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        matcher = OntologyMatcher(loader, cfg.ontology.match_threshold)
        matches = matcher.match({"Rice": 1.0})
        concepts = SemanticGenerator().generate(matches)
        assert "Rice" in concepts
        assert "CerealCrop" in concepts

    def test_empty_matches_returns_empty(self):
        assert SemanticGenerator().generate([]) == []


class TestOntologyRefiner:
    def test_suppressed_branch_removed(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        matcher = OntologyMatcher(loader, cfg.ontology.match_threshold)
        refiner = OntologyRefiner(loader, cfg.ontology.refine_iterations)
        fractions = {"Rice": 1.0}
        matches = matcher.match(fractions)
        concepts = SemanticGenerator().generate(matches)
        result = refiner.refine("p1", concepts, ["vegetation"], 1.0, matches)
        assert "Rice" not in result.refined_concepts


class TestOntologyEngineFacade:
    def test_process_end_to_end(self, cfg):
        engine = OntologyEngine(cfg.ontology)
        reasoned = ReasonedKnowledge(
            patch_id="p1", fired_rules=[], contexts=[], suppressed_categories=[],
            constraint_violations=[], reasoning_confidence=0.9,
        )
        fractions = {"Rice": 0.6, "Water": 0.4}
        result = engine.process("p1", fractions, reasoned)
        assert result.patch_id == "p1"
        assert 0.0 <= result.ontology_score <= 1.0
