import pytest

from hsi_caption.datatypes import OntologySemanticRepresentation, ReasonedKnowledge
from hsi_caption.stage6_facts.fact_extraction import FactExtractor
from hsi_caption.stage6_facts.fact_ranking import FactRanking
from hsi_caption.stage6_facts.relationship_generator import RelationshipGenerator
from hsi_caption.stage6_facts.structured_fact_generator import StructuredFactGenerator
from hsi_caption.stage6_facts.triplet_generator import TripletGenerator


def _reasoned(contexts=None, related=None, confidence=0.8):
    return ReasonedKnowledge(
        patch_id="p1", fired_rules=["R1"], contexts=contexts or [], suppressed_categories=[],
        constraint_violations=[], reasoning_confidence=confidence, related_materials=related or [],
    )


def _ontology_rep(concepts=None):
    return OntologySemanticRepresentation(
        patch_id="p1", matched_classes=concepts or [], class_hierarchy_paths={},
        refined_concepts=concepts or [], ontology_score=0.9,
    )


class TestFactExtractor:
    def test_extracts_material_facts_above_threshold(self):
        extractor = FactExtractor(min_fact_score=0.3)
        fractions = {"Rice": 0.6, "Water": 0.1}
        facts = extractor.extract("p1", fractions, _reasoned(), _ontology_rep())
        objects = [f.object for f in facts]
        assert "Rice" in objects
        assert "Water" not in objects  # below threshold

    def test_extracts_context_and_concept_facts(self):
        extractor = FactExtractor(min_fact_score=0.3)
        facts = extractor.extract(
            "p1", {"Rice": 0.9}, _reasoned(contexts=["paddy_field_context"]),
            _ontology_rep(concepts=["Rice", "CerealCrop"]),
        )
        predicates = {f.predicate for f in facts}
        assert {"hasMaterial", "hasContext", "hasConcept"}.issubset(predicates)


class TestTripletGenerator:
    def test_generate_preserves_count(self):
        extractor = FactExtractor(min_fact_score=0.3)
        facts = extractor.extract("p1", {"Rice": 0.9}, _reasoned(), _ontology_rep())
        triplets = TripletGenerator().generate(facts)
        assert len(triplets) == len(facts)


class TestRelationshipGenerator:
    def test_generates_co_occurrence_above_floor(self):
        gen = RelationshipGenerator(relation_confidence_floor=0.2)
        fractions = {"Rice": 0.5, "Water": 0.3}
        reasoned = _reasoned(related=["Water"])
        triplets = gen.generate("Rice", fractions, reasoned)
        assert any(t.predicate == "coOccursWith" and t.object == "Water" for t in triplets)

    def test_skips_below_floor(self):
        gen = RelationshipGenerator(relation_confidence_floor=0.5)
        fractions = {"Rice": 0.5, "Water": 0.1}
        reasoned = _reasoned(related=["Water"])
        triplets = gen.generate("Rice", fractions, reasoned)
        assert triplets == []


class TestFactRanking:
    def test_deduplicates_and_sorts(self):
        from hsi_caption.datatypes import Triplet
        ranking = FactRanking(min_fact_score=0.0, max_facts_per_patch=10)
        triplets = [
            Triplet("p1", "hasMaterial", "Rice", 0.5),
            Triplet("p1", "hasMaterial", "Rice", 0.9),  # duplicate, higher score should win
            Triplet("p1", "hasMaterial", "Water", 0.3),
        ]
        result = ranking.rank("p1", triplets)
        rice_scores = [t.score for t in result.ranked_facts if t.object == "Rice"]
        assert rice_scores == [0.9]
        assert result.ranked_facts[0].score >= result.ranked_facts[-1].score

    def test_truncates_to_max_facts(self):
        from hsi_caption.datatypes import Triplet
        ranking = FactRanking(min_fact_score=0.0, max_facts_per_patch=2)
        triplets = [Triplet("p1", "hasMaterial", f"M{i}", i / 10) for i in range(10)]
        result = ranking.rank("p1", triplets)
        assert len(result.ranked_facts) == 2


class TestStructuredFactGeneratorFacade:
    def test_process_end_to_end(self, cfg):
        generator = StructuredFactGenerator(cfg.fact)
        fractions = {name: 0.0 for name in cfg.dataset.class_names}
        fractions["Rice"] = 0.6
        fractions["Water"] = 0.4
        reasoned = _reasoned(contexts=["paddy_field_context"], related=["Water"])
        ontology_rep = _ontology_rep(concepts=["Rice", "CerealCrop"])
        result = generator.process("p1", fractions, reasoned, ontology_rep)
        assert result.patch_id == "p1"
        assert len(result.ranked_facts) <= cfg.fact.max_facts_per_patch
