import pytest

from hsi_caption.datatypes import KnowledgeEmbedding, MaterialMatch, OntologySemanticRepresentation, \
    ReasonedKnowledge, StructuredFacts, Triplet
from hsi_caption.stage3_knowledge_engine.knowledge_graph import KnowledgeGraphBuilder
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary
from hsi_caption.stage5_ontology.ontology_loader import OntologyLoader
from hsi_caption.stage8_verification.evidence_retrieval import EvidenceRetriever
from hsi_caption.stage8_verification.fact_verification_engine import FactVerificationEngine
from hsi_caption.stage8_verification.kg_verification import KnowledgeGraphVerifier
from hsi_caption.stage8_verification.ontology_matching import OntologyVerificationMatcher
from hsi_caption.stage8_verification.rule_verification import RuleVerifier
from hsi_caption.stage8_verification.semantic_similarity import SemanticSimilarityVerifier


def _knowledge_embedding(fractions=None):
    return KnowledgeEmbedding(
        patch_id="p1",
        material_matches=[MaterialMatch("Rice", "vegetation", 0.9), MaterialMatch("Water", "water", 0.5)],
        kg_node_ids=["material::Rice", "material::Water"],
        semantic_vector=__import__("numpy").zeros(4),
        material_fractions=fractions or {"Rice": 0.6, "Water": 0.4},
    )


def _reasoned(violations=None):
    return ReasonedKnowledge(
        patch_id="p1", fired_rules=["R1"], contexts=["paddy_field_context"], suppressed_categories=[],
        constraint_violations=violations or [], reasoning_confidence=0.8, related_materials=["Water"],
    )


def _ontology_rep():
    return OntologySemanticRepresentation(
        patch_id="p1", matched_classes=["Rice", "Water"], class_hierarchy_paths={},
        refined_concepts=["Rice", "CerealCrop"], ontology_score=0.9,
    )


class TestEvidenceRetriever:
    def test_retrieve_finds_spectral_similarity(self):
        triplet = Triplet("p1", "hasMaterial", "Rice", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert evidence.spectral_similarity == pytest.approx(0.9)
        assert evidence.in_ontology is True

    def test_retrieve_unknown_material_has_no_similarity(self):
        triplet = Triplet("p1", "hasMaterial", "Unobtainium", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert evidence.spectral_similarity is None
        assert evidence.in_ontology is False


class TestKnowledgeGraphVerifier:
    def test_verify_scores_known_material_highly(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        kg = KnowledgeGraphBuilder(lib, cfg.knowledge.kg_similarity_threshold)
        verifier = KnowledgeGraphVerifier(kg)
        triplet = Triplet("p1", "hasMaterial", "Rice", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        score = verifier.verify(evidence)
        assert score == 1.0


class TestOntologyVerificationMatcher:
    def test_axiom_match_scores_one(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        verifier = OntologyVerificationMatcher(loader)
        triplet = Triplet("Rice", "requiresIrrigationFrom", "Water", 0.8)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert verifier.verify(evidence) == 1.0

    def test_unknown_object_scores_zero(self, cfg):
        loader = OntologyLoader(cfg.ontology.ontology_path)
        verifier = OntologyVerificationMatcher(loader)
        triplet = Triplet("p1", "hasMaterial", "Unobtainium", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert verifier.verify(evidence) == 0.0


class TestRuleVerifier:
    def test_violation_heavily_penalizes(self):
        verifier = RuleVerifier()
        triplet = Triplet("p1", "hasContext", "paddy_field_context", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(violations=["C1: x"]),
                                                 _ontology_rep())
        score = verifier.verify(evidence, _reasoned(violations=["C1: x"]))
        assert score == 0.1

    def test_rule_asserted_uses_reasoning_confidence(self):
        verifier = RuleVerifier()
        reasoned = _reasoned()
        triplet = Triplet("p1", "hasContext", "paddy_field_context", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), reasoned, _ontology_rep())
        assert verifier.verify(evidence, reasoned) == reasoned.reasoning_confidence


class TestSemanticSimilarityVerifier:
    def test_passthrough_similarity(self):
        verifier = SemanticSimilarityVerifier()
        triplet = Triplet("p1", "hasMaterial", "Rice", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert verifier.verify(evidence) == pytest.approx(0.9)

    def test_default_for_no_similarity(self):
        verifier = SemanticSimilarityVerifier()
        triplet = Triplet("p1", "hasContext", "paddy_field_context", 0.6)
        evidence = EvidenceRetriever().retrieve(triplet, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert verifier.verify(evidence) == 0.4


class TestFactVerificationEngineFacade:
    def test_process_splits_verified_and_rejected(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        kg = KnowledgeGraphBuilder(lib, cfg.knowledge.kg_similarity_threshold)
        ontology = OntologyLoader(cfg.ontology.ontology_path)
        engine = FactVerificationEngine(kg, ontology, cfg.verification)

        facts = StructuredFacts(
            patch_id="p1",
            triplets=[],
            ranked_facts=[
                Triplet("p1", "hasMaterial", "Rice", 0.9),
                Triplet("p1", "hasMaterial", "Unobtainium", 0.6),
            ],
        )
        result = engine.process(facts, _knowledge_embedding(), _reasoned(), _ontology_rep())
        assert any(t.object == "Rice" for t in result.verified)
        assert 0.0 <= result.verification_score <= 1.0
