# Stage 8 Module Report — Fact Verification

### Objective
Independently cross-check every Stage 6 candidate triplet against four evidence sources before it may appear
in a caption.

### Inputs
Stage 6's `StructuredFacts.ranked_facts`, Stage 3's `KnowledgeEmbedding`, Stage 4's `ReasonedKnowledge`, Stage
5's `OntologySemanticRepresentation`.

### Internal Modules
EvidenceRetriever, KnowledgeGraphVerifier, OntologyVerificationMatcher, RuleVerifier,
SemanticSimilarityVerifier, VerificationConfidenceFusion, FactVerificationEngine (facade).

### Methods Used
Cross-stage evidence lookup; four independent [0,1] scoring functions; weighted linear score fusion;
threshold-based verified/rejected split.

### Algorithms Used
Weighted linear fusion (chosen over a learned classifier for full decomposability/explainability — see
ARCHITECTURE.md rationale).

### AI Paradigms Used
Explainable AI (fully decomposable per-source score), Trustworthy AI (independent multi-source cross-checking),
Neuro-Symbolic AI (semantic-similarity is measured; KG/ontology/rule checks are symbolic).

### Outputs
`VerifiedFacts` (`verified`, `rejected`, `verification_score`, `fact_confidence`).

### Output Interpretation
`fact_confidence` maps each `(s,p,o)` key to its fused score, making every accept/reject decision fully
auditable against its four named components — exactly the trail Stage 9's `ExplanationGenerator` and a thesis
reviewer both need.

### Module Significance
This is the pipeline's core trustworthy-AI checkpoint: no fact reaches the user-facing caption without
surviving four independent checks, guarding against hallucinated or spuriously-ranked Stage 6 output.

### Advantages
Fully explainable per-component scoring; conservative rule-violation handling (heavy penalty regardless of
which triplet, once a patch-level constraint is violated); configurable weights validated to sum to 1.0 at
startup.

### Limitations
Static, hand-set weights (no learned re-ranking, by design — no labelled caption-quality data yet exists);
axiom coverage in the ontology is small (2 axioms), so `OntologyVerificationMatcher`'s highest-confidence path
(exact axiom match) rarely fires on synthetic data.

### Future Improvements
Expand ontology axioms for richer verification coverage; learn calibrated fusion weights once caption-quality
labels exist; add a temporal/spatial-consistency verifier across neighbouring patches.

### Module Workflow
```
Triplet --> EvidenceRetriever.retrieve --> Evidence (spectral_sim, in_ontology, rule_asserted, in_kg)
Evidence --> {KnowledgeGraphVerifier, OntologyVerificationMatcher, RuleVerifier, SemanticSimilarityVerifier}
          --> four [0,1] scores
four scores --> VerificationConfidenceFusion.fuse (weighted sum) --> verified / rejected + verification_score
```

### Demo Output
```
4 fact(s) verified, 4 rejected (verification score=0.60)
```

### Unit Test Output
```
tests/test_stage8_verification.py ...........  [ 11 passed ]
```

### Summary
Stage 8 is the pipeline's independent, fully explainable trust gate — every fact Stage 9 can caption from has
survived four separately-scored, weighted checks against knowledge graph, ontology, rule, and spectral-
similarity evidence.
