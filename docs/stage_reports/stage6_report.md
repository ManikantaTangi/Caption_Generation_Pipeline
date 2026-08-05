# Stage 6 Module Report — Structured Fact Generator

### Objective
Convert Stage 3–5's numeric/symbolic outputs into ranked, auditable `(subject, predicate, object, score)`
triples.

### Inputs
Stage 3's `material_fractions`, Stage 4's `ReasonedKnowledge`, Stage 5's `OntologySemanticRepresentation`.

### Internal Modules
FactExtractor, TripletGenerator, RelationshipGenerator, FactRanking, StructuredFactGenerator (facade).

### Methods Used
Deterministic template instantiation per material/context/concept; deduplication on `(s,p,o)` keeping max
score; score-threshold filtering; descending sort + truncation.

### Algorithms Used
Simple auditable template-based extraction (rejected: open-ended LLM extraction, for traceability reasons).

### AI Paradigms Used
Knowledge representation (RDF-style triples), Neuro-Symbolic AI (facts mix learned-similarity and
rule/ontology-derived scores in one representation).

### Outputs
`StructuredFacts` (`triplets`, `ranked_facts`).

### Output Interpretation
`ranked_facts` (capped at `max_facts_per_patch`, default 8) is the final candidate-fact set Stage 8 must
independently verify before any of it can appear in a caption.

### Module Significance
Bridges the "what do we know" stages (3–5) and the "is it actually true" stage (8) via one uniform, typed
representation.

### Advantages
Every fact is traceable to a named upstream stage/value; deduplication and thresholding keep the fact list
small and high-signal.

### Limitations
Template vocabulary is fixed (`hasMaterial`/`hasContext`/`hasConcept`/`coOccursWith`); no comparative or
temporal relations yet.

### Future Improvements
Expand predicate vocabulary (e.g. `borders`, `precedesInRotation`); learn a calibrated re-ranker once labelled
caption-quality data exists; multi-hop fact chaining via the knowledge graph.

### Module Workflow
```
[material_fractions, reasoned, ontology_rep] --> FactExtractor.extract --> CandidateFact[]
CandidateFact[] --> TripletGenerator.generate --> Triplet[] (base facts)
[dominant_material, fractions, reasoned] --> RelationshipGenerator.generate --> Triplet[] (relations)
base + relations --> FactRanking.rank --> StructuredFacts (deduped, thresholded, ranked)
```

### Demo Output
```
ranked facts:
  ... hasConcept Narrow-leaf soybean 0.964
  ... hasConcept LegumeCrop 0.964
  ... hasContext soybean_cultivation_context 0.75
```

### Unit Test Output
```
tests/test_stage6_facts.py ..........  [ 10 passed ]
```

### Summary
Stage 6 produces the pipeline's single, uniform, auditable fact representation — the last stop before
independent verification.
