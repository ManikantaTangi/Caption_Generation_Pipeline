# Stage 5 Module Report — Ontology-Enriched Semantic Generator

### Objective
Attach materials to a formal class hierarchy and object-property axioms, enabling hierarchical language and
axiom-based fact checking.

### Inputs
Stage 3's `material_fractions`, Stage 4's `ReasonedKnowledge.suppressed_categories`.

### Internal Modules
OntologyLoader, OntologyMatcher, SemanticGenerator, OntologyRefiner, OntologyEngine (facade).

### Methods Used
NetworkX `DiGraph` subclass hierarchy traversal (`nx.dfs_preorder_nodes`, `nx.descendants`); exact-name class
matching; iterative concept pruning against suppressed categories.

### Algorithms Used
Fixed-point-style iterative refinement (bounded by `refine_iterations`); branch-coherence scoring.

### AI Paradigms Used
Ontology Engineering, Semantic Web (RDF-style subject/predicate/object axioms), Knowledge Graph, Explainable AI
(reportable hierarchy paths).

### Outputs
`OntologySemanticRepresentation` (`matched_classes`, `class_hierarchy_paths`, `refined_concepts`,
`ontology_score`).

### Output Interpretation
`refined_concepts` is the leaf + informative-ancestor concept list a caption can mention (e.g. "Rice" and its
parent "CerealCrop"); `ontology_score` combines branch coherence with a penalty for concepts removed during
refinement.

### Module Significance
Provides the formal semantic backbone that lets captions generalise ("a cereal crop") when appropriate and lets
Stage 8 check facts against declared axioms (e.g. Rice `requiresIrrigationFrom` Water) rather than only
retrieval similarity.

### Advantages
Lightweight YAML-based ontology (fast install, zero heavy dependency) with an interface identical to a real
OWLReady2/RDFlib-backed loader — swappable without touching Stages 5–8's logic.

### Limitations
Exact-name matching assumes vocabulary alignment; small ontology (18 classes, 2 axioms) is illustrative, not
exhaustive.

### Future Improvements
Load a real `.owl`/`.ttl` domain ontology (e.g. AGROVOC) via OWLReady2; add SPARQL-based axiom queries; expand
object properties (e.g. `harvestedIn`, `rotatedWith`) for richer captioning.

### Module Workflow
```
material_fractions --> OntologyMatcher.match --> OntologyMatch[] --> coherence_score
OntologyMatch[]     --> SemanticGenerator.generate --> refined-concept candidates
[candidates, suppressed_categories, coherence_score] --> OntologyRefiner.refine --> OntologySemanticRepresentation
```

### Demo Output
```
refined_concepts: ['Narrow-leaf soybean', 'LegumeCrop', 'Corn', 'Broad-leaf soybean', 'Rice']
ontology_score: 1.0
```

### Unit Test Output
```
tests/test_stage5_ontology.py ..............  [ 14 passed ]
```

### Summary
Stage 5 formalises Stage 3's flat material list into a navigable class hierarchy with axiom support, feeding
both richer caption language (Stage 9) and an independent verification signal (Stage 8).
