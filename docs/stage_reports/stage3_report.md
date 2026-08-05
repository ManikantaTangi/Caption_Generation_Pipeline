# Stage 3 Module Report — Spectral Knowledge Engine

### Objective
Ground the pipeline in physical domain knowledge by matching patch spectra against a curated reference library
and materialising the result as a queryable knowledge graph and semantic vector.

### Inputs
Stage 1's `Patch.cube_data` (raw spectra) and Stage 2's `FusedEmbedding.fused_vector`.

### Internal Modules
SpectralLibrary, KnowledgeRetrieval (SAM/cosine/euclidean), MaterialIdentifier (per-pixel majority vote),
KnowledgeGraphBuilder (NetworkX), SemanticRepresentationBuilder, SpectralKnowledgeEngine (facade).

### Methods Used
Piecewise-linear spectral interpolation onto the sensor grid; Spectral Angle Mapper similarity; per-pixel
top-1 retrieval tallying; static similarity-graph construction.

### Algorithms Used
Spectral Angle Mapper (default, illumination-invariant); cosine similarity and Euclidean distance as
configurable alternatives.

### AI Paradigms Used
Knowledge Graph, Probabilistic AI (empirical material-fraction distribution), Semantic Web (typed
material/category nodes), Neuro-Symbolic AI (semantic_vector concatenates learned + symbolic features).

### Outputs
`KnowledgeEmbedding` (`material_matches`, `kg_node_ids`, `semantic_vector`, `material_fractions`).

### Output Interpretation
`material_fractions` is a per-patch empirical distribution over the 9 library materials, purely from spectral
evidence (never from ground-truth labels) — an independent check Stage 7/8 later cross-reference.

### Module Significance
This is literally the "knowledge-guided" mechanism in the pipeline's name: every downstream symbolic stage
(4, 5, 6, 8) is grounded in this stage's physically-motivated retrieval, not just a black-box classifier.

### Advantages
Interpretable per-material similarity scores; consistent normalization space with Stage 1 (bug caught and fixed
during integration testing, see ARCHITECTURE.md); reusable library across different sensor band counts via
interpolation.

### Limitations
Library is small and hand-curated (9 materials); exact-name matching assumes the label vocabulary and library
vocabulary are aligned — an open-vocabulary sensor would need Sentence-Transformers-based fuzzy matching.

### Future Improvements
Expand the library with real USGS/ECOSTRESS spectral reference data; add open-vocabulary retrieval; persist the
knowledge graph in Neo4j for cross-scene querying.

### Module Workflow
```
Patch.cube_data --center pixel--> KnowledgeRetrieval (SAM) --> top-k MaterialMatch
Patch.cube_data --all pixels--->  MaterialIdentifier        --> material_fractions
MaterialMatch  ----------------->  KnowledgeGraphBuilder.query --> kg_node_ids
[fused_vector, matches, kg_node_ids, fractions] --> SemanticRepresentationBuilder --> KnowledgeEmbedding
```

### Demo Output
```
top1 match rate on 40 patches: 0.80
matches: [('Narrow-leaf soybean', 0.978), ('Broad-leaf soybean', 0.965), ('Corn', 0.951)]
```

### Unit Test Output
```
tests/test_stage3_knowledge_engine.py ..............  [ 14 passed ]
```

### Summary
Stage 3 is the pipeline's knowledge-grounding core, converting raw spectra into an interpretable material
composition, a knowledge graph, and a hybrid semantic embedding — achieving 80–90% top-1 material-match
accuracy against ground truth on synthetic data.
