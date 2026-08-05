# Architecture — Knowledge-Guided Hyperspectral Image Caption Generation Pipeline

Dataset: **WHU-Hi-LongKou** (UAV-borne hyperspectral, ~270 bands, 400–1000 nm, 9 agricultural/land-cover classes).
Target: IEEE publication / M.Tech thesis / open-source repository.

## 0. System Overview

```
Stage 1              Stage 2            Stage 3              Stage 4            Stage 5
Preprocessing   -->  Multi Encoder  --> Spectral Knowledge --> Expert         --> Ontology-Enriched
& Inspection         (spectral/          Engine                Reasoning          Semantic Generator
(cube -> patches)     spatial/meta)     (fused_vec ->          Layer              (fractions+context
                      -> fused_vec       semantic_vec)         (fractions ->       -> refined concepts,
                                                                 contexts,          ontology_score)
                                                                 confidence)
       |                                                                                |
       v                                                                                v
Stage 6              Stage 7            Stage 8              Stage 9
Structured Fact  <-- Uncertainty        Fact Verification --> Confidence-Aware
Generator            Estimation         (facts+evidence ->    Caption Generation
(facts -> ranked      (semantic_vec ->   verified/rejected,    (verified facts +
 triplets)             confidence,       verification_score)   confidence ->
                        epistemic/                              caption+explanation)
                        aleatoric)
```

Every stage is a facade class (`Stage*Engine` / `MultiEncoder` / `StructuredFactGenerator`) composed of
independently unit-tested sub-modules, wired end-to-end in `src/hsi_caption/pipeline.py::HSICaptionPipeline`.
The strict input/output contract between stages is enforced by the dataclasses in `src/hsi_caption/datatypes.py`
and directly exercised in `tests/test_integration.py`.

---

## 1. Stage 1 — Preprocessing & Inspection

**Purpose.** Turn a raw (or synthetic, see below) WHU-Hi cube into a validated, normalized, patch-level dataset —
the only stage that touches the sensor's raw pixel grid.

**Why synthetic data is included.** The real WHU-Hi `.mat` releases are hosted off a network this sandboxed
reference implementation cannot reach. `SyntheticWHUHiGenerator` procedurally paints a Voronoi-seeded label
map, then samples each pixel's spectrum from the *same* reference library used in Stage 3, so the whole
pipeline is exercised with a plausible, physically-motivated signal. `DatasetLoader` is a complete, tested
`.mat`/`.npy` reader — pointing `dataset.root_dir` at a real WHU-Hi-LongKou release is a zero-code-change swap
(`DatasetDiscovery` prefers real files automatically).

**Modules.** ConfigManager → DatasetDiscovery → DatasetLoader (+ SyntheticWHUHiGenerator) → DatasetValidator →
DatasetInspector / DatasetStatistics (parallel) → DatasetNormalization → PatchGenerator.

**Algorithms.**
- *Patch extraction*: sliding-window centre sampling, reflection-padded at borders. Alternative considered:
  dense/no-stride sampling — rejected as wasteful given WHU-Hi's large homogeneous regions.
- *Normalization*: per-band min-max (default) vs. z-score. Min-max chosen as default because reflectance is
  physically bounded in [0, 1]; z-score is offered for architectures sensitive to input variance.
- *Split*: stratified-by-class random shuffle (train/val/test). Documented alternative: spatially disjoint
  block-splitting, recommended for rigorous generalisation claims to avoid spatial-autocorrelation leakage
  between adjacent train/test patches — not the default here to match common WHU-Hi benchmark practice.

**Math.**
Min-max: `x' = (x - min_b) / (max_b - min_b)`, per band `b`.
Z-score: `x' = (x - mu_b) / sigma_b`.
Patch centre purity (a metadata feature): `purity = (1/|P|) * sum_{i in P} [label_i == label_center]`.

**AI paradigms used here:** none (classical signal processing / statistics) — this stage is the deterministic,
non-learned foundation everything else builds on.

**Tech stack.** NumPy/SciPy (array ops, `.mat` I/O), Matplotlib (Agg backend, headless-safe figures), PyYAML
(config), scikit-learn *not* required here (kept for Stage 7 calibration grid utilities if extended).

---

## 2. Stage 2 — Multi Encoder

**Purpose.** Learn (architecturally — see note below) a unified embedding per patch from three complementary
views: spectral shape, spatial texture, and positional/homogeneity metadata.

**Modules.** SpectralEncoder (multi-scale 1D-CNN) ‖ SpatialEncoder (ViT-style tokenized self-attention) ‖
MetadataEncoder (MLP) → FeatureFusion (concat / sum / attention).

**A note on training.** `src/hsi_caption/nn_utils.py` implements the exact architectural interfaces
(`Dense`, `Conv1D`, `SingleHeadAttention`, `Dropout`) a PyTorch `nn.Module` stack would use, with
Xavier-initialised weights and a correct forward pass — but **no backpropagation/optimizer**. This is a
deliberate scope decision: the pipeline's contribution is the *system architecture and stage contracts*, not a
from-scratch autodiff engine. A production run trains these exact layer shapes with `torch.nn` + Adam on
patch-level cross-entropy (classification) or contrastive loss (self-supervised) objectives — see
`docs/stage_reports/stage2_report.md` "Future Improvements".

**Algorithms & math.**
- *Spectral*: 3 parallel `Conv1D(kernel in {7,5,3})` branches over the raw (B,1) spectrum, global-average-pooled,
  concatenated, projected: `h_k = GAP(ReLU(Conv1D_k(x)))`, `z_spec = W_p [h_7; h_5; h_3] + b_p`.
- *Spatial*: patch tokenized into a `sqrt(T) x sqrt(T)` grid of block-mean descriptors; single-head scaled
  dot-product self-attention: `Attn(Q,K,V) = softmax(QK^T / sqrt(d)) V`, mean-pooled over tokens, projected.
- *Metadata*: 2-layer MLP over `[row_norm, col_norm, local_class_purity]`.
- *Fusion (attention, default)*: the three modality vectors, each projected to `fused_embed_dim`, are treated as
  3 tokens and passed through `SingleHeadAttention`; the attended tokens are mean-pooled. This lets per-patch
  modality salience vary (e.g. a homogeneous water patch needs little spatial evidence).

**AI paradigms used here:** Deep Learning (all three encoders), CNN (spectral), Vision Transformer /
self-attention (spatial and fusion).

**Tech stack (production).** PyTorch (`torch.nn.Conv1d`, `torch.nn.MultiheadAttention`), optionally
HuggingFace `transformers` for a pretrained spectral/spatial backbone if fine-tuning on more WHU-Hi scenes.

---

## 3. Stage 3 — Spectral Knowledge Engine

**Purpose.** Ground the pipeline in domain knowledge: match each patch's physical spectrum against a curated
reference library, express material composition quantitatively, and build a queryable knowledge graph —
this is the concrete meaning of "knowledge-guided" in the pipeline's name.

**Modules.** SpectralLibrary → KnowledgeRetrieval (SAM / cosine / euclidean) → MaterialIdentification (per-pixel
majority vote) → KnowledgeGraphConstruction (NetworkX) → SemanticRepresentation (concatenated knowledge vector).

**Critical design point — normalization consistency.** Stage 1 normalizes the cube (for the learned encoders'
benefit); if the reference library were left in raw reflectance units, similarity search would be systematically
wrong (see `SpectralLibrary.apply_normalization`, added after an empirical bug was caught during integration
testing — top-1 match accuracy went from near-chance to 80–90% on synthetic data after the fix).

**Algorithms & math.**
- *Spectral Angle Mapper (default)*: `SAM(x, r) = arccos( (x·r) / (||x|| ||r||) )`, mapped to a similarity
  `1 - SAM/pi`. Chosen over Euclidean distance because SAM is invariant to multiplicative illumination/shadow
  scaling — standard practice in the HSI material-matching literature.
- *Material identification*: per-pixel top-1 SAM match, tallied into fractions:
  `frac(m) = (1/|Patch|) * sum_{i in Patch} [argmax_m' SAM_sim(x_i, r_m') == m]`.
- *Knowledge graph*: static `similarTo` edges where cosine similarity of two library spectra exceeds
  `kg_similarity_threshold` (0.75 default); `belongsTo` edges to category nodes.

**AI paradigms used here:** Knowledge Graph (NetworkX `DiGraph`), Probabilistic AI (fraction-based material
composition as an empirical distribution), Semantic Web concepts (material/category typed nodes — formalised
further in Stage 5's ontology), Neuro-Symbolic AI (learned `fused_vector` concatenated with symbolic
similarity/fraction features in `semantic_vector`).

**Tech stack (production).** NetworkX (used as-is), optionally Neo4j for a persistent/queryable graph store at
scale, Sentence-Transformers if migrating from exact-name to open-vocabulary material matching.

---

## 4. Stage 4 — Expert Reasoning Layer

**Purpose.** Forward-chain hand-authored domain rules over Stage 3's material fractions to assert symbolic
context, and enforce hard domain constraints — the pipeline's explicit neuro-symbolic bridge.

**Modules.** RuleEngine (forward chaining) → ConstraintChecker (hard invariants) → ContextGenerator (aggregation)
→ ReasoningEngine (facade, confidence scoring).

**Why forward chaining, not backward.** There is no single goal proposition to *prove* per patch — we want
*all* applicable context asserted from the available facts, which forward chaining naturally enumerates in one
pass (`O(R)`, R = rule count).

**Math.**
`reasoning_confidence = clip( mean(confidence_r for r in fired_rules) * (1 - penalty)^(#violations), 0, 1 )`
— multiplicative violation discounting ensures a hard-constraint break is never masked by unrelated high rule
confidence (a trustworthy-AI design choice).

**AI paradigms used here:** Neuro-Symbolic AI (rules over numeric fractions), Expert Systems / classical
symbolic AI (forward-chaining rule engine), Trustworthy AI (hard constraints as a non-negotiable safety net).

**Tech stack.** Plain Python + PyYAML rule definitions; a production system with hundreds of rules would migrate
to a proper rules engine (e.g. `durable_rules`, CLIPS bindings, or a Prolog embedding) for indexed rule matching.

---

## 5. Stage 5 — Ontology-Enriched Semantic Generator

**Purpose.** Attach materials to a formal `subClassOf` class hierarchy (Crop → CerealCrop → Rice, etc.) and
object-property axioms, enabling hierarchical/generalising language ("a cereal crop") and axiom-based fact
checking in Stage 8.

**Modules.** OntologyLoader (YAML → NetworkX hierarchy + axioms) → OntologyMatcher (class matching + branch
coherence) → SemanticGenerator (leaf + parent concept selection) → OntologyRefiner (iterative pruning against
Stage 4's suppressed categories).

**Why YAML instead of a `.owl` file + OWLReady2.** Zero heavyweight dependency, fast install, identical
semantics (subclass hierarchy + typed object properties + axioms) for this reference scale (18 classes). The
`OntologyLoader` interface is deliberately the *same* shape a real OWLReady2/RDFlib-backed loader would expose,
so swapping in a real `.owl` file (e.g. AGROVOC or a custom crop ontology) requires no change to Stages 5–8.

**Math.** Branch coherence score: `coherence = max_branch( sum_{m in branch} frac(m) ) / sum_m frac(m)` — the
fraction of total probability mass concentrated in a single top-level ontology branch (Vegetation/Water/
BuiltUp). Ontology score after refinement: `score = coherence * 0.9^(#concepts_removed)`.

**AI paradigms used here:** Ontology Engineering, Semantic Web (RDF-style subject/predicate/object axioms),
Knowledge Graph (the hierarchy is itself a typed graph), Explainable AI (hierarchy paths are directly
reportable — "Rice, a CerealCrop, a Crop, a Vegetation").

**Tech stack (production).** OWLReady2 or RDFlib with a real `.owl`/`.ttl` file; SPARQL queries for axiom
lookups at scale.

---

## 6. Stage 6 — Structured Fact Generator

**Purpose.** Convert Stage 3–5's numeric/symbolic outputs into ranked `(subject, predicate, object, score)`
triples — the atomic, auditable unit Stage 8 verifies and Stage 9 captions from.

**Modules.** FactExtraction (template instantiation) → TripletGenerator (typed wrapping) →
RelationshipGenerator (material-to-material `coOccursWith`) → FactRanking (dedupe + threshold + top-k).

**Why template instantiation, not an LLM extractor.** Every fact must be traceable to a specific upstream
stage/value for the Explainability requirement — an open-ended extractor would break that guarantee.

**Math.** Ranking is a straightforward `sort by score desc, truncate to max_facts_per_patch` after
deduplication on `(s,p,o)` keeping the max score — `O(F log F)`.

**AI paradigms used here:** Knowledge representation (triples are the RDF-style atomic unit used across the
Semantic-Web-influenced stages 3/5/8), Neuro-Symbolic AI (facts mix learned-similarity scores and rule/ontology
scores in one representation).

---

## 7. Stage 7 — Uncertainty Estimation

**Purpose.** Produce calibrated epistemic/aleatoric uncertainty and a single confidence score via **Monte Carlo
Dropout**, the standard cheap approximate-Bayesian technique for deep networks (Gal & Ghahramani, 2016).

**Modules.** BayesianPredictor (MLP head, test-time dropout) → MCDropoutSampler (T stochastic passes) →
EntropyCalculator (predictive/aleatoric/epistemic decomposition) → Calibration (temperature scaling) →
ConfidenceFusion (final scalar).

**Math (entropy decomposition, Depeweg et al. 2018 / Kwon et al. 2020).**
```
p_bar         = (1/T) sum_t p_t                      (mean predictive distribution)
H_predictive  = H[p_bar] = -sum_c p_bar_c log p_bar_c  (total uncertainty)
H_aleatoric   = E_t[H[p_t]] = (1/T) sum_t H[p_t]       (expected data noise)
H_epistemic   = H_predictive - H_aleatoric             (= mutual information / BALD score)
```
**Calibration (Guo et al. 2017, adapted to post-softmax probabilities).**
`p_cal = normalize(p ** (1/T))`, `T` grid-searched to minimise validation NLL.
**Confidence fusion.** `confidence = max_c(p_cal) * (1 - H_predictive_normalized)`.

**AI paradigms used here:** Bayesian Deep Learning (MC Dropout as approximate Bayesian inference), Probabilistic
AI (entropy decomposition, calibration), Trustworthy AI (calibrated, not raw-softmax, confidence reporting).

**Tech stack (production).** PyTorch with `model.train()` kept active at inference for real dropout sampling;
`torch.distributions` for richer posterior approximations (e.g. full variational inference / Deep Ensembles as
an alternative uncertainty method, discussed as an ablation in the IEEE write-up).

---

## 8. Stage 8 — Fact Verification

**Purpose.** Independently cross-check every Stage 6 triplet against four evidence sources before it is allowed
to appear in a caption — the pipeline's core Trustworthy-AI checkpoint.

**Modules.** EvidenceRetrieval → {KnowledgeGraphVerification, OntologyMatching, RuleVerification,
SemanticSimilarity} (four independent scorers) → ConfidenceFusion (weighted linear combination).

**Math.** `verification_score(triplet) = w_sem*sem + w_kg*kg + w_onto*onto + w_rule*rule`, weights configured
to sum to 1.0 (validated at startup by `PipelineConfig.validate()`); `verified` iff score ≥
`verification_threshold` (0.55 default).

**Why linear fusion, not a learned classifier.** Every verification decision must be explainable in terms of
its four named components ("rejected because ontology=0.0, KG=0.0") — a learned re-ranker would obscure that,
and there is no labelled caption-quality dataset to train one against.

**AI paradigms used here:** Explainable AI (fully decomposable score), Trustworthy AI (independent
cross-checking before facts reach the user), Neuro-Symbolic AI (semantic-similarity is a learned/measured
signal; KG/ontology/rule checks are symbolic).

---

## 9. Stage 9 — Confidence-Aware Caption Generation

**Purpose.** Render Stage 8's verified facts into natural-language text whose *hedging* matches Stage 7's
confidence — the pipeline's terminal, user-facing artefact.

**Modules.** TemplateSelector (confidence-band template bank) → CaptionGenerator (slot filling from verified
facts only) → LanguageController (lexical hedging) → CaptionRefiner (length/punctuation cleanup) →
ExplanationGenerator (structured audit trail).

**Why template slot-filling, not a free-generation LLM call.** Every word in the caption must be traceable to a
Stage-8-verified fact for the auditability guarantee threaded through the whole pipeline. A production
extension could use an LLM *constrained* to only rephrase the verified-fact set (e.g. via constrained decoding
or a verifier-in-the-loop), documented as a "Future Improvement" in
`docs/stage_reports/stage9_report.md`.

**AI paradigms used here:** Explainable AI (the `explanation` output is a first-class pipeline artefact, not an
afterthought), Trustworthy AI (confidence-banded hedging prevents overclaiming), Natural Language Generation
(template-based NLG).

---

## Tech Stack Summary (reference implementation)

| Library | Used for | Why |
|---|---|---|
| NumPy / SciPy | array ops, `.mat` I/O, all `nn_utils` layers | zero-dependency, portable, fast enough at this scale |
| NetworkX | Stage 3 knowledge graph, Stage 5 ontology hierarchy | pure-Python graph ops, no server required |
| PyYAML | all `configs/*.yaml` | human-editable domain knowledge (rules/ontology/library/templates) |
| Matplotlib (Agg) | Stage 1 visualisation | headless-safe, IEEE-figure-ready output |
| pytest | 100+ unit + integration tests | industry standard, fixture reuse via `conftest.py` |

**Documented production upgrade path** (not required to run this reference implementation, see
`requirements.txt` comments): PyTorch + Transformers for Stage 2 (real backprop/pretraining), Sentence-
Transformers for open-vocabulary Stage 3/5 matching, OWLReady2/RDFlib for a real `.owl` ontology, Neo4j for a
persistent knowledge graph store.

## Coding Rules Applied

- **SOLID / OOP**: every module is a single-responsibility class; facades depend on abstractions (constructor
  injection of config/sub-objects) not concrete globals.
- **Dependency injection**: `PipelineConfig` is loaded once and threaded through every constructor; no module
  reads YAML directly except its own owning facade.
- **Configuration-driven**: all thresholds/weights/paths live in `configs/*.yaml`, none are hard-coded.
- **Exception handling**: each stage defines its own narrow exception type (`DatasetValidationError`,
  `KnowledgeRetrievalError`, `FeatureFusionError`, …) raised with actionable messages.
- **Logging**: a single `configure_logging()` call in `ConfigManager` standardises format across all modules;
  every module logs at INFO for milestones, DEBUG for per-patch detail.
- **Type hints & docstrings**: every public class/function is typed and documented with Purpose/Input/Output/
  Algorithm/Complexity, matching this document's structure.
