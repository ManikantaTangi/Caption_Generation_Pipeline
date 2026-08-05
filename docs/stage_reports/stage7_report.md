# Stage 7 Module Report — Uncertainty Estimation

### Objective
Produce calibrated epistemic/aleatoric uncertainty and a single trustworthy confidence score via Monte Carlo
Dropout, rather than a raw, typically overconfident softmax.

### Inputs
Stage 3's `KnowledgeEmbedding.semantic_vector`.

### Internal Modules
BayesianPredictor (MLP head with test-time dropout), MCDropoutSampler, EntropyCalculator, Calibration
(temperature scaling), ConfidenceFusion, UncertaintyEngine (facade).

### Methods Used
Repeated stochastic forward passes with independent dropout masks (T=30 default); entropy decomposition into
predictive / aleatoric / epistemic components; grid-search temperature calibration; multiplicative confidence
fusion.

### Algorithms Used
Monte Carlo Dropout (Gal & Ghahramani, 2016) as approximate Bayesian inference; Depeweg et al. (2018) /
Kwon et al. (2020) style predictive-entropy decomposition; temperature scaling (Guo et al., 2017).

### AI Paradigms Used
Bayesian Deep Learning, Probabilistic AI, Trustworthy AI.

### Outputs
`UncertaintyEstimate` (`class_probs_mean`, `epistemic_uncertainty`, `aleatoric_uncertainty`,
`predictive_entropy`, `confidence_score`, `calibrated`).

### Output Interpretation
High epistemic (relative to aleatoric) suggests the *model* is unsure and more/better training data would help;
high aleatoric suggests the *input itself* is ambiguous (e.g. a genuine field-boundary mixed pixel) — Stage 9's
explanation directly surfaces this distinction.

### Module Significance
This is the pipeline's calibrated-honesty mechanism: every caption's confidence language (Stage 9) and every
fact's weighting (Stage 8, indirectly via the reasoning confidence chain) traces back to this stage's numbers.

### Advantages
Principled uncertainty decomposition (not just a single softmax max); calibration explicitly corrects
overconfidence rather than trusting raw probabilities; MC Dropout requires no architecture change beyond
keeping dropout active at inference.

### Limitations
`BayesianPredictor` is untrained (Xavier-initialised weights only, matching Stage 2's documented scope
decision) — in the reference demo this correctly produces near-uniform, low-confidence predictions throughout,
which is architecturally honest but not yet class-discriminative. Calibration's temperature grid-search needs
labelled validation data to be meaningful; with none, it falls back to the configured default.

### Future Improvements
Train the classifier head (supervised cross-entropy on the fused/semantic vectors, using Stage 1's train
split); benchmark MC Dropout against Deep Ensembles as an uncertainty-estimation ablation; expand calibration
to per-class temperature or Dirichlet calibration.

### Module Workflow
```
semantic_vector --> BayesianPredictor.forward (dropout ON) x T passes --> (T,C) samples
(T,C) samples --> EntropyCalculator.decompose --> p_bar, H_pred, H_aleatoric, H_epistemic
p_bar --> Calibration.apply --> p_calibrated
[p_calibrated, H_pred] --> ConfidenceFusion.fuse --> UncertaintyEstimate
```

### Demo Output
```
confidence: 0.001 epistemic: 0.001 aleatoric: 0.992 pred_entropy: 0.993
```
(Expected given an untrained classifier head — see Limitations above; the *decomposition mechanics* are fully
exercised and correct.)

### Unit Test Output
```
tests/test_stage7_uncertainty.py .............  [ 13 passed ]
```

### Summary
Stage 7 supplies the pipeline's calibrated uncertainty decomposition — architecturally complete and correctly
wired, currently reporting low confidence throughout because the classifier head has not yet been trained,
which is the expected and documented state of this reference implementation.
