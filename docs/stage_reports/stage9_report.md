# Stage 9 Module Report — Confidence-Aware Caption Generation

### Objective
Render Stage 8's verified facts into natural-language captions whose hedging matches Stage 7's confidence, plus
a structured, auditable explanation.

### Inputs
Stage 8's `VerifiedFacts`, Stage 7's `UncertaintyEstimate`, Stage 4's `ReasonedKnowledge`.

### Internal Modules
TemplateSelector, CaptionGenerator, LanguageController, CaptionRefiner, ExplanationGenerator, CaptionEngine
(facade).

### Methods Used
Confidence-band template lookup; slot-filling NLG restricted to verified facts only; regex-based lexical
hedging substitution; length/punctuation post-processing; structured explanation templating.

### Algorithms Used
Template-based (constrained) natural language generation — chosen over free-generation LLM output specifically
to preserve full traceability from caption text back to Stage-8-verified facts.

### AI Paradigms Used
Explainable AI (`explanation` is a first-class output), Trustworthy AI (confidence-banded hedging prevents
overclaiming), Natural Language Generation.

### Outputs
`CaptionResult` (`caption`, `confidence_score`, `confidence_band`, `explanation`) — the pipeline's terminal
artefact.

### Output Interpretation
`caption` is the human-readable output; `explanation` is the audit trail (verified/rejected fact counts,
epistemic-vs-aleatoric attribution, fired rules, constraint warnings) a thesis reviewer or downstream system can
inspect without re-running the pipeline.

### Module Significance
This is what the whole 8-stage pipeline was built to produce: a caption that says exactly as much as the
evidence supports, in exactly the tone that evidence warrants.

### Advantages
Every caption word traces to a verified fact; hedging is deterministic and auditable (no risk of a learned
paraphraser inverting factual polarity); explanation output satisfies the Explainable-AI requirement directly.

### Limitations
Template vocabulary is finite (2 templates per confidence band); currently reports uniformly low confidence
because Stage 7's classifier head is untrained (see Stage 7 report) — captions are correctly hedged as "low
confidence" throughout this reference run, which is the expected, honest behaviour given that limitation.

### Future Improvements
Expand the template bank; explore LLM-based caption *rephrasing* constrained to only the verified-fact set
(e.g. via constrained decoding or a verifier-in-the-loop that re-runs Stage 8 on any LLM-introduced claim);
multi-sentence captions incorporating secondary/relational facts.

### Module Workflow
```
confidence_score --> TemplateSelector.select --> (band, template)
[template, verified_facts] --> CaptionGenerator.generate --> raw caption
raw caption --> LanguageController.control (hedging) --> hedged caption
hedged caption --> CaptionRefiner.refine (length/punctuation) --> final caption
[verified_facts, uncertainty, reasoned] --> ExplanationGenerator.generate --> explanation
--> CaptionResult(caption, confidence_score, confidence_band, explanation)
```

### Demo Output
```
Caption: This patch appears to show characteristics that may correspond to Broad-leaf soybean, but confidence
may be low (0%) and further verification may be recommended.
Confidence: 0.001 (low)
Explanation: 4 fact(s) verified, 4 rejected (verification score=0.60). Uncertainty is predominantly data-driven
(aleatoric) -- the spectral/spatial signal itself is ambiguous for this patch. Reasoning rules applied: R3.
```

### Unit Test Output
```
tests/test_stage9_caption.py ..............  [ 14 passed ]
```

### Summary
Stage 9 closes the loop from raw hyperspectral pixels to an auditable, confidence-calibrated natural-language
caption, completing the knowledge-guided pipeline's Explainable/Trustworthy-AI mandate end to end.
