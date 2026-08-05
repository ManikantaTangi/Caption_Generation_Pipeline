# Stage 4 Module Report — Expert Reasoning Layer

### Objective
Forward-chain hand-authored domain rules over Stage 3's material fractions to assert symbolic context and
enforce hard domain constraints.

### Inputs
Stage 3's `material_fractions`, Stage 1's `Patch.label_patch` (for entropy).

### Internal Modules
RuleEngine, ConstraintChecker, ContextGenerator, ReasoningEngine (facade).

### Methods Used
Forward-chaining condition/conclusion evaluation over `configs/rules.yaml`; Shannon entropy of the local label
patch; multiplicative violation-penalty confidence scoring.

### Algorithms Used
Forward chaining (chosen over backward chaining — no single goal to prove; we want all applicable context).

### AI Paradigms Used
Neuro-Symbolic AI, Expert Systems / classical symbolic AI, Trustworthy AI (hard constraints as a safety net).

### Outputs
`ReasonedKnowledge` (`fired_rules`, `contexts`, `suppressed_categories`, `constraint_violations`,
`reasoning_confidence`, `related_materials`).

### Output Interpretation
`contexts` are the symbolic propositions ("paddy_field_context") Stage 9 can reference in captions;
`reasoning_confidence` is heavily discounted (not just averaged) whenever a hard constraint fires, so
downstream stages never trust a self-contradictory patch highly.

### Module Significance
This is the pipeline's explicit neuro-symbolic bridge: numeric evidence in, auditable symbolic propositions
out, with a built-in trustworthiness discount mechanism.

### Advantages
Fully auditable (every fired rule/violation is named in the output); rules and constraints are edited in YAML
without touching code; conservative-by-design confidence discounting.

### Limitations
Rule set is small and hand-authored (5 rules, 2 constraints); does not learn new rules from data.

### Future Improvements
Rule mining from labelled co-occurrence statistics; migrate to an indexed rules engine (e.g. RETE algorithm via
`durable_rules`) for large rule sets; probabilistic (Markov Logic Network) rule weighting instead of static
confidences.

### Module Workflow
```
material_fractions, label_patch --> RuleEngine.evaluate --> fired_rules
material_fractions              --> ConstraintChecker.check --> violations
fired_rules                     --> ContextGenerator.generate --> contexts, suppressed_categories, related_materials
[fired_rules, violations] --> confidence = mean(rule_conf) * (1-penalty)^violations --> ReasonedKnowledge
```

### Demo Output
```
fired_rules: ['R3'] contexts: ['soybean_cultivation_context'] conf: 0.75
violations: []
```

### Unit Test Output
```
tests/test_stage4_reasoning.py ..........  [ 10 passed ]
```

### Summary
Stage 4 converts Stage 3's raw material statistics into auditable symbolic context while guaranteeing that any
detected domain-constraint violation strictly and multiplicatively lowers downstream trust.
