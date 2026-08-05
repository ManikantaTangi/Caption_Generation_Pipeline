# Stage 1 Module Report — Preprocessing & Inspection

### Objective
Transform a raw (or synthetic) WHU-Hi-LongKou hyperspectral cube into a validated, normalized, patch-level
dataset ready for feature encoding.

### Inputs
Dataset root directory (`dataset.root_dir`); falls back to a procedurally generated synthetic cube if no real
`.mat`/`.npy` artefacts are found.

### Internal Modules
ConfigManager, DatasetDiscovery, DatasetLoader (+ SyntheticWHUHiGenerator), DatasetValidator, DatasetInspector,
DatasetVisualizer, DatasetStatisticsComputer, DatasetNormalizer, PatchGenerator.

### Methods Used
Directory-walk discovery; `.mat`/`.npy` parsing; battery-of-checks validation; vectorised NumPy statistics;
per-band min-max/z-score normalization; stratified sliding-window patch extraction and split.

### Algorithms Used
Voronoi-seeded synthetic label painting; reflection-padding for border patches; stratified-by-class random
shuffle split.

### AI Paradigms Used
None (classical signal processing / statistics) — the deterministic foundation stage.

### Outputs
`PatchDataset` (train/val/test `Patch` lists), `DatasetStatistics`, `InspectionSummary`, saved figures
(false-colour composite, class map, mean spectra).

### Output Interpretation
Each `Patch` carries a `(patch_size, patch_size, num_bands)` cube, its centre label, a local label patch (for
entropy/purity), and normalized-position metadata — this is the atomic unit every later stage operates on.

### Module Significance
No downstream stage can produce trustworthy output on top of silently corrupt or unnormalized data; this stage
is the fail-fast gate for the entire pipeline.

### Advantages
Fully config-driven; synthetic-data fallback keeps the pipeline runnable without network access to the dataset
host; explicit validation with actionable error messages.

### Limitations
Synthetic cube is illustrative, not a substitute for real WHU-Hi-LongKou benchmarking; default class-stratified
split can leak spatially-correlated information between splits (documented alternative: block-splitting).

### Future Improvements
Add a real WHU-Hi `.mat` download/cache utility (network-permitting); implement spatially-disjoint block
splitting as a configurable alternative; add band-selection/dimensionality-reduction as an optional Stage 1
sub-module.

### Module Workflow
```
root_dir --> DatasetDiscovery --> [found?] --yes--> DatasetLoader --> HSICube
                                   |no
                                   v
                          SyntheticWHUHiGenerator --> HSICube
HSICube --> DatasetValidator --> DatasetInspector / DatasetStatistics
                                        |
                                        v
                              DatasetNormalizer --> normalized HSICube
                                        |
                                        v
                                 PatchGenerator --> PatchDataset (train/val/test)
```

### Demo Output
```
Cube: synthetic_WHU-Hi-LongKou
  Shape: 120 x 120 x 270 bands (float32)
  Value range: [0.0000, 0.6133], mean=0.2024
  Class distribution:
    Corn                  :    1185 px ( 8.23%)
    Cotton                :    1697 px (11.78%)
    ...
Patches -> train=2016, val=432, test=432
```

### Unit Test Output
```
tests/test_stage1_preprocessing.py ......................  [ 22 passed ]
```

### Summary
Stage 1 establishes a validated, normalized, patch-level dataset contract that every later stage relies on,
with a synthetic-data fallback that keeps the whole pipeline runnable end-to-end without requiring network
access to the real WHU-Hi-LongKou release.
