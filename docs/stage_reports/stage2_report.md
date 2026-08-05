# Stage 2 Module Report — Multi Encoder

### Objective
Produce a single unified embedding per patch by fusing spectral, spatial, and metadata views.

### Inputs
Stage 1's `Patch` objects (`cube_data`, `metadata`).

### Internal Modules
SpectralEncoder (multi-scale 1D-CNN), SpatialEncoder (ViT-style tokenized self-attention), MetadataEncoder
(2-layer MLP), FeatureFusion (concat / sum / attention), MultiEncoder (facade).

### Methods Used
1D convolution over the spectral axis; block-mean tokenization + single-head scaled dot-product attention over
spatial patches; dense projection layers throughout (`nn_utils.py`).

### Algorithms Used
Multi-scale kernel spectral CNN (kernels {7,5,3}); ViT-style patch tokenization; attention-based multimodal
fusion (default) vs. concat/sum baselines.

### AI Paradigms Used
Deep Learning, CNN (spectral), Vision Transformer / self-attention (spatial + fusion).

### Outputs
`FusedEmbedding` (`spectral_vector`, `spatial_vector`, `metadata_vector`, `fused_vector`, optional
`attention_weights`).

### Output Interpretation
`fused_vector` (default 128-d) is the compact numeric summary of "what does this patch look like spectrally,
spatially, and where is it" — consumed directly by Stage 3.

### Module Significance
Provides the *learned* half of the neuro-symbolic pipeline; Stage 3 concatenates this with symbolic knowledge
features rather than relying on either signal alone.

### Advantages
Modular per-modality encoders are independently swappable/testable; attention fusion adapts modality weighting
per patch instead of a fixed combination rule.

### Limitations
Weights are Xavier-initialised and **not trained** (no backprop implemented in this reference scope) — the
embeddings are architecturally meaningful but not yet discriminatively optimised; see Future Improvements.

### Future Improvements
Port `nn_utils.py` layer-for-layer to `torch.nn`; add a supervised (cross-entropy on class labels) or
self-supervised (SimCLR-style spectral-spatial contrastive) training loop; multi-layer ViT for larger patches.

### Module Workflow
```
Patch.cube_data --center pixel--> SpectralEncoder --> spectral_vector (64d)
Patch.cube_data --full patch-->   SpatialEncoder  --> spatial_vector  (64d)
Patch.metadata  ------------->    MetadataEncoder --> metadata_vector (16d)
[spectral_vector, spatial_vector, metadata_vector] --> FeatureFusion (attention) --> fused_vector (128d)
```

### Demo Output
```
fused vector shape (128,) [ 0.0123 -0.0456  0.0789 ... ]
```

### Unit Test Output
```
tests/test_stage2_multi_encoder.py ..............  [ 14 passed ]
```

### Summary
Stage 2 turns each raw patch into a compact, multimodal, architecturally-CNN/ViT-grounded embedding that Stage
3 enriches with explicit domain knowledge.
