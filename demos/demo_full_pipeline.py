#!/usr/bin/env python3
"""
Demo: Knowledge-Guided Hyperspectral Image Caption Generation Pipeline
==========================================================================
Runs Stage 1 -> Stage 9 end-to-end on a synthetic WHU-Hi-LongKou-shaped
cube, saves Stage-1 inspection figures, and prints a captioned report
for a handful of test patches plus aggregate accuracy/verification
statistics.

Usage
    PYTHONPATH=src python demos/demo_full_pipeline.py [--config configs/config.yaml] [--n 10]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hsi_caption.pipeline import HSICaptionPipeline
from hsi_caption.stage1_preprocessing.dataset_visualization import DatasetVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n", type=int, default=10, help="Number of test patches to caption")
    parser.add_argument("--random-sample", action="store_true",
                         help="Randomly sample the --n patches from the full test set instead of taking the "
                              "first N (WHU-Hi's official test mask is row-major, so the first N patches are "
                              "spatially clustered and not representative -- use this for an honest accuracy read)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-weights", default=None,
                         help="Path to trained encoder weights (outputs/encoder_weights.pt); omit for untrained")
    parser.add_argument("--uncertainty-weights", default=None,
                         help="Path to trained uncertainty head weights; omit for untrained")
    args = parser.parse_args()

    print("=" * 78)
    print("Knowledge-Guided Hyperspectral Image Caption Generation Pipeline - DEMO")
    print("=" * 78)

    pipeline = HSICaptionPipeline(args.config)

    print("\n[Stage 1] Loading dataset (synthetic WHU-Hi-LongKou-shaped cube if no real data found)...")
    cube = pipeline.load_cube()
    summary, stats, norm_cube, norm_params, patch_dataset = pipeline.preprocess(cube)
    print(summary.as_text())
    print(f"\nClass imbalance ratio: {stats.class_imbalance_ratio:.2f}")
    print(f"Patches -> train={len(patch_dataset.train)}, val={len(patch_dataset.val)}, "
          f"test={len(patch_dataset.test)}")

    viz_dir = os.path.join(pipeline.config.output_dir, "figures")
    visualizer = DatasetVisualizer(viz_dir)
    fc_path = visualizer.save_false_color(cube)
    cm_path = visualizer.save_class_map(cube)
    ms_path = visualizer.save_mean_spectra(cube)
    print(f"\nSaved figures:\n  {fc_path}\n  {cm_path}\n  {ms_path}")

    print("\n[Stage 2-9] Building engines and running the pipeline on test patches...")
    if args.encoder_weights:
        print(f"  Using TRAINED encoder weights: {args.encoder_weights}")
    if args.uncertainty_weights:
        print(f"  Using TRAINED uncertainty head weights: {args.uncertainty_weights}")
    pipeline.build_engines(cube.wavelengths_nm, norm_params, args.encoder_weights, args.uncertainty_weights)

    patches = patch_dataset.test if patch_dataset.test else patch_dataset.train
    if args.random_sample:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(patches), size=min(args.n, len(patches)), replace=False)
        patches = [patches[i] for i in idx]
    else:
        patches = patches[: args.n]
    results = [pipeline.run_patch(p) for p in patches]

    print("\n" + "-" * 78)
    print(f"{'Patch':<42} {'True class':<20} {'Material match':<20}")
    print("-" * 78)
    correct = 0
    for r in results:
        match = "✓" if r.true_class == r.material_top1 else "✗"
        correct += int(r.true_class == r.material_top1)
        print(f"{r.patch_id:<42} {r.true_class:<20} {r.material_top1:<20} {match}")
    print("-" * 78)
    print(f"Top-1 spectral material-match accuracy on shown patches: {correct}/{len(results)} "
          f"({100*correct/len(results):.1f}%)")

    print("\n" + "=" * 78)
    print("SAMPLE CAPTIONS")
    print("=" * 78)
    for r in results[:5]:
        print(f"\nPatch: {r.patch_id}")
        print(f"  True class:       {r.true_class}")
        print(f"  Caption:          {r.caption_result.caption}")
        print(f"  Confidence:       {r.caption_result.confidence_score:.3f} "
              f"({r.caption_result.confidence_band})")
        print(f"  Explanation:      {r.caption_result.explanation}")

    avg_conf = sum(r.caption_result.confidence_score for r in results) / len(results)
    avg_verif = sum(r.verification_score for r in results) / len(results)
    print("\n" + "=" * 78)
    print("AGGREGATE STATISTICS")
    print("=" * 78)
    print(f"  Patches captioned:            {len(results)}")
    print(f"  Mean confidence score:        {avg_conf:.3f}")
    print(f"  Mean verification score:      {avg_verif:.3f}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
