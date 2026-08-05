#!/usr/bin/env python3
"""
Before/After Comparison: Untrained vs. Trained Pipeline
============================================================
Runs the exact same set of test patches through the full 9-stage
pipeline twice -- once with the architecturally-identical-but-untrained
numpy reference encoder/uncertainty-head ("Before"), and once with the
trained PyTorch weights ("After") -- and produces report-ready output:
a per-patch comparison table, aggregate metrics (accuracy, mean
confidence, mean verification score), a bar-chart figure, and a
Markdown report file suitable for pasting directly into a thesis or
IEEE write-up.

Usage
    PYTHONPATH=src python demos/compare_before_after.py \
        --config configs/config.yaml \
        --encoder-weights outputs/encoder_weights.pt \
        --uncertainty-weights outputs/uncertainty_head_weights.pt \
        --n-samples 10 --n-eval 300 --seed 0

`--n-samples` controls how many patches get full side-by-side caption
detail in the report (keep this small -- e.g. 10 -- for readability).
`--n-eval` controls how many patches are used for the aggregate
accuracy/confidence statistics (larger = more statistically meaningful;
each additional patch costs one full Stage-1..9 pass through BOTH
pipelines, so a few hundred is a reasonable default; use the full test
set for final reported numbers if time allows).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from hsi_caption.pipeline import HSICaptionPipeline


def run_batch(pipeline: HSICaptionPipeline, patches, label: str):
    """Runs `pipeline.run_patch` over `patches` and returns a list of results
    plus timing, printing progress every 50 patches for long runs."""
    results = []
    t0 = time.time()
    for i, patch in enumerate(patches):
        results.append(pipeline.run_patch(patch))
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(patches)} patches ({time.time()-t0:.1f}s elapsed)")
    print(f"  [{label}] done: {len(patches)} patches in {time.time()-t0:.1f}s")
    return results


def compute_aggregate(results) -> dict:
    correct = sum(1 for r in results if r.predicted_material == r.true_class)
    n = len(results)
    confidences = [r.caption_result.confidence_score for r in results]
    verif_scores = [r.verification_score for r in results]
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "std_confidence": float(np.std(confidences)) if confidences else 0.0,
        "mean_verification_score": float(np.mean(verif_scores)) if verif_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--encoder-weights", default="outputs/encoder_weights.pt")
    parser.add_argument("--uncertainty-weights", default="outputs/uncertainty_head_weights.pt")
    parser.add_argument("--n-samples", type=int, default=10, help="patches shown in full side-by-side detail")
    parser.add_argument("--n-eval", type=int, default=300, help="patches used for aggregate statistics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    print("=" * 78)
    print("BEFORE / AFTER COMPARISON: Untrained vs. Trained Pipeline")
    print("=" * 78)

    pipeline = HSICaptionPipeline(args.config)
    print("\n[Stage 1] Loading and preprocessing dataset...")
    cube = pipeline.load_cube()
    summary, stats, norm_cube, norm_params, patch_dataset = pipeline.preprocess(cube)
    print(f"train={len(patch_dataset.train)} val={len(patch_dataset.val)} test={len(patch_dataset.test)}")

    all_test = patch_dataset.test if patch_dataset.test else patch_dataset.train
    n_eval = min(args.n_eval, len(all_test))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_test), size=n_eval, replace=False)
    eval_patches = [all_test[i] for i in idx]
    sample_patches = eval_patches[: args.n_samples]

    # ---- BEFORE: untrained numpy reference engines ----
    print(f"\n[BEFORE] Building untrained reference engines and running on {n_eval} patches...")
    pipeline.build_engines(cube.wavelengths_nm, norm_params)  # no weight paths => untrained numpy backend
    before_results = run_batch(pipeline, eval_patches, "BEFORE")
    before_agg = compute_aggregate(before_results)
    before_samples = {r.patch_id: r for r in before_results[: args.n_samples]}

    # ---- AFTER: trained PyTorch engines ----
    print(f"\n[AFTER] Building trained engines and running on the SAME {n_eval} patches...")
    pipeline.build_engines(cube.wavelengths_nm, norm_params, args.encoder_weights, args.uncertainty_weights)
    after_results = run_batch(pipeline, eval_patches, "AFTER")
    after_agg = compute_aggregate(after_results)
    after_samples = {r.patch_id: r for r in after_results[: args.n_samples]}

    # ---- Aggregate comparison table ----
    print("\n" + "=" * 78)
    print("AGGREGATE METRICS (n=%d randomly sampled test patches)" % n_eval)
    print("=" * 78)
    chance_accuracy = 1.0 / pipeline.config.dataset.num_classes
    print(f"(Reference: random-chance accuracy for {pipeline.config.dataset.num_classes} classes = "
          f"{chance_accuracy*100:.1f}%)")
    print(f"{'Metric':<30} {'Before (untrained)':<22} {'After (trained)':<22} {'Change':<12}")
    print("-" * 86)
    rows = [
        ("Material-match accuracy", "accuracy", "%"),
        ("Mean confidence score", "mean_confidence", ""),
        ("Std. dev. of confidence", "std_confidence", ""),
        ("Mean verification score", "mean_verification_score", ""),
    ]
    for label, key, fmt in rows:
        b, a = before_agg[key], after_agg[key]
        if fmt == "%":
            print(f"{label:<30} {b*100:>18.1f}%   {a*100:>18.1f}%   {(a-b)*100:>+9.1f} pp")
        else:
            print(f"{label:<30} {b:>19.3f}   {a:>19.3f}   {a-b:>+11.3f}")

    # ---- Side-by-side sample captions ----
    print("\n" + "=" * 78)
    print("SAMPLE CAPTIONS: BEFORE vs. AFTER")
    print("=" * 78)
    for patch in sample_patches:
        b = before_samples[patch.patch_id]
        a = after_samples[patch.patch_id]
        print(f"\nPatch: {patch.patch_id}")
        print(f"  True class: {b.true_class}")
        print(f"  BEFORE -> predicted: {b.predicted_material:<20} confidence: {b.caption_result.confidence_score:.3f} "
              f"({b.caption_result.confidence_band})")
        print(f"            caption: {b.caption_result.caption}")
        print(f"  AFTER  -> predicted: {a.predicted_material:<20} confidence: {a.caption_result.confidence_score:.3f} "
              f"({a.caption_result.confidence_band})")
        print(f"            caption: {a.caption_result.caption}")

    # ---- Figures ----
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    _save_comparison_chart(before_agg, after_agg, os.path.join(args.output_dir, "figures", "before_after_comparison.png"))
    _save_confidence_histograms(before_results, after_results,
                                 os.path.join(args.output_dir, "figures", "before_after_confidence_hist.png"))

    # ---- Markdown report ----
    report_path = os.path.join(args.output_dir, "before_after_report.md")
    _write_markdown_report(report_path, n_eval, before_agg, after_agg, sample_patches, before_samples,
                            after_samples, pipeline.config.dataset.num_classes)
    print(f"\nSaved comparison figures to {args.output_dir}/figures/")
    print(f"Saved report-ready Markdown table to {report_path}")
    print("\nDone.")


def _save_comparison_chart(before_agg: dict, after_agg: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Material-match\naccuracy", "Mean confidence\nscore", "Mean verification\nscore"]
    before_vals = [before_agg["accuracy"], before_agg["mean_confidence"], before_agg["mean_verification_score"]]
    after_vals = [after_agg["accuracy"], after_agg["mean_confidence"], after_agg["mean_verification_score"]]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before (untrained)", color="#c0c0c0")
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After (trained)", color="#3070b3")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Performance: Before vs. After Training")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_confidence_histograms(before_results, after_results, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before_conf = [r.caption_result.confidence_score for r in before_results]
    after_conf = [r.caption_result.confidence_score for r in after_results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].hist(before_conf, bins=20, range=(0, 1), color="#c0c0c0")
    axes[0].set_title("Before (untrained)")
    axes[0].set_xlabel("Confidence score")
    axes[0].set_ylabel("Number of patches")
    axes[1].hist(after_conf, bins=20, range=(0, 1), color="#3070b3")
    axes[1].set_title("After (trained)")
    axes[1].set_xlabel("Confidence score")
    fig.suptitle("Confidence Score Distribution: Before vs. After Training")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_markdown_report(path, n_eval, before_agg, after_agg, sample_patches, before_samples, after_samples,
                            num_classes: int):
    chance_accuracy = 1.0 / num_classes
    lines = []
    lines.append("# Before / After Comparison: Untrained vs. Trained Pipeline\n")
    lines.append(f"Evaluated on **{n_eval}** randomly sampled real test patches "
                 f"(same patches used for both runs). Random-chance accuracy for "
                 f"{num_classes} classes is {chance_accuracy*100:.1f}%.\n")
    lines.append(
        "**Note on interpreting material-match accuracy below:** this pipeline's Stage 3 "
        "(knowledge-guided spectral retrieval) already contributes strong raw material-identification "
        "accuracy on its own, since its reference library is empirically fit from the same real "
        "labelled training pixels the classifier uses. Training therefore does not necessarily move raw "
        "top-1 accuracy dramatically in this architecture -- its clearest, most direct effect is on "
        "**confidence calibration and verification quality** (see mean/std confidence and verification "
        "score below). For a standard accuracy-focused ablation, report the dedicated encoder evaluation "
        "from `demos/train_encoder.py` instead (Overall Accuracy / Average Accuracy / Cohen's Kappa "
        "against a random-chance baseline), which isolates the classifier's own performance.\n")

    lines.append("## Aggregate Metrics\n")
    lines.append("| Metric | Before (untrained) | After (trained) | Change |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Material-match accuracy | {before_agg['accuracy']*100:.1f}% | "
                 f"{after_agg['accuracy']*100:.1f}% | "
                 f"{(after_agg['accuracy']-before_agg['accuracy'])*100:+.1f} pp |")
    lines.append(f"| Mean confidence score | {before_agg['mean_confidence']:.3f} | "
                 f"{after_agg['mean_confidence']:.3f} | "
                 f"{after_agg['mean_confidence']-before_agg['mean_confidence']:+.3f} |")
    lines.append(f"| Std. dev. of confidence | {before_agg['std_confidence']:.3f} | "
                 f"{after_agg['std_confidence']:.3f} | "
                 f"{after_agg['std_confidence']-before_agg['std_confidence']:+.3f} |")
    lines.append(f"| Mean verification score | {before_agg['mean_verification_score']:.3f} | "
                 f"{after_agg['mean_verification_score']:.3f} | "
                 f"{after_agg['mean_verification_score']-before_agg['mean_verification_score']:+.3f} |\n")

    lines.append("## Sample Captions\n")
    for patch in sample_patches:
        b = before_samples[patch.patch_id]
        a = after_samples[patch.patch_id]
        lines.append(f"### {patch.patch_id} (True class: {b.true_class})\n")
        lines.append(f"**Before** -- predicted: `{b.predicted_material}`, "
                     f"confidence: {b.caption_result.confidence_score:.3f} ({b.caption_result.confidence_band})")
        lines.append(f"> {b.caption_result.caption}\n")
        lines.append(f"**After** -- predicted: `{a.predicted_material}`, "
                     f"confidence: {a.caption_result.confidence_score:.3f} ({a.caption_result.confidence_band})")
        lines.append(f"> {a.caption_result.caption}\n")

    lines.append("## Figures\n")
    lines.append("- `figures/before_after_comparison.png` -- bar chart of aggregate metrics")
    lines.append("- `figures/before_after_confidence_hist.png` -- confidence score distributions\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
