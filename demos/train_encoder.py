#!/usr/bin/env python3
"""
Phase A: Train the Stage 2 encoder (spectral + spatial + metadata + fusion)
end-to-end against patch class labels.

This is what turns Stage 2's embeddings from "architecturally correct
but untrained" into genuinely discriminative features, and produces the
standard WHU-Hi benchmark metrics (Overall Accuracy, Average Accuracy,
Cohen's Kappa) an IEEE write-up needs.

Usage
    PYTHONPATH=src python demos/train_encoder.py --config configs/config.yaml \
        --epochs 60 --batch-size 64 --lr 1e-3 --patience 8 \
        --test-subsample 5000

`--test-subsample` caps how many of the (potentially huge, WHU-Hi's
official Test<N>.mat is "all remaining labelled pixels") test patches
are evaluated on, since a 200k+-patch test set evaluated one-by-one in
Python is otherwise very slow. Batched inference makes this much faster
than the reference numpy pipeline, but a subsample keeps iteration fast
during development; omit the flag (or set to 0) to evaluate on the full
test set for final reported numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from hsi_caption.pipeline import HSICaptionPipeline
from hsi_caption.torch_modules import EncoderNet


class PatchTorchDataset(Dataset):
    """Wraps Stage-1 `Patch` objects for PyTorch training."""

    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        p = patch.cube_data.shape[0] // 2
        center_spectrum = patch.cube_data[p, p, :].astype(np.float32)
        full_patch = patch.cube_data.astype(np.float32)
        metadata = np.array([
            patch.metadata.get("row_norm", 0.0), patch.metadata.get("col_norm", 0.0),
            patch.metadata.get("local_class_purity", 0.0),
        ], dtype=np.float32)
        label = patch.center_label
        return center_spectrum, full_patch, metadata, label


def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for spectrum, patch, meta, label in loader:
            spectrum, patch, meta = spectrum.to(device), patch.to(device), meta.to(device)
            _, logits = model(spectrum, patch, meta)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(label.numpy().tolist())

    oa = accuracy_score(all_labels, all_preds)
    label_ids = list(range(len(class_names)))
    per_class_recall = classification_report(all_labels, all_preds, labels=label_ids,
                                               output_dict=True, zero_division=0)
    aa = np.mean([v["recall"] for k, v in per_class_recall.items() if k.isdigit()])
    kappa = cohen_kappa_score(all_labels, all_preds, labels=label_ids)
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    return {"OA": oa, "AA": aa, "Kappa": kappa, "confusion_matrix": cm, "y_true": all_labels, "y_pred": all_preds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    parser.add_argument("--test-subsample", type=int, default=5000,
                         help="cap on test patches evaluated; 0 = use full test set")
    parser.add_argument("--output", default="outputs/encoder_weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[Stage 1] Loading and preprocessing dataset...")
    pipeline = HSICaptionPipeline(args.config)
    cube = pipeline.load_cube()
    summary, stats, norm_cube, norm_params, patch_dataset = pipeline.preprocess(cube)
    print(f"train={len(patch_dataset.train)} val={len(patch_dataset.val)} test={len(patch_dataset.test)}")

    test_patches = patch_dataset.test
    if args.test_subsample and len(test_patches) > args.test_subsample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(test_patches), size=args.test_subsample, replace=False)
        test_patches = [test_patches[i] for i in idx]
        print(f"(evaluating on a {args.test_subsample}-patch random subsample of the {len(patch_dataset.test)}-patch "
              f"official test set; pass --test-subsample 0 for the full set)")

    train_loader = DataLoader(PatchTorchDataset(patch_dataset.train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PatchTorchDataset(patch_dataset.val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PatchTorchDataset(test_patches), batch_size=args.batch_size, shuffle=False)

    cfg = pipeline.config
    model = EncoderNet(
        num_bands=cfg.dataset.num_bands, num_classes=cfg.dataset.num_classes,
        spectral_dim=cfg.encoder.spectral_embed_dim, spatial_dim=cfg.encoder.spatial_embed_dim,
        metadata_dim=cfg.encoder.metadata_embed_dim, fused_dim=cfg.encoder.fused_embed_dim,
        kernels=cfg.encoder.spectral_conv_kernels, spatial_tokens=cfg.encoder.spatial_patch_tokens,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Phase A] Training encoder for up to {args.epochs} epochs (patience={args.patience})...")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()
        for spectrum, patch, meta, label in train_loader:
            spectrum, patch, meta, label = spectrum.to(device), patch.to(device), meta.to(device), label.to(device)
            optimizer.zero_grad()
            _, logits = model(spectrum, patch, meta)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for spectrum, patch, meta, label in val_loader:
                spectrum, patch, meta, label = spectrum.to(device), patch.to(device), meta.to(device), label.to(device)
                _, logits = model(spectrum, patch, meta)
                val_loss += criterion(logits, label).item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        print(f"  epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {
        "num_bands": cfg.dataset.num_bands, "num_classes": cfg.dataset.num_classes,
        "spectral_dim": cfg.encoder.spectral_embed_dim, "spatial_dim": cfg.encoder.spatial_embed_dim,
        "metadata_dim": cfg.encoder.metadata_embed_dim, "fused_dim": cfg.encoder.fused_embed_dim,
        "kernels": list(cfg.encoder.spectral_conv_kernels), "spatial_tokens": cfg.encoder.spatial_patch_tokens,
    }}, args.output)
    print(f"\nSaved trained encoder weights to {args.output}")

    print("\n[Evaluation] Running trained encoder on the test set...")
    metrics = evaluate(model, test_loader, device, cfg.dataset.class_names)
    print(f"\nOverall Accuracy (OA):  {metrics['OA']*100:.2f}%")
    print(f"Average Accuracy (AA):  {metrics['AA']*100:.2f}%")
    print(f"Cohen's Kappa:          {metrics['Kappa']:.4f}")
    print("\nPer-class report:")
    print(classification_report(metrics["y_true"], metrics["y_pred"],
                                 labels=list(range(len(cfg.dataset.class_names))),
                                 target_names=cfg.dataset.class_names, zero_division=0))

    # confusion matrix figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(len(cfg.dataset.class_names)))
    ax.set_yticks(range(len(cfg.dataset.class_names)))
    ax.set_xticklabels(cfg.dataset.class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(cfg.dataset.class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (OA={metrics['OA']*100:.1f}%, Kappa={metrics['Kappa']:.3f})")
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(cfg.output_dir, "figures", "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    print(f"\nSaved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
