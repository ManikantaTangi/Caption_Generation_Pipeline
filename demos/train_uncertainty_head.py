#!/usr/bin/env python3
"""
Phase B: Train Stage 7's Bayesian classifier head on Stage 3's actual
knowledge-guided semantic_vector (fused_vector + spectral-library
similarity/fraction features), using the frozen Phase-A-trained encoder
to produce fused_vector.

This is what makes Stage 7's MC-Dropout uncertainty (epistemic/aleatoric
entropy, confidence score) meaningful end-to-end, since it now reflects
a genuinely trained classifier operating on the full neuro-symbolic
feature vector -- not just the raw encoder output.

Usage
    PYTHONPATH=src python demos/train_uncertainty_head.py \
        --config configs/config.yaml --encoder-weights outputs/encoder_weights.pt \
        --epochs 60 --patience 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score

from hsi_caption.pipeline import HSICaptionPipeline
from hsi_caption.torch_modules import BayesianHead, EncoderNet


def load_trained_encoder(weights_path: str, device) -> EncoderNet:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = EncoderNet(
        num_bands=cfg["num_bands"], num_classes=cfg["num_classes"], spectral_dim=cfg["spectral_dim"],
        spatial_dim=cfg["spatial_dim"], metadata_dim=cfg["metadata_dim"], fused_dim=cfg["fused_dim"],
        kernels=cfg["kernels"], spatial_tokens=cfg["spatial_tokens"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def encoder_fused_vector(model: EncoderNet, patch, device) -> np.ndarray:
    p = patch.cube_data.shape[0] // 2
    spectrum = torch.from_numpy(patch.cube_data[p, p, :].astype(np.float32)).unsqueeze(0).to(device)
    full_patch = torch.from_numpy(patch.cube_data.astype(np.float32)).unsqueeze(0).to(device)
    metadata = torch.tensor([[
        patch.metadata.get("row_norm", 0.0), patch.metadata.get("col_norm", 0.0),
        patch.metadata.get("local_class_purity", 0.0),
    ]], dtype=torch.float32).to(device)
    with torch.no_grad():
        fused, _ = model(spectrum, full_patch, metadata)
    return fused.squeeze(0).cpu().numpy()


def build_semantic_vectors(pipeline: HSICaptionPipeline, encoder: EncoderNet, patches, device):
    """Runs frozen-encoder Stage 2 + Stage 3 (knowledge engine) to get each
    patch's real semantic_vector, mirroring exactly what pipeline.run_patch()
    does for Stage 2/3 -- so the vectors this trains on match production use."""
    from hsi_caption.datatypes import FusedEmbedding

    vectors, labels = [], []
    for patch in patches:
        fused_vec = encoder_fused_vector(encoder, patch, device)
        fused_embedding = FusedEmbedding(
            patch_id=patch.patch_id, spectral_vector=np.zeros(1), spatial_vector=np.zeros(1),
            metadata_vector=np.zeros(1), fused_vector=fused_vec,
        )
        ke = pipeline.knowledge_engine.process(patch, fused_embedding)
        vectors.append(ke.semantic_vector.astype(np.float32))
        labels.append(patch.center_label)
    return np.stack(vectors, axis=0), np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--encoder-weights", default="outputs/encoder_weights.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output", default="outputs/uncertainty_head_weights.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[Stage 1] Loading and preprocessing dataset...")
    pipeline = HSICaptionPipeline(args.config)
    cube = pipeline.load_cube()
    summary, stats, norm_cube, norm_params, patch_dataset = pipeline.preprocess(cube)
    pipeline.build_engines(cube.wavelengths_nm, norm_params)  # builds Stage 3's knowledge_engine

    print(f"\nLoading frozen trained encoder from {args.encoder_weights}...")
    encoder = load_trained_encoder(args.encoder_weights, device)

    print("\nBuilding Stage-3 knowledge-guided semantic vectors for train/val patches "
          "(runs frozen encoder + spectral retrieval per patch)...")
    X_train, y_train = build_semantic_vectors(pipeline, encoder, patch_dataset.train, device)
    X_val, y_val = build_semantic_vectors(pipeline, encoder, patch_dataset.val, device)
    print(f"train semantic vectors: {X_train.shape}, val: {X_val.shape}")

    cfg = pipeline.config
    semantic_dim = X_train.shape[1]
    model = BayesianHead(semantic_dim, cfg.dataset.num_classes, cfg.encoder.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    print(f"\n[Phase B] Training Bayesian head for up to {args.epochs} epochs...")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = accuracy_score(y_val, val_logits.argmax(dim=1).cpu().numpy())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs} | train_loss={loss.item():.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "semantic_dim": semantic_dim,
                "num_classes": cfg.dataset.num_classes, "dropout": cfg.encoder.dropout}, args.output)
    print(f"\nSaved trained Bayesian head weights to {args.output}")

    # Fit Stage 7's temperature-scaling calibration on the validation set,
    # using this newly trained head's (deterministic, dropout-off) probabilities.
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
    from hsi_caption.stage7_uncertainty.calibration import Calibration
    calibration = Calibration(cfg.uncertainty.temperature)
    fitted_temp = calibration.fit(list(zip(val_probs, y_val.tolist())))
    print(f"Fitted calibration temperature: {fitted_temp:.2f} (update uncertainty.temperature in "
          f"configs/config.yaml to this value for production use)")


if __name__ == "__main__":
    main()
