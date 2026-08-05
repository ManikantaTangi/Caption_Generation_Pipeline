"""
Lightweight NumPy neural-network primitives.

Why NumPy instead of PyTorch here
----------------------------------
The reference implementation targets portability (installs in seconds,
runs on CPU-only sandboxes/CI, zero GPU/CUDA dependency) while preserving
the *exact* architectural interfaces (Dense, Conv1D, MultiHeadAttention,
Dropout) that a production PyTorch/`nn.Module` implementation would use.
Every class below is a drop-in conceptual equivalent of its `torch.nn`
counterpart; swapping the backend means re-implementing these same
class signatures with `torch.nn.Linear`, `torch.nn.Conv1d`, and
`torch.nn.MultiheadAttention` and wiring in autograd + an optimizer
(see docs/ARCHITECTURE.md, Stage 2, Tech Stack, for the migration note).

These layers are inference-only (forward pass, Xavier-initialized
weights). No backpropagation is implemented; that is out of scope for
this architectural reference pipeline, whose contribution is the
end-to-end *system design*, not a from-scratch autodiff engine.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def xavier_init(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


class Dense:
    """Fully-connected layer: y = ReLU(xW + b) if `activation` else xW + b."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator, activation: bool = True) -> None:
        self.W = xavier_init(rng, (in_dim, out_dim))
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        return relu(out) if self.activation else out


class Conv1D:
    """1D convolution over the spectral axis: (L, C_in) -> (L', C_out)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rng: np.random.Generator, stride: int = 2) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.W = xavier_init(rng, (kernel_size, in_channels, out_channels))
        self.b = np.zeros(out_channels, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (L, C_in) -> (L_out, C_out)."""
        length = x.shape[0]
        out_len = (length - self.kernel_size) // self.stride + 1
        out = np.zeros((out_len, self.W.shape[-1]), dtype=np.float32)
        for i in range(out_len):
            start = i * self.stride
            window = x[start:start + self.kernel_size]  # (K, C_in)
            out[i] = np.tensordot(window, self.W, axes=([0, 1], [0, 1])) + self.b
        return relu(out)


class Dropout:
    """Inference-mode-aware dropout, used for Monte Carlo Dropout (Stage 7).

    Unlike standard inference-time dropout (disabled at test time), this
    layer supports an explicit `training=True` override so Stage 7 can
    keep dropout *active* at inference to sample the predictive
    distribution (Gal & Ghahramani, 2016 - MC Dropout as approximate
    Bayesian inference).
    """

    def __init__(self, rate: float, rng: np.random.Generator) -> None:
        self.rate = rate
        self.rng = rng

    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.rate <= 0:
            return x
        mask = (self.rng.random(x.shape) > self.rate).astype(np.float32)
        return x * mask / (1.0 - self.rate)


class SingleHeadAttention:
    """Simplified single-head scaled dot-product self-attention over tokens.

    Stands in for a Vision-Transformer attention block operating on
    patch tokens (Stage 2 spatial encoder) or on modality embeddings
    (Stage 2 feature fusion).
    """

    def __init__(self, dim: int, rng: np.random.Generator) -> None:
        self.Wq = xavier_init(rng, (dim, dim))
        self.Wk = xavier_init(rng, (dim, dim))
        self.Wv = xavier_init(rng, (dim, dim))
        self.scale = 1.0 / np.sqrt(dim)

    def __call__(self, tokens: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """tokens: (N, dim) -> (context: (N, dim), attn_weights: (N, N))."""
        q = tokens @ self.Wq
        k = tokens @ self.Wk
        v = tokens @ self.Wv
        scores = (q @ k.T) * self.scale
        weights = softmax(scores, axis=-1)
        context = weights @ v
        return context, weights
