"""Simple linear probes for activation-space analysis."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ProbeSplit:
    """Train/test split for a binary linear probe."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def make_binary_probe_split(
    positive_vectors: Sequence[Sequence[float]],
    negative_vectors: Sequence[Sequence[float]],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> ProbeSplit:
    """Create a balanced train/test split for binary probe training."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not positive_vectors or not negative_vectors:
        raise ValueError("both classes must contain at least one example")

    rng = random.Random(seed)
    pos_idx = list(range(len(positive_vectors)))
    neg_idx = list(range(len(negative_vectors)))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_cutoff = min(max(int(len(pos_idx) * train_fraction), 1), len(pos_idx) - 1)
    neg_cutoff = min(max(int(len(neg_idx) * train_fraction), 1), len(neg_idx) - 1)

    train_vectors = [positive_vectors[i] for i in pos_idx[:pos_cutoff]] + [negative_vectors[i] for i in neg_idx[:neg_cutoff]]
    train_labels = [1.0] * pos_cutoff + [0.0] * neg_cutoff
    test_vectors = [positive_vectors[i] for i in pos_idx[pos_cutoff:]] + [negative_vectors[i] for i in neg_idx[neg_cutoff:]]
    test_labels = [1.0] * (len(pos_idx) - pos_cutoff) + [0.0] * (len(neg_idx) - neg_cutoff)

    return ProbeSplit(
        train_x=torch.tensor(train_vectors, dtype=torch.float32),
        train_y=torch.tensor(train_labels, dtype=torch.float32),
        test_x=torch.tensor(test_vectors, dtype=torch.float32),
        test_y=torch.tensor(test_labels, dtype=torch.float32),
    )


def standardize_probe_split(split: ProbeSplit) -> ProbeSplit:
    """Z-score standardize features using the train-set statistics."""
    mean = split.train_x.mean(dim=0, keepdim=True)
    std = split.train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return ProbeSplit(
        train_x=(split.train_x - mean) / std,
        train_y=split.train_y,
        test_x=(split.test_x - mean) / std,
        test_y=split.test_y,
    )


def train_logistic_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 0.01,
) -> torch.nn.Linear:
    """Fit a single-layer logistic probe."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_targets = train_y.unsqueeze(1)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_fn(logits, train_targets)
        loss.backward()
        optimizer.step()

    return model


def evaluate_logistic_probe(
    model: torch.nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate a binary logistic probe."""
    with torch.no_grad():
        logits = model(features).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    accuracy = float((preds == labels).float().mean().item())
    positive_rate = float(preds.mean().item())
    return {
        "accuracy": accuracy,
        "positive_prediction_rate": positive_rate,
        "avg_logit": float(logits.mean().item()),
    }


def fit_and_evaluate_binary_probe(
    positive_vectors: Sequence[Sequence[float]],
    negative_vectors: Sequence[Sequence[float]],
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 0.01,
) -> Dict[str, Any]:
    """Train and evaluate a binary logistic probe on two activation groups."""
    split = standardize_probe_split(
        make_binary_probe_split(
            positive_vectors,
            negative_vectors,
            train_fraction=train_fraction,
            seed=seed,
        )
    )
    model = train_logistic_probe(
        split.train_x,
        split.train_y,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    train_metrics = evaluate_logistic_probe(model, split.train_x, split.train_y)
    test_metrics = evaluate_logistic_probe(model, split.test_x, split.test_y)
    return {
        "train": train_metrics,
        "test": test_metrics,
        "num_train": int(split.train_y.numel()),
        "num_test": int(split.test_y.numel()),
    }