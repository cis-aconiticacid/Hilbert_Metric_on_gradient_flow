"""Linear SVM training with explicit parameter trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LinearSVMConfig:
    """Hyperparameters controlling the primal linear SVM run."""

    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 128
    c: float = 1.0
    seed: int = 0
    record_every: int = 1
    device: str = "cpu"


@torch.no_grad()
def _param_vector(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Concatenate weight and bias so geometry utilities can treat them uniformly."""

    return torch.cat([weight.view(-1), bias.view(1)])


def train_linear_svm(
    X_train: np.ndarray, y_train: np.ndarray, config: LinearSVMConfig | None = None
) -> Dict[str, List]:
    """Train a linear SVM with hinge loss and collect parameter trajectories.

    Args:
        X_train: Training features of shape ``(n_samples, d)``.
        y_train: Labels in ``{-1, +1}``.
        config: Optional ``LinearSVMConfig``. Defaults mirror the low-epoch
            exploratory setting used in the notebook experiments.

    Returns:
        Dictionary containing trajectories and simple metrics:
            - ``weights``: list of parameter vectors (including bias as the last
              coordinate).
            - ``losses``: per-step hinge + L2 objective values.
            - ``train_accuracy``: accuracy on the full training set at the same
              recording cadence as ``record_every``.
            - ``margin_min``: minimum signed margin over the recorded step.
    """

    if config is None:
        config = LinearSVMConfig()

    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    features = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    dataset = TensorDataset(features, labels)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator(device=device).manual_seed(config.seed),
    )

    d = features.shape[1]
    weight = torch.zeros(d, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)

    optimizer = torch.optim.SGD([weight, bias], lr=config.lr)

    traj: Dict[str, List] = {"weights": [], "losses": [], "train_accuracy": [], "margin_min": []}
    step = 0

    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            margin = batch_y * (batch_x @ weight + bias)
            hinge = torch.clamp(1 - margin, min=0)
            loss = 0.5 * torch.sum(weight * weight) + config.c * hinge.mean()
            loss.backward()
            optimizer.step()

            if step % config.record_every == 0:
                with torch.no_grad():
                    traj["weights"].append(_param_vector(weight, bias).cpu().clone())
                    traj["losses"].append(loss.item())

                    full_margin = labels * (features @ weight + bias)
                    margin_min = float(full_margin.min().item())
                    traj["margin_min"].append(margin_min)

                    predictions = torch.sign(full_margin)
                    accuracy = float((predictions == labels).float().mean().item())
                    traj["train_accuracy"].append(accuracy)

            step += 1

    return traj
