"""Linear SVM training loop with parameter trajectory logging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .analysis_utils import TrajectoryMetrics, compute_projective_metrics


def _make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    tensor_x = torch.as_tensor(X, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@dataclass
class LinearSVMConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 5
    log_interval: int = 50
    record_interval: int = 10
    device: str = "cpu"


@dataclass
class LinearSVMRun:
    param_traj: List[torch.Tensor]
    train_losses: List[float]
    margins: List[float]
    metrics: TrajectoryMetrics
    final_weights: torch.Tensor
    bias: torch.Tensor
    history_steps: List[int]


class LinearSVMTrainer:
    def __init__(self, config: LinearSVMConfig):
        self.config = config

    def _hinge_loss(self, outputs: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        margin = 1 - labels * outputs
        hinge = torch.relu(margin).mean()
        reg = 0.5 * self.config.weight_decay * torch.sum(weight * weight)
        return hinge + reg

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        step_seed: int = 0,
    ) -> LinearSVMRun:
        cfg = self.config
        dataloader = _make_dataloader(
            X_train, y_train, batch_size=cfg.batch_size, shuffle=True, seed=step_seed
        )

        model = nn.Linear(X_train.shape[1], 1, bias=True, device=cfg.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

        param_traj: List[torch.Tensor] = []
        train_losses: List[float] = []
        margins: List[float] = []
        history_steps: List[int] = []

        step_idx = 0
        for epoch in range(cfg.num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self._hinge_loss(outputs, batch_y, model.weight)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_losses.append(float(loss.detach().cpu()))
                    margin_values = batch_y * outputs
                    margins.append(float(margin_values.min().detach().cpu()))

                    if step_idx % cfg.record_interval == 0:
                        weight_vec = model.weight.detach().flatten().cpu().clone()
                        bias = model.bias.detach().cpu().clone()
                        full_param = torch.cat([weight_vec, bias])
                        param_traj.append(full_param)
                        history_steps.append(step_idx)

                step_idx += 1

        final_weight = model.weight.detach().flatten().cpu().clone()
        final_bias = model.bias.detach().cpu().clone()
        param_traj.append(torch.cat([final_weight, final_bias]))
        history_steps.append(step_idx)

        metrics = compute_projective_metrics(param_traj, use_mask=True)

        return LinearSVMRun(
            param_traj=param_traj,
            train_losses=train_losses,
            margins=margins,
            metrics=metrics,
            final_weights=final_weight,
            bias=final_bias,
            history_steps=history_steps,
        )


def run_multiple_seeds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: LinearSVMConfig,
    seeds: Sequence[int],
) -> List[LinearSVMRun]:
    """Train multiple linear SVM runs with different data orders."""

    runs: List[LinearSVMRun] = []
    for seed in seeds:
        trainer = LinearSVMTrainer(config)
        runs.append(trainer.train(X_train, y_train, step_seed=seed))
    return runs
