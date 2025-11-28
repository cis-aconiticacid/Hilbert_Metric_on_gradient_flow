"""Kernel SVM helpers focusing on dual variables for Hilbert analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from sklearn.svm import SVC

from .analysis_utils import TrajectoryMetrics, compute_projective_metrics
from .data_utils import StandardizedSplit


@dataclass
class KernelConfig:
    C: float = 5.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    degree: int = 3


@dataclass
class KernelRun:
    model: SVC
    dual_alpha: torch.Tensor
    support_indices: np.ndarray
    metrics: TrajectoryMetrics


def train_kernel_svm(split: StandardizedSplit, config: KernelConfig) -> KernelRun:
    """Fit a kernel SVM and extract the dual coefficients."""

    svm = SVC(
        C=config.C,
        kernel=config.kernel,
        gamma=config.gamma,
        degree=config.degree,
    )
    svm.fit(split.X_train, split.y_train)

    dual_coef = svm.dual_coef_[0]
    support_indices = svm.support_

    full_alpha = np.zeros(len(split.X_train), dtype=np.float64)
    full_alpha[support_indices] = dual_coef
    alpha_tensor = torch.from_numpy(np.abs(full_alpha) + 1e-8)

    metrics = compute_projective_metrics([alpha_tensor])

    return KernelRun(
        model=svm,
        dual_alpha=alpha_tensor,
        support_indices=support_indices,
        metrics=metrics,
    )


def multi_seed_kernel(
    split: StandardizedSplit,
    *,
    config: KernelConfig,
    seeds: Sequence[int],
) -> List[KernelRun]:
    """Train kernel SVMs with different shuffles of the training data."""

    runs: List[KernelRun] = []
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(len(split.X_train))
        permuted = StandardizedSplit(
            X_train=split.X_train[perm],
            X_test=split.X_test,
            y_train=split.y_train[perm],
            y_test=split.y_test,
            scaler=split.scaler,
        )
        runs.append(train_kernel_svm(permuted, config))
    return runs
