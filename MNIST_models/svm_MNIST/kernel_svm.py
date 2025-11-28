"""Kernel SVM helpers that expose dual variables for Hilbert analysis."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class KernelSVMConfig:
    kernel: str = "rbf"
    c: float = 5.0
    gamma: str | float = "scale"
    degree: int = 3
    coef0: float = 0.0
    max_iter: int = -1
    random_state: int = 0


def _build_kernel_svm(config: KernelSVMConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=config.kernel,
                    C=config.c,
                    gamma=config.gamma,
                    degree=config.degree,
                    coef0=config.coef0,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _extract_dual_alphas(model: Pipeline, y_train: np.ndarray) -> torch.Tensor:
    svm = model.named_steps["svm"]
    support = svm.support_
    dual = svm.dual_coef_[0]
    labels = y_train[support]

    alpha_positive = dual * labels  # recover non-negative dual variables
    full_alpha = np.zeros_like(y_train, dtype=float)
    full_alpha[support] = alpha_positive
    return torch.from_numpy(full_alpha.astype(np.float32))


def train_kernel_svm_runs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: KernelSVMConfig | None = None,
    runs: int = 5,
    seeds: Iterable[int] | None = None,
) -> List[Dict[str, torch.Tensor | Pipeline | KernelSVMConfig]]:
    """Train multiple kernel SVMs and return their dual variables."""

    if config is None:
        config = KernelSVMConfig()

    if seeds is None:
        seeds = [config.random_state + i for i in range(runs)]

    results: List[Dict[str, torch.Tensor | Pipeline | KernelSVMConfig]] = []
    for seed in seeds:
        run_config = replace(config, random_state=seed)
        model = _build_kernel_svm(run_config)
        model.fit(X_train, y_train)
        alpha = _extract_dual_alphas(model, y_train)
        results.append({"model": model, "alpha": alpha, "config": run_config})

    return results
