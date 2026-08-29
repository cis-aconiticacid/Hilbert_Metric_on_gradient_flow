"""Experiment orchestration for linear, projected, and kernel SVMs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from .analysis_utils import TrajectoryMetrics, svd_spectrum
from .data_utils import StandardizedSplit, load_and_project
from .kernel_svm import KernelConfig, KernelRun, multi_seed_kernel
from .linear_svm import (
    LinearSVMConfig,
    LinearSVMRun,
    LinearSVMTrainer,
    run_multiple_seeds,
)


@dataclass
class LinearSweepResult:
    runs: List[LinearSVMRun]
    singular_values: np.ndarray


@dataclass
class KernelSweepResult:
    runs: List[KernelRun]
    singular_values: np.ndarray


def run_linear_sweep(
    *,
    split: StandardizedSplit,
    seeds: Sequence[int],
    config: LinearSVMConfig,
) -> LinearSweepResult:
    runs = run_multiple_seeds(split.X_train, split.y_train, config=config, seeds=seeds)
    final_vecs = [torch.cat([r.final_weights, r.bias]) for r in runs]
    spectrum = svd_spectrum(final_vecs)
    return LinearSweepResult(runs=runs, singular_values=spectrum)


def run_projection_grid(
    *,
    dims: Iterable[int | None],
    seeds: Sequence[int],
    base_config: LinearSVMConfig,
    data_kwargs: Dict,
) -> Dict[int | None, LinearSweepResult]:
    results: Dict[int | None, LinearSweepResult] = {}
    for d in dims:
        split = load_and_project(target_dim=d, **data_kwargs)
        results[d] = run_linear_sweep(split=split, seeds=seeds, config=base_config)
    return results


def run_kernel_sweep(
    *,
    split: StandardizedSplit,
    seeds: Sequence[int],
    config: KernelConfig,
) -> KernelSweepResult:
    runs = multi_seed_kernel(split, config=config, seeds=seeds)
    alphas = [r.dual_alpha for r in runs]
    spectrum = svd_spectrum(alphas)
    return KernelSweepResult(runs=runs, singular_values=spectrum)
