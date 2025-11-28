"""High-level orchestration for linear, projected, and kernel SVM studies."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .data import MNISTBinaryConfig, load_binary_mnist
from .geometry import endpoint_svd, hilbert_distance_matrix, trajectory_geometry
from .kernel_svm import KernelSVMConfig, train_kernel_svm_runs
from .linear_svm import LinearSVMConfig, train_linear_svm
from .projections import apply_projection, make_random_projection

Projection = Callable[[np.ndarray], np.ndarray]


def run_linear_runs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    runs: int = 5,
    config: Optional[LinearSVMConfig] = None,
) -> Dict[str, List]:
    """Train multiple linear SVMs and collect geometry diagnostics."""

    if config is None:
        config = LinearSVMConfig()

    trajectories: List[Dict] = []
    geometries: List[Dict] = []
    endpoints: List[torch.Tensor] = []

    seeds = [config.seed + i for i in range(runs)]
    for seed in seeds:
        run_cfg = replace(config, seed=seed)
        traj = train_linear_svm(X_train, y_train, run_cfg)
        trajectories.append(traj)
        endpoints.append(torch.as_tensor(traj["weights"][-1]))
        geometries.append(trajectory_geometry(traj["weights"]))

    svd = endpoint_svd(endpoints)
    hilbert = hilbert_distance_matrix(endpoints)

    return {
        "trajectories": trajectories,
        "geometries": geometries,
        "endpoints": endpoints,
        "endpoint_svd": svd,
        "endpoint_hilbert": hilbert,
    }


def run_random_projection_suite(
    *,
    dimensions: Iterable[int],
    runs: int = 5,
    base_config: Optional[LinearSVMConfig] = None,
    data_config: Optional[MNISTBinaryConfig] = None,
    random_state: int = 0,
) -> Dict[int, Dict]:
    """Repeat the linear runs after applying Gaussian random projections."""

    X_train, X_test, y_train, y_test = load_binary_mnist(data_config)

    results: Dict[int, Dict] = {}
    for idx, dim in enumerate(dimensions):
        proj = make_random_projection(dim, random_state=random_state + idx)
        X_train_proj, _ = apply_projection(proj, X_train, X_test)
        results[dim] = run_linear_runs(
            X_train_proj,
            y_train,
            runs=runs,
            config=base_config,
        )

    return results


def run_kernel_suite(
    *,
    runs: int = 5,
    config: Optional[KernelSVMConfig] = None,
    data_config: Optional[MNISTBinaryConfig] = None,
) -> Dict[str, np.ndarray | List]:
    """Train multiple kernel SVMs and summarize dual-variable geometry."""

    X_train, _, y_train, _ = load_binary_mnist(data_config)
    kernel_runs = train_kernel_svm_runs(X_train, y_train, config=config, runs=runs)

    alphas = [r["alpha"] for r in kernel_runs]
    return {
        "runs": kernel_runs,
        "endpoint_svd": endpoint_svd(alphas),
        "endpoint_hilbert": hilbert_distance_matrix(alphas),
    }


if __name__ == "__main__":
    # A minimal smoke-test run that exercises the baseline pipeline without
    # performing any plotting. The defaults keep runtime reasonable.
    X_train, X_test, y_train, y_test = load_binary_mnist()

    linear_summary = run_linear_runs(X_train, y_train, runs=2, config=LinearSVMConfig(epochs=1))
    print("Linear endpoint singular values:", linear_summary["endpoint_svd"])

    rp_summary = run_random_projection_suite(dimensions=[256], runs=2, base_config=LinearSVMConfig(epochs=1))
    print("Random projection dims:", list(rp_summary.keys()))

    kernel_summary = run_kernel_suite(runs=2, config=KernelSVMConfig(max_iter=1000))
    print("Kernel endpoint singular values:", kernel_summary["endpoint_svd"])
