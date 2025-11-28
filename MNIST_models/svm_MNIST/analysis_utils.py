"""Projective metrics and spectral summaries for SVM experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

# Import graph_print_analysis from MNIST_models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MNIST_models import graph_print_analysis as gpa  # type: ignore


@dataclass
class TrajectoryMetrics:
    hilbert_to_final: List[float]
    hilbert_between: List[float]
    hilbert_to_init: List[float]
    l2_to_final: List[float]
    cosine_to_final: List[float]


def _prepare_positive_traj(param_traj: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """Clamp parameters to the positive cone for Hilbert analysis."""

    processed = []
    for p in param_traj:
        flat = p.detach().view(-1)
        flat = flat.abs() + 1e-8
        processed.append(flat)
    return processed


def compute_projective_metrics(
    param_traj: Iterable[torch.Tensor],
    *,
    threshold: float = 1e-5,
    use_mask: bool = True,
) -> TrajectoryMetrics:
    """Compute Hilbert, Euclidean, and angular metrics against the final point."""

    processed = _prepare_positive_traj(param_traj)
    w_star = processed[-1]

    hilbert_res = gpa.compute_hilbert_metrics(
        processed,
        threshold=threshold,
        if_mask=use_mask,
        if_threshold=not use_mask,
        if_self_adaptive=False,
        w_star=w_star,
    )

    w_star_unit = w_star / (w_star.norm() + 1e-12)
    l2_to_final = []
    cosine_to_final = []
    for w in processed:
        l2_to_final.append(float(torch.norm(w - w_star)))
        w_norm = w.norm() + 1e-12
        cosine_to_final.append(float(torch.dot(w, w_star_unit) / w_norm))

    return TrajectoryMetrics(
        hilbert_to_final=hilbert_res["hilbert_to_final"],
        hilbert_between=hilbert_res["hilbert_between"],
        hilbert_to_init=hilbert_res["hilbert_to_init"],
        l2_to_final=l2_to_final,
        cosine_to_final=cosine_to_final,
    )


def stack_for_svd(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Stack a set of 1D tensors into a 2D matrix for SVD/PCA."""

    return torch.stack([v.detach().view(-1) for v in vectors], dim=0)


def svd_spectrum(vectors: Iterable[torch.Tensor]) -> np.ndarray:
    """Return singular values of stacked vectors (descending order)."""

    matrix = stack_for_svd(vectors).double()
    _, s, _ = torch.svd(matrix)
    return s.cpu().numpy()
