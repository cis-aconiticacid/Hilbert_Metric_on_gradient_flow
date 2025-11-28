"""Geometric diagnostics for SVM trajectories and endpoints."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from environment.hilbert_distance import hilbert_analysis


def _ensure_tensor_list(vectors: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.as_tensor(v).detach().float() for v in vectors]


def trajectory_geometry(
    weights: Iterable[torch.Tensor], *, threshold: float = 1e-5, mask: bool = True
) -> Dict[str, List[float]]:
    """Compute Hilbert, Euclidean, and cosine metrics against the final iterate."""

    traj = _ensure_tensor_list(weights)
    if len(traj) < 2:
        raise ValueError("Trajectory must contain at least two points.")

    w_star = traj[-1]
    stacked = torch.stack(traj)

    l2_distance = torch.norm(stacked - w_star, dim=1).tolist()

    cosine = torch.nn.functional.cosine_similarity(stacked, w_star.unsqueeze(0), dim=1)
    cosine = torch.clamp(cosine, -1.0, 1.0).tolist()

    hilbert_results = hilbert_analysis.analysis_distance_on_cone(
        param_traj=[t.abs() + 1e-8 for t in traj],
        w_star=w_star.abs() + 1e-8,
        threshold=threshold,
        ifmask=mask,
        if_threshold=not mask,
        if_self_adaptive=False,
    )

    return {
        "hilbert_to_final": hilbert_results["hilbert_to_final"],
        "hilbert_to_init": hilbert_results["hilbert_to_init"],
        "hilbert_between": hilbert_results["hilbert_between"],
        "l2_to_final": l2_distance,
        "cosine_to_final": cosine,
    }


def endpoint_svd(endpoints: Iterable[torch.Tensor]) -> np.ndarray:
    """Compute singular values of stacked endpoint vectors."""

    vectors = _ensure_tensor_list(endpoints)
    matrix = torch.stack(vectors).numpy()
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return s


def hilbert_distance_matrix(endpoints: Iterable[torch.Tensor], eps: float = 1e-8) -> np.ndarray:
    """Pairwise Hilbert distances between positive endpoint vectors."""

    vectors = [torch.clamp(torch.as_tensor(v).float().view(-1).abs(), min=eps) for v in endpoints]
    n = len(vectors)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hilbert_analysis.hilbert_distance(vectors[i], vectors[j])
            dist[i, j] = dist[j, i] = d
    return dist
