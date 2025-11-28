"""Embedding utilities for random projection experiments."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from sklearn.random_projection import GaussianRandomProjection


Projection = Callable[[np.ndarray], np.ndarray]


def make_random_projection(dim: int, *, random_state: int | None = None) -> Projection:
    """Create a Gaussian random projection mapping to ``dim`` dimensions."""

    projector = GaussianRandomProjection(n_components=dim, random_state=random_state)

    def transform(x: np.ndarray) -> np.ndarray:
        return projector.fit_transform(x)

    return transform


def apply_projection(
    projection: Projection, X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a projection function to train and test matrices."""

    return projection(X_train), projection(X_test)
