"""Data loading helpers for MNIST SVM experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

# Ensure the project root is importable when running from notebooks
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_mnist_binary(
    digit_pos: int = 3,
    digit_neg: int = 8,
    *,
    max_samples: int | None = 20000,
    test_size: float = 0.2,
    random_state: int = 42,
    flatten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a fixed MNIST split for two digits.

    Args:
        digit_pos: Digit mapped to ``+1``.
        digit_neg: Digit mapped to ``-1``.
        max_samples: Optional cap on the total number of samples taken from the
            original training set before splitting.
        test_size: Fraction of samples reserved for evaluation.
        random_state: Seed controlling shuffling and optional subsampling.
        flatten: Whether to flatten 28x28 images to vectors.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` with labels in ``{-1, +1}``.
    """

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        download=True,
        transform=transform,
    )

    targets = dataset.targets
    mask = (targets == digit_pos) | (targets == digit_neg)
    data = dataset.data[mask].float() / 255.0
    targets = targets[mask]

    if max_samples is not None and max_samples < len(data):
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(data), generator=generator)[:max_samples]
        data = data[indices]
        targets = targets[indices]

    if flatten:
        data = data.view(len(data), -1)

    y = torch.where(targets == digit_pos, 1, -1).numpy()
    X = data.numpy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_random_projection(
    X: np.ndarray,
    target_dim: int,
    *,
    random_state: int = 0,
    mode: str = "gaussian",
) -> np.ndarray:
    """Project features into a new dimension with a fixed random matrix."""

    rng = np.random.default_rng(random_state)
    if mode == "gaussian":
        proj_matrix = rng.normal(loc=0.0, scale=1.0, size=(X.shape[1], target_dim))
    elif mode == "orthogonal":
        gaussian = rng.normal(loc=0.0, scale=1.0, size=(X.shape[1], target_dim))
        q, _ = np.linalg.qr(gaussian)
        proj_matrix = q[:, :target_dim]
    else:
        raise ValueError(f"Unknown projection mode: {mode}")

    scaled = proj_matrix / np.sqrt(target_dim)
    return X @ scaled


@dataclass
class StandardizedSplit:
    """Standardized train/test split with a stored scaler."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def standardize_split(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features using a train-fitted :class:`StandardScaler`."""

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, scaler


def load_and_project(
    *,
    digit_pos: int = 3,
    digit_neg: int = 8,
    max_samples: int | None = 20000,
    test_size: float = 0.2,
    random_state: int = 42,
    target_dim: int | None = None,
    projection_mode: str = "gaussian",
) -> StandardizedSplit:
    """Convenience wrapper to get a standardized split with optional projection."""

    X_train, X_test, y_train, y_test = load_mnist_binary(
        digit_pos=digit_pos,
        digit_neg=digit_neg,
        max_samples=max_samples,
        test_size=test_size,
        random_state=random_state,
        flatten=True,
    )

    if target_dim is not None:
        X_train = apply_random_projection(
            X_train, target_dim, random_state=random_state, mode=projection_mode
        )
        X_test = apply_random_projection(
            X_test, target_dim, random_state=random_state, mode=projection_mode
        )

    X_train_std, X_test_std, scaler = standardize_split(X_train, X_test)
    return StandardizedSplit(X_train_std, X_test_std, y_train, y_test, scaler)
