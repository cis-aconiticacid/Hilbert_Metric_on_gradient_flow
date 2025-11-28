"""RBF-kernel SVM training utilities for MNIST.

This module loads the MNIST digits dataset, flattens the images, and trains
an RBF-kernel support vector machine. Utility functions expose loading,
training, and accuracy evaluation so they can be reused in scripts or
notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets, transforms


def load_mnist_data(
    *,
    max_samples: int | None = 20000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST digits and return flattened train/test splits.

    Args:
        max_samples: Optional cap on the number of training samples loaded
            from the original 60k MNIST training examples. When ``None``
            the full training set is used. The selected subset is shuffled
            with the provided ``random_state`` before splitting.
        test_size: Fraction of samples reserved for evaluation.
        random_state: Seed used for the deterministic shuffle/split.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` where each feature matrix has
        shape ``(n_samples, 784)`` with pixel intensities in ``[0, 1]``.
    """

    transform = transforms.Compose(
        [transforms.ToTensor()],
    )

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    data = dataset.data.float() / 255.0  # Normalize to [0, 1]
    targets = dataset.targets

    if max_samples is not None and max_samples < len(data):
        generator = torch.Generator().manual_seed(random_state)
        indices = torch.randperm(len(data), generator=generator)[:max_samples]
        data = data[indices]
        targets = targets[indices]

    # Flatten images to vectors
    data = data.view(len(data), -1).numpy()
    targets = targets.numpy()

    return train_test_split(
        data,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )


@dataclass
class SVMConfig:
    """Configuration for RBF SVM training."""

    c: float = 5.0
    gamma: str | float = "scale"
    max_iter: int = -1  # -1 means no limit in scikit-learn


def build_rbf_svm(config: SVMConfig | None = None) -> Pipeline:
    """Create a standard-scaled RBF SVM pipeline.

    Scaling the flattened pixel vectors improves SVM convergence.
    """

    if config is None:
        config = SVMConfig()

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=config.c,
                    gamma=config.gamma,
                    max_iter=config.max_iter,
                ),
            ),
        ]
    )


def train_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SVMConfig | None = None,
) -> Pipeline:
    """Fit an RBF SVM on the provided features and labels."""

    model = build_rbf_svm(config)
    model.fit(X_train, y_train)
    return model


def evaluate_accuracy(model: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy for the given model and dataset."""

    predictions = model.predict(X)
    return float(accuracy_score(y, predictions))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_data(max_samples=5000)
    svm = train_rbf_svm(X_train, y_train)
    acc = evaluate_accuracy(svm, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
