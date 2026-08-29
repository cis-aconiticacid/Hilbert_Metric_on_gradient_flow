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
        max_samples (int | None): Maximum number of samples to keep from the
            60k MNIST training set. ``None`` disables subsampling, otherwise
            the dataset is deterministically shuffled with ``random_state``
            before truncation.
        test_size (float): Fraction of the retained samples to allocate to the
            test split passed to :func:`sklearn.model_selection.train_test_split`.
        random_state (int): Seed used both for subsampling and for the
            train/test split to keep runs reproducible.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``X_train``,
        ``X_test``, ``y_train`` and ``y_test`` arrays. The feature arrays are
        two-dimensional with shape ``(n_samples, 784)`` where each row contains
        normalized pixel intensities in ``[0, 1]``.

    Notes:
        Images are flattened from ``28×28`` tensors to one-dimensional vectors
        so they can be consumed by scikit-learn's SVM implementation.
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
    """Configuration container for the RBF SVM.

    Attributes:
        c (float): Soft-margin penalty ``C`` passed to :class:`sklearn.svm.SVC`.
        gamma (str | float): Kernel coefficient; string values forward to
            scikit-learn's presets (e.g., ``"scale"`` or ``"auto"``).
        max_iter (int): Maximum number of iterations for the solver; ``-1``
            delegates the stopping condition to scikit-learn defaults.
    """

    c: float = 5.0
    gamma: str | float = "scale"
    max_iter: int = -1  # -1 means no limit in scikit-learn


def build_rbf_svm(config: SVMConfig | None = None) -> Pipeline:
    """Create a standard-scaled RBF SVM pipeline.

    Args:
        config (SVMConfig | None): Hyperparameter settings for the SVM model.
            When omitted, :class:`SVMConfig` defaults are used.

    Returns:
        Pipeline: A scikit-learn :class:`~sklearn.pipeline.Pipeline`
        consisting of a ``StandardScaler`` followed by an RBF-kernel
        :class:`~sklearn.svm.SVC` configured with ``config``.

    Notes:
        Feature standardization typically improves SVM convergence and avoids
        the penalty term being dominated by input scale differences.
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
    """Fit an RBF SVM on the provided features and labels.

    Args:
        X_train (np.ndarray): Feature matrix with shape ``(n_samples, 784)``
            containing flattened, normalized MNIST images.
        y_train (np.ndarray): Target labels matching ``X_train`` rows.
        config (SVMConfig | None): Optional hyperparameter bundle passed to
            :func:`build_rbf_svm`; defaults provide a reasonable baseline.

    Returns:
        Pipeline: A trained scikit-learn pipeline ready for inference.
    """

    model = build_rbf_svm(config)
    model.fit(X_train, y_train)
    return model


def evaluate_accuracy(model: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy for the given model and dataset.

    Args:
        model (Pipeline): A fitted pipeline produced by
            :func:`train_rbf_svm` or :func:`build_rbf_svm`.
        X (np.ndarray): Feature matrix to evaluate, typically from
            :func:`load_mnist_data`.
        y (np.ndarray): Ground-truth labels corresponding to ``X`` rows.

    Returns:
        float: Proportion of correctly classified samples in ``[0.0, 1.0]``.
    """

    predictions = model.predict(X)
    return float(accuracy_score(y, predictions))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_data(max_samples=5000)
    svm = train_rbf_svm(X_train, y_train)
    acc = evaluate_accuracy(svm, X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
