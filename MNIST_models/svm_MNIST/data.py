"""MNIST data loading helpers tailored to the SVM experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


@dataclass
class MNISTBinaryConfig:
    """Configuration controlling the MNIST binary subset."""

    positive_digit: int = 3
    negative_digit: int = 8
    max_samples: int | None = 20000
    test_size: float = 0.2
    random_state: int = 42


def load_binary_mnist(config: MNISTBinaryConfig | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a fixed train/test split for a two-digit MNIST task.

    The function filters the canonical MNIST training set down to two digits,
    assigns labels ``+1`` for the ``positive_digit`` and ``-1`` for the
    ``negative_digit``, and returns flattened feature matrices.

    Args:
        config: Optional configuration. When omitted, defaults to the standard
            ``3 vs 8`` task with a 20% evaluation split.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` where feature matrices have shape
        ``(n_samples, 784)`` with pixel intensities normalized to ``[0, 1]``.
    """

    if config is None:
        config = MNISTBinaryConfig()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    images = dataset.data.float() / 255.0
    targets = dataset.targets

    mask = (targets == config.positive_digit) | (targets == config.negative_digit)
    images = images[mask]
    targets = targets[mask]

    if config.max_samples is not None and config.max_samples < len(images):
        generator = torch.Generator().manual_seed(config.random_state)
        indices = torch.randperm(len(images), generator=generator)[: config.max_samples]
        images = images[indices]
        targets = targets[indices]

    flat_images = images.view(len(images), -1).numpy()
    labels = targets.numpy()

    y = np.where(labels == config.positive_digit, 1.0, -1.0)

    return train_test_split(
        flat_images,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
