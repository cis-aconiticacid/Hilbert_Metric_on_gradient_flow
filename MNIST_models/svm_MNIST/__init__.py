"""Experiment utilities for MNIST SVM Hilbert-metric analysis."""

from .data import load_binary_mnist
from .linear_svm import LinearSVMConfig, train_linear_svm
from .projections import make_random_projection, apply_projection
from .geometry import trajectory_geometry, endpoint_svd, hilbert_distance_matrix
from .kernel_svm import KernelSVMConfig, train_kernel_svm_runs

__all__ = [
    "load_binary_mnist",
    "LinearSVMConfig",
    "train_linear_svm",
    "make_random_projection",
    "apply_projection",
    "trajectory_geometry",
    "endpoint_svd",
    "hilbert_distance_matrix",
    "KernelSVMConfig",
    "train_kernel_svm_runs",
]
