import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

"""
This module provides functions to generate swiss roll datasets
with either regression or classification targets in a D-dimensional space.
"""

# Randomly generate swiss roll samples over D-dimensional space
# with regression or classification targets.
def _swiss_roll_geometry(n_samples, D=32, noise=0.0, device="cpu"):
    """
    Generates the geometric structure of a swiss roll in D-dimensional space.
    Args:
        n_samples (int): Number of samples to generate.
        D (int): Dimensionality of the ambient space.
        noise (float): Standard deviation of Gaussian noise added to the data.
        device (str): Device to use for tensor computations.
    Returns:
        X_high (torch.Tensor): Generated data of shape (n_samples, D).
        u (torch.Tensor): Parameter u used in swiss roll generation.
        v (torch.Tensor): Parameter v used in swiss roll generation.
    """
    u = torch.empty(n_samples, device=device).uniform_(3 * math.pi, 9 * math.pi) # the 
    v = torch.empty(n_samples, device=device).uniform_(0.0, 20.0)

    x1 = u * torch.cos(u)
    x2 = v
    x3 = u * torch.sin(u)

    X3 = torch.stack([x1, x2, x3], dim=1)  # (n, 3)

    if noise > 0:
        X3 = X3 + noise * torch.randn_like(X3)

    A = torch.randn(D, 3, device=device)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (D, 3)
    X_high = X3 @ Q.T  # (n, D)

    return X_high, u, v

# Generate swiss roll dataset with regression or classification targets
def generate_swiss_roll(n_samples, task="regression", D=32, noise=0.0, device="cpu"):
    """
    Generates a D-dimensional swiss roll dataset with either regression or classification targets.
    Args:
        n_samples (int): Number of samples to generate.
        task (str): Type of task, either "regression" or "classification".
        D (int): Dimensionality of the ambient space.
        noise (float): Standard deviation of Gaussian noise added to the data.
        device (str): Device to use for tensor computations.
    Returns:
        X_high (torch.Tensor): Generated data of shape (n_samples, D).
        y (torch.Tensor): Target values of shape (n_samples,).
        u (torch.Tensor): Parameter u used in swiss roll generation.
        v (torch.Tensor): Parameter v used in swiss roll generation.
    """
    X_high, u, v = _swiss_roll_geometry(n_samples, D=D, noise=noise, device=device)

    if task == "regression":
        y = torch.sin(u) / (u + 1.0)
    elif task == "classification":
        threshold = 6 * math.pi
        y = (u > threshold).long()
    else:
        raise ValueError(f"Unknown task type: {task}")

    return X_high, y, u, v
