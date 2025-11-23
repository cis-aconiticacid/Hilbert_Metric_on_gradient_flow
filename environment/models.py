import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .hilbert_distance import hilbert_distance


def _get_activation(name: str):
    """
    Map a string to a PyTorch activation module.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unknown activation {name}")


class MLP(nn.Module):
    """
    A simple MLP for swiss-roll regression/classification tasks.
    The head is a single unit for regression, otherwise sized for classification.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims=(128, 64),
        out_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        task: str = "regression",
    ):
        super().__init__()
        layers = []
        last = in_dim
        act = _get_activation(activation)
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.backbone = nn.Sequential(*layers)

        if task == "regression":
            self.head = nn.Linear(last, out_dim)
        elif task == "classification":
            self.head = nn.Linear(last, out_dim)
        else:
            raise ValueError(f"Unknown task {task}")
        self.task = task

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head(h)
        if self.task == "regression":
            return logits.squeeze(-1)
        return logits


class PosLinear(nn.Module):
    """
    Linear layer constrained to the positive cone via softplus reparameterization.
    All effective weights/biases are positive, which keeps parameters in a cone.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_raw = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_raw", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_raw, a=math.sqrt(5))
        if self.bias_raw is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_raw)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_raw, -bound, bound)

    def forward(self, x):
        weight = F.softplus(self.weight_raw)
        bias = F.softplus(self.bias_raw) if self.bias_raw is not None else None
        return F.linear(x, weight, bias)


class ConeMLP(nn.Module):
    """
    Positive-cone MLP where all weights stay positive; designed for
    Hilbert metric experiments and cone-constrained dynamics.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims=(64, 32),
        out_dim: int = 1,
        activation: str = "softplus",
        task: str = "regression",
    ):
        super().__init__()
        layers = []
        last = in_dim
        act = _get_activation(activation)
        for h in hidden_dims:
            layers.append(PosLinear(last, h))
            layers.append(act)
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = PosLinear(last, out_dim)
        self.task = task

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head(h)
        if self.task == "regression":
            return logits.squeeze(-1)
        return logits

    def flatten_positive_params(self):
        """
        Return a single positive vector from all params, for Hilbert metric usage.
        """
        flats = []
        with torch.no_grad():
            for name, param in self.named_parameters():
                flats.append(F.softplus(param).reshape(-1))
        return torch.cat(flats)


def projective_distance(model_a: ConeMLP, model_b: ConeMLP, eps: float = 1e-8):
    """
    Compute Hilbert projective distance between two ConeMLP parameter vectors.
    Call only when both models share the same architecture.
    """
    va = model_a.flatten_positive_params()
    vb = model_b.flatten_positive_params()
    return hilbert_distance(va, vb, eps=eps)


def build_model(task: str, in_dim: int, use_cone: bool = False, **kwargs):
    """
    Convenience factory for the baseline MLP or the cone-constrained version.
    Args:
        task: "regression" or "classification".
        in_dim: input dimensionality of each sample.
        use_cone: True -> ConeMLP (positive weights), False -> standard MLP.
        **kwargs: forwarded to the chosen model constructor (e.g., hidden_dims).
    """
    out_dim_default = 1 if task == "regression" else 2
    out_dim = kwargs.pop("out_dim", out_dim_default)
    if use_cone:
        return ConeMLP(in_dim=in_dim, out_dim=out_dim, task=task, **kwargs)
    return MLP(in_dim=in_dim, out_dim=out_dim, task=task, **kwargs)
