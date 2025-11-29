
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
from torchvision.datasets.utils import download_url
import graph_print_analysis as gp_tool

# add repo root so swiss_roll_models can be imported from anywhere
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "swiss_roll_models").exists():
        sys.path.insert(0, str(p))
        break

from environment.hilbert_distance import hilbert_analysis as hda

class MNISTNet(nn.Module):
    def __init__(self, number_of_layerss=1, hidden_dim=256):
        super().__init__()
        if number_of_layerss < 1:
            raise ValueError("number_of_layerss must be at least 1.")

        layers = []
        input_dim = 28 * 28
        for _ in range(number_of_layerss):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(input_dim, 10)  # We track the weights of this layer

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return logits

class HBModel_MNIST:
    # ============================
    # 1. Define a simple MNIST model
    # ============================



    def train_mnist_with_hilbert(
        number_of_layerss: int = 1,
        num_epochs: Optional[int] = None,      
        max_steps: Optional[int] = None,       
        batch_size: int = 128,
        lr: float = 1e-2,
        device: Optional[str] = None,
        if_regularize: bool = True,
        if_decay: bool = False,
        loss_type: str = 'ce',
        huber_beta: float = 1.0,
        regularization_coeff: float = 1e-4,
        if_regularize_all: bool = False,
        trajectory_save_path: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Train MNIST classifier while tracking output-layer trajectories and optional regularization.
        Args:
            number_of_layerss (int): Number of hidden linear/ReLU blocks to include (>=1).
            num_epochs (Optional[int]): Number of full epochs to run; mutually exclusive with max_steps.
            max_steps (Optional[int]): Fixed number of training steps when set; mutually exclusive with num_epochs.
            batch_size (int): Mini-batch size for training.
            lr (float): Learning rate for the SGD optimizer.
            device (Optional[str]): Optional device override (defaults to CUDA when available).
            initial_vector (Optional[Sequence[float]]): Optional 1D vector to initialize the output layer weights (flattened order).
            if_regularize (bool): Whether to apply weight decay regularization.
            if_decay (bool): Legacy flag preserved for compatibility; when True applies decay to all parameters.
            loss_type (str): Loss choice: ce, huber, or mse.
            huber_beta (float): Beta parameter for SmoothL1 loss when loss_type=huber.
            regularization_coeff (float): Weight decay coefficient when regularization is enabled.
            if_regularize_all (bool): When True, apply regularization to all parameters; otherwise only the output layer.
            trajectory_save_path (Optional[str]): Absolute path to save the parameter trajectory; when None, do not save.
            seed (Optional[int]): Random seed for reproducibility; when None, leave the current RNG state unchanged.

        Returns:
            Dict[str, Any]: "model"
            "output_log",
            "batch_size",
            "lr",
            "epochs_or_steps"
        """
        # ------------ 基本参数检查 ------------
        if (num_epochs is None) and (max_steps is None):
            raise ValueError("必须在 num_epochs 和 max_steps 里至少指定一个。")
        if (num_epochs is not None) and (max_steps is not None):
            raise ValueError("只能二选一：要么用 num_epochs，要么用 max_steps。")
        if number_of_layerss < 1:
            raise ValueError("number_of_layerss must be at least 1.")

        output_log = ""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        output_log += f"Using device: {device}\n"

        # ---- cuDNN 加速（对卷积网络一般有帮助）----
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        # ---- Random seed (for reproducibility) ----
        if seed is not None:
            torch.manual_seed(seed)

        # ---- MNIST data ----
        transform = transforms.Compose([
            transforms.ToTensor(),                  # [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=7,      # 多进程加载
            pin_memory=True,    # 加速 CPU→GPU 拷贝
        )

        model = MNISTNet(number_of_layerss=number_of_layerss).to(device)

        # 根据 loss_type 选择不同的 criterion
        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "huber":
            # Huber/SmoothL1，对 logits 和 one-hot target 做
            criterion = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


        if if_regularize:
            if if_regularize_all or if_decay:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=regularization_coeff)
            else:
                optimizer = torch.optim.SGD(
                    [
                        {"params": model.hidden_layers.parameters(), "weight_decay": 0},
                        {"params": model.output_layer.parameters(), "weight_decay": regularization_coeff},
                    ],
                    lr=lr
                )
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # ---- 用于存储参数轨迹（最后一层 output_layer.weight 的真实向量）----
        # 不做任何 abs / eps / mask / threshold 处理，全部留到外部分析函数统一处理
        param_traj = []

        # 先记录初始权重 w_0
        with torch.no_grad():
            w0 = model.output_layer.weight.detach().cpu().reshape(-1).clone()
            param_traj.append(w0)

        # =======================
        # Training loop
        # =======================
        global_step = 0
        epoch_idx = 0

        # 训练条件：
        # - epoch 模式：epoch_idx < num_epochs
        # - step  模式：global_step < max_steps
        def should_continue():
            cond_epoch = (num_epochs is None) or (epoch_idx < num_epochs)
            cond_steps = (max_steps is None) or (global_step < max_steps)
            return cond_epoch and cond_steps

        while should_continue():
            model.train()
            # running_loss = 0.0
            # correct = 0
            # total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                # 如果是固定 step 模式，先检查是否已经够了
                if (max_steps is not None) and (global_step >= max_steps):
                    break

                # non_blocking=True 在 pin_memory=True 时可以略微提升吞吐
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(images)

                if loss_type == "ce":
                    # 标准分类交叉熵
                    loss = criterion(logits, labels)

                else:
                    # 先把 label 变成 one-hot
                    # logits.shape: [B, 10]
                    # labels.shape: [B]
                    target_onehot = F.one_hot(labels, num_classes=10).float()

                    if loss_type == "huber":
                        # 对 logits 和 one-hot target 做 Huber (SmoothL1)
                        # 这里 target_onehot 已经是 0/1，逻辑上相当于让正确类的 logit 逼近 1，其余逼近 0
                        loss = criterion(logits, target_onehot)

                    elif loss_type == "mse":
                        # 纯 MSE 版本
                        loss = criterion(logits, target_onehot)
                loss.backward()
                optimizer.step()

                # # Track training accuracy (optional)
                # running_loss += loss.item() * images.size(0)
                # _, predicted = logits.max(1)
                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)

                # ====== 每次参数更新之后，记录 output_layer.weight 的真实向量 ======
                with torch.no_grad():
                    w_t = model.output_layer.weight.detach().cpu().reshape(-1).clone()
                    param_traj.append(w_t)

                global_step += 1

                # 再次检查 step 是否超限（防止多跑）
                if (max_steps is not None) and (global_step >= max_steps):
                    break

            # epoch 级别的 log 你暂时注释掉了，就保持不动
            epoch_idx += 1

            if not should_continue():
                break

        steps_run = len(param_traj) - 1  # 去掉初始 w0
        output_log += (
            f"Training finished. Recorded steps (updates) = {steps_run}, "
            f"trajectory length (including init) = {len(param_traj)}\n"
        )

        # =======================
        # 这里不再做任何 Hilbert 分析，只是简单留下最终权重
        # =======================
        # w_star = param_traj[-1]

        if trajectory_save_path is not None:
            traj_path = Path(trajectory_save_path)
            if not traj_path.is_absolute():
                raise ValueError("trajectory_save_path must be an absolute path.")
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(param_traj, traj_path)
            output_log += f"Trajectory saved to {traj_path}\n"

        # 统一交给外部 analysis(...) 去做 Hilbert / mask / threshold 等等
        return {
            "model": model,
            "param_traj": param_traj,
            "output_log": output_log,
            "batch_size": batch_size,
            "lr": lr,
            "epochs_or_steps": f"steps{max_steps}" if max_steps is not None else f"ep{num_epochs}",
        }


    def train_full_batch_with_hilbert(
        self,
        model: Optional[nn.Module] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        lr: float = 1e-2,
        device: Optional[str] = None,
        initial_vector: Optional[Sequence[float]] = None,
        if_regularize: bool = True,
        if_decay: bool = False,
        loss_type: str = "ce",
        huber_beta: float = 1.0,
        regularization_coeff: float = 1e-4,
        if_regularize_all: bool = False,
        seed: Optional[int] = 42,
        batch_size: Optional[int] = None,
        number_of_layerss: int = 1,
    ) -> Dict[str, Any]:
        """
        Run full-batch training for a fixed number of steps on a provided model.

        Args:
            model (Optional[nn.Module]): Model to train. When ``None``, a ``MNISTNet``
                with ``number_of_layerss`` hidden blocks is created automatically.
            images (Optional[torch.Tensor]): Input images as a single batch. When
                ``None``, a batch is drawn from MNIST using ``batch_size``.
            labels (Optional[torch.Tensor]): Corresponding labels for ``images``;
                must be provided together with ``images`` when not auto-loading.
            num_steps (int): Number of optimization steps to run.
            lr (float): Learning rate for the SGD optimizer.
            device (Optional[str]): Target device. Defaults to CUDA when available.
            initial_vector (Optional[Sequence[float]]): Optional flattened vector to initialize
                ``output_layer.weight``.
            if_regularize (bool): Whether to apply weight decay regularization.
            if_decay (bool): Legacy flag to regularize all parameters when True.
            loss_type (str): "ce", "huber", or "mse".
            huber_beta (float): Beta for SmoothL1 when ``loss_type="huber"``.
            regularization_coeff (float): Weight decay coefficient.
            if_regularize_all (bool): When True, regularize all parameters rather than only the output layer.
            seed (Optional[int]): Random seed for reproducibility.
            batch_size (Optional[int]): Batch size used when auto-loading MNIST data.
                Required when ``images`` or ``labels`` is ``None``.
            number_of_layerss (int): Number of hidden linear/ReLU blocks when constructing
                a default ``MNISTNet``.

        Returns:
            Dict[str, Any]: Dictionary with keys "model", "param_traj", "lr", and "steps".
        """
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        if (images is None) != (labels is None):
            raise ValueError("images and labels must both be provided or both be None.")

        if images is None:
            if batch_size is None:
                raise ValueError("batch_size is required when images/labels are not provided.")

            if number_of_layerss < 1:
                raise ValueError("number_of_layerss must be at least 1.")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            train_dataset = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transform,
            )

            data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=7,
                pin_memory=True,
            )

            try:
                images, labels = next(iter(data_loader))
            except StopIteration:
                raise RuntimeError("Failed to draw a batch from the MNIST dataset.")

        if model is None:
            model = MNISTNet(number_of_layerss=number_of_layerss)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if seed is not None:
            torch.manual_seed(seed)

        model = model.to(device)

        if initial_vector is not None:
            init_vec = torch.tensor(initial_vector, dtype=torch.float32, device=device)
            expected_numel = model.output_layer.weight.numel()
            if init_vec.numel() != expected_numel:
                raise ValueError(
                    f"Initial vector size {init_vec.numel()} does not match output_layer weight size {expected_numel}."
                )
            with torch.no_grad():
                model.output_layer.weight.copy_(init_vec.view_as(model.output_layer.weight))

        images = images.to(device)
        labels = labels.to(device)

        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "huber":
            criterion = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if if_regularize:
            if if_regularize_all or if_decay:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=regularization_coeff)
            else:
                optimizer = torch.optim.SGD(
                    [
                        {"params": model.hidden_layers.parameters(), "weight_decay": 0},
                        {"params": model.output_layer.parameters(), "weight_decay": regularization_coeff},
                    ],
                    lr=lr,
                )
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        param_traj: List[torch.Tensor] = []
        with torch.no_grad():
            w0 = model.output_layer.weight.detach().cpu().reshape(-1).clone()
            param_traj.append(w0)

        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = model(images)

            if loss_type == "ce":
                loss = criterion(logits, labels)
            else:
                target_onehot = F.one_hot(labels, num_classes=10).float()
                loss = criterion(logits, target_onehot)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                w_t = model.output_layer.weight.detach().cpu().reshape(-1).clone()
                param_traj.append(w_t)

        return {"model": model, "param_traj": param_traj, "lr": lr, "steps": len(param_traj) - 1}


