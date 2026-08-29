
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
from torchvision.datasets.utils import download_url
_old_sys_path = sys.path.copy()
try:
    project_root = Path.cwd().parent.parent
    sys.path.insert(0, str(Path.cwd().resolve().parents[1]))
    print(F"Project root added to sys.path: {project_root}")
    from functions import graph_print_analysis as gp_tool

finally:
    sys.path = _old_sys_path

def train_mnist_with_hilbert(
    # ... 你原来的参数 ...
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    # 新增
    if_overwrite: bool = True,

    if_record_test: bool = False,
    check_distance: int = 500,
) -> Dict[str, Any]:

    # ... 你原来的训练代码不动 ...

    def safe_torch_save(obj, path: Path, if_overwrite: bool):
        """
        Save obj to path. If file exists and not overwriting, raise.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (not if_overwrite):
            raise FileExistsError(f"File already exists: {path} (set if_overwrite=True to overwrite)")
        torch.save(obj, path)

    # =======================
    # Save
    # =======================
    if save_path is not None:
        sp = Path(save_path)
        if not sp.is_absolute():
            raise ValueError("save_path must be an absolute path.")
        sp.mkdir(parents=True, exist_ok=True)

        # （可选）把 save_name 真的用起来：作为文件前缀
        prefix = save_name if (save_name is not None and len(save_name) > 0) else ""

        def with_prefix(filename: str) -> str:
            return f"{prefix}_{filename}" if prefix else filename

        traj_path = sp / with_prefix("param_trajectory.pt")
        loss_path = sp / with_prefix("loss_trajectory.pt")

        safe_torch_save(param_traj, traj_path, if_overwrite)
        safe_torch_save(loss_traj, loss_path, if_overwrite)

        output_log += f"Trajectory saved to {traj_path}\n"
        output_log += f"Loss saved to {loss_path}\n"

        if if_record_test:
            acc_path = sp / with_prefix(f"test_acc_every_{check_distance}_steps.pt")
            safe_torch_save(test_acc_record, acc_path, if_overwrite)
            output_log += f"Test accuracy record saved to {acc_path}\n"
    return


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
        number_of_layers: int = 1,
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
        save_path: Optional[str] = None,
        save_name: Optional[str] = None,
        seed: Optional[int] = 42,

        # ===== 新增参数 =====
        if_record_test: bool = False,
        check_distance: int = 500,

        # ===== 新增：是否允许覆盖保存文件 =====
        if_overwrite: bool = True,

    ) -> Dict[str, Any]:
        """
        Train MNIST classifier while tracking output-layer trajectories and optional regularization.

        New:
            if_record_test (bool): whether to periodically evaluate test accuracy.
            check_distance (int): checkpoint interval (record param/loss and optionally eval test acc).
            if_overwrite (bool): whether to overwrite existing saved files.

        Returns:
            Dict[str, Any]:
                "model"
                "param_traj"         (List[tensor], checkpointed)
                "traj_steps"         (List[int], same length as param_traj)
                "loss_traj"          (List[float], checkpointed)
                "loss_steps"         (List[int], same length as loss_traj)
                "test_acc_record"    (dict or None)
                "output_log"
                "batch_size"
                "lr"
                "epochs_or_steps"
                "steps_run"
        """
        # ------------ 基本参数检查 ------------
        if (num_epochs is None) and (max_steps is None):
            raise ValueError("必须在 num_epochs 和 max_steps 里至少指定一个。")
        if (num_epochs is not None) and (max_steps is not None):
            raise ValueError("只能二选一：要么用 num_epochs，要么用 max_steps。")
        if number_of_layers < 1:
            raise ValueError("number_of_layerss must be at least 1.")
        if (check_distance is None) or (not isinstance(check_distance, int)) or (check_distance <= 0):
            raise ValueError("check_distance 必须是正整数（想每一步都记录就设为 1）。")

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
            transforms.ToTensor(),
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
            num_workers=7,
            pin_memory=True,
        )

        # ===== Test loader (optional) =====
        test_loader = None
        if if_record_test:
            test_dataset = datasets.MNIST(
                root="./data",
                train=False,
                download=True,
                transform=transform,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=7,
                pin_memory=True,
            )

        model = MNISTNet(number_of_layerss=number_of_layers).to(device)

        # 根据 loss_type 选择不同的 criterion
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
                    lr=lr
                )
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # ====== checkpointed trajectories ======
        param_traj: List[torch.Tensor] = []
        traj_steps: List[int] = []

        loss_traj: List[float] = []
        loss_steps: List[int] = []

        # ===== 新增：test accuracy 记录结构 =====
        test_acc_record = None
        if if_record_test:
            test_acc_record = {
                "check_distance": check_distance,
                "acc_traj": [],       # list of {"step": int, "acc": float}
                "final_test_acc": None
            }

        def eval_test_accuracy() -> float:
            """Compute accuracy on test set."""
            if test_loader is None:
                raise RuntimeError("test_loader is None but eval_test_accuracy() was called.")
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    logits = model(images)
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            return correct / max(total, 1)

        # ===== step=0 记录初始权重 w0 =====
        with torch.no_grad():
            w0 = model.output_layer.weight.detach().cpu().reshape(-1).clone()
        param_traj.append(w0)
        traj_steps.append(0)

        # =======================
        # Training loop
        # =======================
        global_step = 0
        epoch_idx = 0
        last_loss_value: Optional[float] = None

        def should_continue():
            cond_epoch = (num_epochs is None) or (epoch_idx < num_epochs)
            cond_steps = (max_steps is None) or (global_step < max_steps)
            return cond_epoch and cond_steps

        while should_continue():
            model.train()

            for batch_idx, (images, labels) in enumerate(train_loader):
                if (max_steps is not None) and (global_step >= max_steps):
                    break

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(images)

                if loss_type == "ce":
                    loss = criterion(logits, labels)
                else:
                    target_onehot = F.one_hot(labels, num_classes=10).float()
                    if loss_type == "huber":
                        loss = criterion(logits, target_onehot)
                    elif loss_type == "mse":
                        loss = criterion(logits, target_onehot)

                last_loss_value = float(loss.item())
                loss.backward()
                optimizer.step()

                global_step += 1

                # ===== 统一 checkpoint：param / loss / (optional) test acc =====
                if (global_step % check_distance == 0):
                    # record param
                    with torch.no_grad():
                        w_t = model.output_layer.weight.detach().cpu().reshape(-1).clone()
                    param_traj.append(w_t)
                    traj_steps.append(global_step)

                    # record loss (this step's batch loss)
                    loss_traj.append(last_loss_value)
                    loss_steps.append(global_step)

                    # record test acc
                    if if_record_test:
                        acc = eval_test_accuracy()
                        test_acc_record["acc_traj"].append({"step": global_step, "acc": acc})
                        output_log += f"[Test Acc] step={global_step}: {acc*100:.2f}%\n"

                if (max_steps is not None) and (global_step >= max_steps):
                    break

            epoch_idx += 1
            if not should_continue():
                break

        steps_run = global_step
        output_log += (
            f"Training finished. updates(steps) = {steps_run}, "
            f"checkpoint interval = {check_distance}, "
            f"param checkpoints = {len(param_traj)} (including init)\n"
        )

        # ===== 训练结束：保证最终 step 被记录（param/loss/test acc）=====
        if (len(traj_steps) == 0) or (traj_steps[-1] != steps_run):
            with torch.no_grad():
                w_final = model.output_layer.weight.detach().cpu().reshape(-1).clone()
            param_traj.append(w_final)
            traj_steps.append(steps_run)

            if steps_run > 0 and (last_loss_value is not None):
                loss_traj.append(last_loss_value)
                loss_steps.append(steps_run)

        if if_record_test:
            final_acc = eval_test_accuracy()
            test_acc_record["final_test_acc"] = final_acc
            if (len(test_acc_record["acc_traj"]) == 0) or (test_acc_record["acc_traj"][-1]["step"] != steps_run):
                test_acc_record["acc_traj"].append({"step": steps_run, "acc": final_acc})
            output_log += f"[Test Acc] final (step={steps_run}): {final_acc*100:.2f}%\n"

        # =======================
        # Save (with overwrite control)
        # =======================
        def safe_torch_save(obj, path: Path, if_overwrite_: bool):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and (not if_overwrite_):
                raise FileExistsError(f"File already exists: {path} (set if_overwrite=True to overwrite)")
            torch.save(obj, path)

        if save_path is not None:
            sp = Path(save_path)
            if not sp.is_absolute():
                raise ValueError("save_path must be an absolute path.")
            sp.mkdir(parents=True, exist_ok=True)

            prefix = save_name if (save_name is not None and len(save_name) > 0) else ""

            def with_prefix(filename: str) -> str:
                return f"{prefix}_{filename}" if prefix else filename

            # 用 check_distance 进文件名，避免多实验覆盖还看不出
            traj_path = sp / with_prefix(f"param_trajectory_every_{check_distance}_steps.pt")
            loss_path = sp / with_prefix(f"loss_trajectory_every_{check_distance}_steps.pt")
            steps_path = sp / with_prefix(f"record_steps_every_{check_distance}_steps.pt")

            safe_torch_save(param_traj, traj_path, if_overwrite)
            safe_torch_save(loss_traj, loss_path, if_overwrite)
            safe_torch_save({"traj_steps": traj_steps, "loss_steps": loss_steps}, steps_path, if_overwrite)

            output_log += f"Param trajectory saved to {traj_path}\n"
            output_log += f"Loss trajectory saved to {loss_path}\n"
            output_log += f"Record steps saved to {steps_path}\n"

            if if_record_test:
                acc_path = sp / with_prefix(f"test_acc_every_{check_distance}_steps.pt")
                safe_torch_save(test_acc_record, acc_path, if_overwrite)
                output_log += f"Test accuracy record saved to {acc_path}\n"

        return {
            "model": model,
            "param_traj": param_traj,
            "traj_steps": traj_steps,
            "loss_traj": loss_traj,
            "loss_steps": loss_steps,
            "test_acc_record": test_acc_record,
            "output_log": output_log,
            "batch_size": batch_size,
            "lr": lr,
            "epochs_or_steps": f"steps{max_steps}" if max_steps is not None else f"ep{num_epochs}",
            "steps_run": steps_run,
            "check_distance": check_distance,
        }

    @staticmethod
    def train_full_batch_with_hilbert(
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int,
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
    ) -> Dict[str, Any]:
        """
        Run full-batch training for a fixed number of steps on a provided model.

        Args:
            model (nn.Module): Model to train. Must expose an ``output_layer`` attribute.
            images (torch.Tensor): Input images as a single batch.
            labels (torch.Tensor): Corresponding labels for ``images``.
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

        Returns:
            Dict[str, Any]: Dictionary with keys "model", "param_traj", "lr", and "steps".
        """
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

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

    @staticmethod
    def train_with_optional_initialization(
        num_steps: int,
        if_initialize: bool,
        number_of_layerss: int = 1,
        lr: float = 1e-2,
        device: Optional[str] = None,
        model: Optional[nn.Module] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        initial_vector: Optional[Sequence[float]] = None,
        if_regularize: bool = True,
        if_decay: bool = False,
        loss_type: str = "ce",
        huber_beta: float = 1.0,
        regularization_coeff: float = 1e-4,
        if_regularize_all: bool = False,
        seed: Optional[int] = 42,
        sample_size: int = 128,
    ) -> Dict[str, Any]:
        """
        Train for a fixed number of steps with optional model/data initialization.

        When ``if_initialize`` is True, the method builds a fresh ``MNISTNet`` and
        samples ``sample_size`` examples from the MNIST training set. When False, it
        expects a pre-created ``model`` along with ``images`` and ``labels``
        representing a full batch. The routine then runs ``num_steps`` of
        full-batch training and returns the resulting trajectory.

        Returns a dictionary containing "model", "param_traj", "lr", and "step".
        """
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if if_initialize:
            if number_of_layerss < 1:
                raise ValueError("number_of_layerss must be at least 1.")
            if sample_size < 1:
                raise ValueError("sample_size must be at least 1 when initializing.")
            if seed is not None:
                torch.manual_seed(seed)

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

            sample_loader = DataLoader(
                train_dataset,
                batch_size=sample_size,
                shuffle=True,
                num_workers=7,
                pin_memory=True,
            )

            images_batch, labels_batch = next(iter(sample_loader))
            model_to_train = MNISTNet(number_of_layerss=number_of_layerss)
            images_to_use = images_batch
            labels_to_use = labels_batch
            init_vector = None
        else:
            if model is None or images is None or labels is None:
                raise ValueError(
                    "When if_initialize is False, provide model, images, and labels for full-batch training."
                )
            model_to_train = model
            images_to_use = images
            labels_to_use = labels
            init_vector = initial_vector

        result = HBModel_MNIST.train_full_batch_with_hilbert(
            model=model_to_train,
            images=images_to_use,
            labels=labels_to_use,
            num_steps=num_steps,
            lr=lr,
            device=device,
            initial_vector=init_vector,
            if_regularize=if_regularize,
            if_decay=if_decay,
            loss_type=loss_type,
            huber_beta=huber_beta,
            regularization_coeff=regularization_coeff,
            if_regularize_all=if_regularize_all,
            seed=seed,
        )

        return {
            "model": result["model"],
            "param_traj": result["param_traj"],
            "lr": result["lr"],
            "step": result.get("steps", len(result.get("param_traj", [])) - 1),
        }
