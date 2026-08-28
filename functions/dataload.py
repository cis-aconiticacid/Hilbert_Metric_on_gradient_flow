import torch
import os

datapath = "./MNIST_models/model_result/"

def loadresults(batchsize, lr, epoch_or_step, ifstep, ModelClass, filename=None, device="cuda"):
    """Load a saved model checkpoint for an MNIST experiment.

    Args:
        batchsize (int): Training batch size used to name the checkpoint.
        lr (float): Learning rate used to name the checkpoint.
        epoch_or_step (int): Epoch index or global step for the checkpoint.
        ifstep (bool): Interpret epoch_or_step as a step count when true.
        ModelClass (type): Callable returning a checkpoint-compatible model.
        filename (str | None): Explicit checkpoint name, if supplied.
        device (str): Device used for model construction and checkpoint load.

    Returns:
        torch.nn.Module: An initialized model with its state dictionary loaded.
    """
    step_or_epoch = f"steps{epoch_or_step}" if ifstep else f"ep{epoch_or_step}"

    if filename is None:
        filename = f"models_bs{batchsize}_lr{lr}_{step_or_epoch}.pt"

    fullpath = os.path.join(datapath, filename)

    # Initialize model
    model = ModelClass().to(device)

    # Load state_dict
    state_dict = torch.load(fullpath, map_location=device)
    model.load_state_dict(state_dict)

    return model


def build_param_traj(batchsize, lr, step_list, ModelClass, device="cuda"):
    """Build a flattened weight trajectory from saved checkpoints.

    Args:
        batchsize (int): Training batch size used in checkpoint names.
        lr (float): Learning rate used in checkpoint names.
        step_list (Sequence[int]): Ordered checkpoint step indices.
        ModelClass (type): Callable returning a checkpoint-compatible model.
        device (str): Device used while loading each checkpoint.

    Returns:
        list[torch.Tensor]: Flattened fc2 weight tensors in step order.
    """
    param_traj = []

    for step in step_list:
        model = loadresults(
            batchsize=batchsize,
            lr=lr,
            epoch_or_step=step,
            ifstep=True,           # Because we name by step here
            ModelClass=ModelClass,
            device=device
        )
        with torch.no_grad():
            w_t = model.fc2.weight.detach().cpu().reshape(-1).clone()
            param_traj.append(w_t)

    return param_traj
