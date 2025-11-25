import torch
import os

datapath = "./MNIST_models/model_result/"

def loadresults(batchsize, lr, epoch_or_step, ifstep, ModelClass, filename=None, device="cuda"):
    """
    Return a load_state_dict completed model.
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
    """
    Build parameter trajectory from saved checkpoints at specified steps.
    Returns a list of parameter tensors (flattened) at each specified step.
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
