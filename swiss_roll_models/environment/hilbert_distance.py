import math
import torch

def hilbert_distance(x, y, eps=1e-10):
    """
    x, y: (m,) positive (>0)
    d_H(x, y) = log( max_i x_i / y_i ) - log( min_i x_i / y_i )

    Returns: float
    """
    # Avoid zeros
    # If it's zero, it collapse techically to infinity
    # However, we have two options to make it up
    # 1) mask zeros after training (use thresholding)
    # 2) use 
    if(x<eps).any() or (y<eps).any():
        raise ValueError("Inputs to hilbert_distance must be strictly positive.")
    ratio = x / y
    max_r = ratio.max()
    min_r = ratio.min()
    
    return (max_r.log() - min_r.log()).item()

def mask_by_wstar_support(param_traj, w_star, threshold):
    """
    Use w_star to define a small positive cone by its support,
        mask_i = (|w_star_i| > threshold)

    Returns:
        masked_traj: list of 1D tensors, shows the trajectory restricted to the small cone
        w_star_masked: w_star restricted to the small cone
        mask: 1D bool tensor, records the retained coordinates (i.e., the support of the small positive cone)
    """
    w_ref_raw = w_star.detach().clone().view(-1)

    # Use w_star to select the "non-vanishing dimensions" that form the small positive cone
    mask = (w_ref_raw.abs() > threshold)

    if mask.sum() == 0:
        raise ValueError(
            "Dimension of the small positive cone is zero."
            "Crash happening during masking by w_star support."
        )

    w_star_masked = w_ref_raw[mask]

    masked_traj = []
    for idx, vect in enumerate(param_traj):
        v_raw = vect.detach().clone().view(-1)
        if v_raw.numel() != w_ref_raw.numel():
            raise ValueError(
                f"Shape mismatch at step {idx}: "
                f"param_traj[{idx}].numel()={v_raw.numel()} vs w_star.numel()={w_ref_raw.numel()}"
            )
        masked_traj.append(v_raw[mask])

    return masked_traj, w_star_masked, mask


def var_analysis_distance_on_cone(param_traj, w_star, threshold,ifmask=False):
    """
    Analyze the Hilbert distance along a parameter trajectory restricted to the small positive cone
    """
    ref_traj= []
    masked_traj= []
    # if make is True, restrict to small positive cone defined by w_star and threshold
    if ifmask:
        masked_traj, mask = mask_by_wstar_support(param_traj, w_star, threshold)
        ref_traj = masked_traj
    else:
        ref_traj = [p.detach().clone().view(-1) for p in param_traj]

    # Initialize
    w_init = ref_traj[0]
    w_star_new = ref_traj[-1]
    hilbert_to_final = []
    hilbert_to_init = []
    hilbert_between = []
    prev_v = None

    for idx, v in enumerate(ref_traj):
        # always compute distances in the masked space if masking is applied
        hilbert_to_final.append(hilbert_distance(v, w_star_new))
        hilbert_to_init.append(hilbert_distance(v, w_init))

        if idx > 0:
            hilbert_between.append(hilbert_distance(v, prev_v))
        prev_v = v

    return {
        "hilbert_to_final": hilbert_to_final,
        "hilbert_to_init": hilbert_to_init,
        "hilbert_between": hilbert_between,
        "traj_masked": masked_traj,
    }

def Thompson_distance(x,y):
    if (x<0).any() or (y<0).any():
        raise ValueError("Inputs to Thompson_distance must be non-negative.")
    return raiseValue("Thompson distance not implemented yet.")

def Furstenberg_Khasminskii_distance(x,y):
    return raiseValue("Furstenberg–Khasminskii distance not implemented yet.")

"""

For negative circumtance, we are not able to address it now.
The following is my guess:
    1:  I guess Von Neumann metric could be related, it converges to a cone
    2:  try to use Jordan decomopsition to embed it into R^2 (This might involves complex numbers??? only god knows 只有天知道了)
"""