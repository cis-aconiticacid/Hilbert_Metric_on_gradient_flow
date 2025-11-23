from .hilbert_distance import hilbert_distance, mask_by_wstar_support


def analyze_param_trajectory(
    param_traj,
    w_star=None,
    threshold: float = 1e-10,
    mask_by_star: bool = False,
):
    """
    Compute Hilbert-distance statistics along a parameter trajectory.

    Args:
        param_traj: list of 1D tensors capturing the optimization path.
        w_star: optional reference vector; defaults to the last element of param_traj.
        threshold: minimum value allowed when comparing ratios.
        mask_by_star: if True, restrict computation to the support of w_star.

    Returns:
        dict with hilbert_to_final, hilbert_to_init, hilbert_between,
        traj_masked (if masking applied), mask, and w_star_used.
    """
    if not param_traj:
        raise ValueError("param_traj must be a non-empty list of tensors.")

    ref_traj = [p.detach().clone().view(-1) for p in param_traj]
    w_star_vec = w_star.detach().clone().view(-1) if w_star is not None else ref_traj[-1]

    masked_traj = None
    mask = None
    if mask_by_star:
        masked_traj, w_star_vec, mask = mask_by_wstar_support(ref_traj, w_star_vec, threshold)
        ref_traj = masked_traj

    w_init = ref_traj[0]

    hilbert_to_final = [hilbert_distance(v, w_star_vec, eps=threshold) for v in ref_traj]
    hilbert_to_init = [hilbert_distance(v, w_init, eps=threshold) for v in ref_traj]

    # keep length aligned with trajectory (first entry is distance to itself)
    hilbert_between = [0.0]
    for prev, cur in zip(ref_traj, ref_traj[1:]):
        hilbert_between.append(hilbert_distance(cur, prev, eps=threshold))

    return {
        "hilbert_to_final": hilbert_to_final,
        "hilbert_to_init": hilbert_to_init,
        "hilbert_between": hilbert_between,
        "traj_masked": masked_traj,
        "mask": mask,
        "w_star_used": w_star_vec,
    }
