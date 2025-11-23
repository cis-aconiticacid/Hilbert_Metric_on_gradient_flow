import math
import torch
class hilbert_analysis:
    @staticmethod
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
    
    @staticmethod
    def mask_by_wstar_support(param_traj, w_star, threshold):
        """
        Use w_star to define a supportive cone”：
            mask_i = (|w_star_i| > threshold)

        Parameters：
            param_traj: list[Tensor]，every param is a vecotrt, use view(-1) to flatten
            w_star:     Tensor，] the reference weight (usually final weight or closed-form solution)
            threshold:  Support threshold for small positive cone

        Returns：
            masked_traj:  Tensor, shape (T, d_small)
            w_star_masked: Tensor, shape (d_small,)
            mask:          BoolTensor, shape (D,)
        """
        # Flatten to a 1D vector
        w_ref_raw = w_star.detach().clone().view(-1)

        # Use w_star to determine the support of "non-vanishing dimensions"
        mask = (w_ref_raw.abs() > threshold)

        if mask.sum().item() == 0:
            print("Error happening, The info of: w_star min, max, norm:",
                  w_star.min().item(),
                  w_star.max().item(),
                  w_star.norm().item())
            print("traj last param norm:", param_traj[-1].detach().clone().view(-1).norm().item())
            print("provided w_star norm:", w_star.norm().item())
            raise ValueError(
                "Dimension of the small positive cone is zero. "
                "Crash happening during masking by w_star support."
            )

        # Restrict w_star to the small positive cone
        w_star_masked = w_ref_raw[mask]

        # Apply the same mask to each step in param_traj
        masked_list = []
        for idx, vect in enumerate(param_traj):
            v_raw = vect.detach().clone().view(-1)
            if v_raw.numel() != w_ref_raw.numel():
                raise ValueError(
                    f"Shape mismatch at step {idx}: "
                    f"param_traj[{idx}].numel()={v_raw.numel()} vs w_star.numel()={w_ref_raw.numel()}"
                )
            masked_list.append(v_raw[mask])

        # Stack into a (T, d_small) tensor for unified processing later
        masked_traj = torch.stack(masked_list, dim=0)

        return masked_traj, w_star_masked, mask

    @staticmethod
    def analysis_distance_on_cone(param_traj, w_star, threshold=None, ifmask=False):
        """
        Analyze Hilbert distance on parameter trajectory (optionally restricted to the small positive cone defined by w_star)

        Parameters：
            param_traj: list[Tensor],Ttrajectory parameters (one at each step)
            w_star:     Tensor，The reference weight (usually final weight or closed-form solution)
            threshold:  float，support threshold for small positive cone
            ifmask:     bool，whether to mask according to w_star support

        Returns：
            dict containing：
                - hilbert_to_final: list[float]
                - hilbert_to_init:  list[float]
                - hilbert_between:  list[float]
                - traj_masked:      Tensor or None
                - w_star_masked:    Tensor or None
                - mask:             BoolTensor or None
        """
        if ifmask:
            if threshold is None:
                raise ValueError("ifmask=True but threshold is None.")

            masked_traj, w_star_masked, mask = hilbert_analysis.mask_by_wstar_support(
                param_traj, w_star, threshold
            )
            # ref_traj is a (T, d_small) tensor
            ref_traj = masked_traj
            w_init = ref_traj[0]          # First point: init projection on the small positive cone
            w_star_new = w_star_masked    # Target point: w_star projection on the small positive cone
        else:
            # No mask: directly flatten all parameters
            ref_traj = torch.stack(
                [p.detach().clone().view(-1) for p in param_traj],
                dim=0
            )  # shape (T, D)
            w_init = ref_traj[0]
            w_star_new = ref_traj[-1]     # Last point: reference weight
            masked_traj = None
            w_star_masked = None
            mask = None

        hilbert_to_final = []
        hilbert_to_init = []
        hilbert_between = []

        prev_v = None
        for idx, v in enumerate(ref_traj):
            # v is a 1D vector (d_small,) or (D,)
            hilbert_to_final.append(hilbert_analysis.hilbert_distance(v, w_star_new))
            hilbert_to_init.append(hilbert_analysis.hilbert_distance(v, w_init))

            if idx > 0:
                hilbert_between.append(hilbert_analysis.hilbert_distance(v, prev_v))
            prev_v = v

        return {
            "hilbert_to_final": hilbert_to_final,
            "hilbert_to_init": hilbert_to_init,
            "hilbert_between": hilbert_between,
            "traj_masked": masked_traj,
            "w_star_masked": w_star_masked,
            "mask": mask,
        }







    # @staticmethod
    # def Thompson_distance(x,y):
    #     if (x<0).any() or (y<0).any():
    #         raise ValueError("Inputs to Thompson_distance must be non-negative.")
    #     return raiseValue("Thompson distance not implemented yet.")

    # @staticmethod
    # def Furstenberg_Khasminskii_distance(x,y):
    #     return raiseValue("Furstenberg–Khasminskii distance not implemented yet.")

    """

    For negative circumtance, we are not able to address it now.
    The following is my guess:
        1:  I guess Von Neumann metric could be related, it converges to a cone
        2:  try to use Jordan decomopsition to embed it into R^2 (This might involves complex numbers??? only god knows 只有天知道了)
    """