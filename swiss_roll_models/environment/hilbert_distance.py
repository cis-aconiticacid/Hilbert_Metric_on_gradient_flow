"""
This module provides functions to compute the Hilbert distance between positive vectors
and analyze parameter trajectories in the context of optimization within positive cones.

Author: Xinyang Wen(sometimes goes by Elizabeth Wen)

History:
    1.This is part of swiss_roll_models project. in 22/11/2025

"""
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
    def analysis_distance_on_cone(param_traj, w_star, threshold=None, ifmask=False,if_threshold=False,if_self_adaptive=False):
        """
        Analyze Hilbert distance on parameter trajectory (optionally restricted to the small positive cone defined by w_star)

        Parameters：
            param_traj: list[Tensor],Ttrajectory parameters (one at each step)
            w_star:     Tensor，The reference weight (usually final weight or closed-form solution)
            threshold:  float，support threshold for small positive cone
            ifmask:     bool，whether to mask according to w_star support
            if_threshold: bool, whether to use thresholding to replace masking
            if_self_adaptive: bool, whether to use self-adaptive contraction(Too lazy to implement now, QWQ)

        Returns：
            dict containing：
                - hilbert_to_final: list[float]
                - hilbert_to_init:  list[float]
                - hilbert_between:  list[float]
                - traj_masked:      Tensor or None
                - w_star_masked:    Tensor or None
                - mask:             BoolTensor or None
        """

        if if_self_adaptive:
            """
            Self-adaptive contraction is used by traceing last time vector,
            if the GD has elimiated some dimensions, then we contract the cone accordingly.
            and recount the last step hilbert distance.(Just a vision, not implemented yet, QWQ)
            """
            raise ProcessLookupError("Elizabeth is too lazy to implement self-adaptive contraction now QWQ, maybe in next paper?")
        
        ref_traj = None
        if ifmask:
            if if_threshold:
                raise ValueError("cannot set both ifmask and if_threshold to True.")
            if threshold is None:
                raise ValueError("ifmask=True but threshold is None.")
            masked_traj, w_star_masked, mask = hilbert_analysis.mask_by_wstar_support(
            param_traj, w_star, threshold)
            ref_traj = masked_traj  # shape (T, d_small)
            w_init = ref_traj[0]          # First point: init projection on the small positive cone
            w_star_new = w_star_masked    # Target point: w_star projection on the small positive cone
        else:
            ref_traj = torch.stack(
            [p.detach().clone().view(-1) for p in param_traj],
            dim=0)  # shape (T, D)
            if if_threshold:
                if threshold is None:
                    raise ValueError("if_threshold=True but threshold is None.")
                # Use a relative small epsilon based on w_star minimum positive entry
                with torch.no_grad():
                    w_flat = w_star.detach().clone().view(-1)
                    positive = w_flat[w_flat > 0]
                    if positive.numel() > 0:
                        eps = positive.min().item() * 1e-3
                    else:
                        eps = 1e-10
                ref_traj = torch.clamp(ref_traj, min=eps)
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
    The following are the truth:
        3： If someone interested in this, please contact me QWQ at elizabethwen2005@gmail.com or xwen57@wisc.edu before my graduatioon(not decided yet XD)
        4:  谁来帮我实现负锥距离啊，我太菜了啊啊啊啊啊
        5:  Chatgpt真好用哈哈哈哈哈哈
            Gpt说应该加上"（感谢你陪我写奇怪的动力系统 XD）这句话"
        6:  好像希尔伯特也这么干过挖坑不管的事情（毕竟这个就是他挖的坑(Legacy)）
        7： 以上，写于22/11/2025, 傍晚时分，
            也算是给未来的自己留个纪念：
            ——如果这个方向真的成了，那我会笑着再读这段；
            ——如果没成……那我就是个笑话罢了 XD
    """