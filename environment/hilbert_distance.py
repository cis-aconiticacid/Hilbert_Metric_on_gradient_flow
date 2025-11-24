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

        Return a Python float
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        x = x.view(-1)
        y = y.view(-1)

        if (x < eps).any() or (y < eps).any():
            raise ValueError("Inputs to hilbert_distance must be strictly positive.")

        ratio = x / y
        max_r = ratio.max()
        min_r = ratio.min()
        return (max_r.log() - min_r.log()).item()

    @staticmethod
    def mask_by_wstar_support(param_traj, w_star, threshold):
        """
        Use w_star to support small cone，和你原来完全一致。
        参数和返回值接口保持不变。
        """
        w_ref_raw = w_star.detach().clone().view(-1)

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

        w_star_masked = w_ref_raw[mask]

        masked_list = []
        for idx, vect in enumerate(param_traj):
            v_raw = vect.detach().clone().view(-1)
            if v_raw.numel() != w_ref_raw.numel():
                raise ValueError(
                    f"Shape mismatch at step {idx}: "
                    f"param_traj[{idx}].numel()={v_raw.numel()} vs w_star.numel()={w_ref_raw.numel()}"
                )
            masked_list.append(v_raw[mask])

        masked_traj = torch.stack(masked_list, dim=0)
        return masked_traj, w_star_masked, mask

    @staticmethod
    def analysis_distance_on_cone(param_traj, w_star,
                                  threshold=None,
                                  ifmask=False,
                                  if_threshold=False,
                                  if_self_adaptive=False):
        """
        界面和原来一模一样，但内部计算全面向量化，加速很多。
        返回 dict:
            - hilbert_to_final: list[float]
            - hilbert_to_init:  list[float]
            - hilbert_between:  list[float]
            - traj_masked:      Tensor or None
            - w_star_masked:    Tensor or None
            - mask:             BoolTensor or None
        """

        if if_self_adaptive:
            raise ProcessLookupError(
                "Elizabeth is too lazy to implement self-adaptive contraction now QWQ, maybe in next paper?"
            )

        # --------- 1. 构造 ref_traj、w_init、w_star_new ----------
        if ifmask:
            if if_threshold:
                raise ValueError("cannot set both ifmask and if_threshold to True.")
            if threshold is None:
                raise ValueError("ifmask=True but threshold is None.")

            masked_traj, w_star_masked, mask = hilbert_analysis.mask_by_wstar_support(
                param_traj, w_star, threshold
            )
            ref_traj = masked_traj          # (T, d_small)
            w_init = ref_traj[0]            # 初始点
            w_star_new = w_star_masked      # 目标点
        else:
            ref_traj = torch.stack(
                [p.detach().clone().view(-1) for p in param_traj],
                dim=0
            )  # (T, D)
            if if_threshold:
                if threshold is None:
                    raise ValueError("if_threshold=True but threshold is None.")
                with torch.no_grad():
                    w_flat = w_star.detach().clone().view(-1)
                    positive = w_flat[w_flat > 0]
                    if positive.numel() > 0:
                        eps = positive.min().item() * 1e-3
                    else:
                        eps = 1e-10
                ref_traj = torch.clamp(ref_traj, min=eps)

            w_init = ref_traj[0]
            w_star_new = ref_traj[-1]
            masked_traj = None
            w_star_masked = None
            mask = None

        # --------- 2. 统一 dtype + eps clamp，避免 0 ---------
        eps = 1e-10
        ref_traj = ref_traj.to(dtype=torch.float64)

        w_init = w_init.to(dtype=torch.float64)
        w_star_new = w_star_new.to(dtype=torch.float64)

        ref_traj = torch.clamp(ref_traj, min=eps)
        w_init = torch.clamp(w_init, min=eps)
        w_star_new = torch.clamp(w_star_new, min=eps)

        T, D = ref_traj.shape

        # --------- 3. 向量化计算 d_H(w_t, w*) ----------
        # ratio_final: (T, D)
        ratio_final = ref_traj / w_star_new  # 广播 (T,D) / (D,)
        max_r_f, _ = ratio_final.max(dim=1)  # (T,)
        min_r_f, _ = ratio_final.min(dim=1)
        hilbert_to_final = (max_r_f.log() - min_r_f.log()).tolist()

        # --------- 4. 向量化计算 d_H(w_t, w_0) ----------
        ratio_init = ref_traj / w_init      # (T, D)
        max_r_i, _ = ratio_init.max(dim=1)
        min_r_i, _ = ratio_init.min(dim=1)
        hilbert_to_init = (max_r_i.log() - min_r_i.log()).tolist()

        # --------- 5. 向量化计算 d_H(w_t, w_{t-1}) ----------
        if T >= 2:
            ratio_between = ref_traj[1:] / ref_traj[:-1]   # (T-1, D)
            max_r_b, _ = ratio_between.max(dim=1)
            min_r_b, _ = ratio_between.min(dim=1)
            hilbert_between = (max_r_b.log() - min_r_b.log()).tolist()
        else:
            hilbert_between = []

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