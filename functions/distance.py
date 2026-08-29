"""
This module provides functions to compute the Hilbert distance between positive vectors
and analyze parameter trajectories in the context of optimization within positive cones.

Author: Xinyang Wen(sometimes goes by Elizabeth Wen)

History:
    1.This is part of swiss_roll_models project. in 22/11/2025

"""
import torch
class distance_func:
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