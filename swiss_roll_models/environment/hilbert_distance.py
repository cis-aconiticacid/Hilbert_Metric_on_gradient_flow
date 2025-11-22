import math

def hilbert_distance(x, y, eps=1e-8):
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