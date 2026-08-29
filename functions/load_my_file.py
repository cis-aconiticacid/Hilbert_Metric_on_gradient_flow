import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# from MNIST_models.graph_print_analysis import moving_average_xy

# ====== 配置 ======
BASE_DIR = Path(__file__).resolve().parents[1] / "MNIST_models" / "Nureon Network MNIST" / "analysis_result"

batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
num_epochs = [1, 2, 3, 4, 5, 6]
lrs = [1e-4, 1e-3, 1e-2, 1e-1]


# ====== 小工具：平滑 ======
def moving_average_xy(y, window=50):
    """
    和你之前一样：对 y 做滑动平均，返回 (x_smooth, y_smooth).
    如果长度不足窗口，就直接原样返回。
    """
    y = np.array(y, dtype=float)
    n = len(y)
    if n == 0:
        return np.array([]), np.array([])

    if window <= 1 or n < window:
        x = np.arange(n, dtype=float)
        return x, y

    kernel = np.ones(window) / window
    x = np.arange(n, dtype=float)
    y_smooth = np.convolve(y, kernel, mode="valid")
    x_smooth = np.convolve(x, kernel, mode="valid")
    return x_smooth, y_smooth


# ====== 读 json 的函数 ======
def load_ratios_between(json_path: Path):
    """
    读取形如 analysis_resultAnalysis_bs*_lr*_epep*ratios_between.json 的文件，
    尽量兼容几种结构：
      - 直接是 list
      - {"ratios_between": [...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 优先找这些 key
        for key in ["ratios_between", "between", "ratios"]:
            if key in data:
                return np.array(data[key], dtype=float)
        # 如果结构完全不同，先给个报错把 key 打印出来，你可以看一眼改一下
        raise KeyError(
            f"Cannot find ratios_between-like key in {json_path.name}, keys={list(data.keys())}"
        )
    else:
        # 顶层是 list
        return np.array(data, dtype=float)


# ====== 画一张图：单个 (bs, lr, ep) ======
def plot_between_for_config(
    batch_size,
    lr,
    ep=None,
    step=None,
    base_dir: Path = BASE_DIR,
    filename: str | None = None,
    smooth=True,
    window=None,
    save_dir: Path | None = None,
):
    """
    画出某个 (batch_size, lr, ep) 的 ratios_between 曲线。
    """
    # 文件名按照你说的模式拼
    if filename is  None:
        if ep is None and step is not None:
            filename = f"analysis_resultAnalysis_bs{batch_size}_lr{lr}_epsteps{step}ratios_between.json"
        else:
            filename = f"analysis_resultAnalysis_bs{batch_size}_lr{lr}_epep{ep}ratios_between.json"

    json_path = base_dir / filename

    if not json_path.exists():
        print(f"[WARN] File not found: {json_path}")
        return

    ratios = load_ratios_between(json_path)

    if smooth:
        if window is None:
            window = max(20, len(ratios) // 100)  # 和 HTF 差不多的策略
        x, y = moving_average_xy(ratios, window=window)
        label_suffix = f"(smoothed, window={window})"
    else:
        x = np.arange(len(ratios))
        y = ratios
    label_suffix = "(raw)"

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.yscale("log")
    plt.axhline(1.0, linestyle="--")  # ratio=1 参考线

    plt.xlabel("step t")
    plt.ylabel("ratio_between(t) = d_H(w_{t+1}, w_t) / ...")
    plt.title(
        f"Hilbert contraction between steps {label_suffix}\n"
        f"batch_size={batch_size}, lr={lr}, epochs={ep}"
    )
    plt.tight_layout()

    # 保存或直接 show
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        png_name = f"between_bs{batch_size}_lr{lr}_ep{ep}.png"
        out_path = save_dir / png_name
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] Saved: {out_path}")
        plt.close()
    else:
        plt.show()


# # ====== 使用示例 ======
# # 1. 先画一张你想检查的：
# # plot_between_for_config(batch_size=256, lr=1e-4, ep=4,
# #                         smooth=True,
# #                         save_dir=BASE_DIR / "plots_between")

# # 2. 如果你想一口气全跑（小心图太多），可以这样：

# root_path=Path.cwd().parent.parent
# experimental_path=root_path / "experiments"
