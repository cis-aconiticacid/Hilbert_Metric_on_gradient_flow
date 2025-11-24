import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os

for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "swiss_roll_models").exists():
        sys.path.insert(0, str(p))
        break

from environment.hilbert_distance import hilbert_analysis as hda

# ============================
# 工具函数
# ============================

def moving_average_xy(y, window=50):
    """
    对 (step, y) 做滑动平均，返回 (x_smooth, y_smooth)，
    x_smooth 是 step 的滑动平均，保证 x 轴和真实 step 对齐。
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


def downsample_xy(x, y, max_points=1000):
    """
    将 (x, y) 等距下采样到最多 max_points 个点。
    用于避免画图太密集，同时不依赖固定的 stride。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x[idx], y[idx]


def describe_segment(vals):
    """
    对一段 ratio 列表做统计，返回 mean/min/max/Q25/Q75 的 dict。
    NaN 会被过滤；如果全是 NaN，返回 None。
    """
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def compute_hilbert_metrics(param_traj,
                            threshold=1e-3,
                            if_mask=False,
                            if_threshold=True,
                            if_self_adaptive=False,
                            w_star=None):
    """
    调用 hda.analysis_distance_on_cone，返回：
      - hilbert_to_final, hilbert_between, hilbert_to_init
      - ratios_to_final, ratios_between
    """
    if w_star is None:
        w_star = param_traj[-1]

    res = hda.analysis_distance_on_cone(
        param_traj=param_traj,
        w_star=w_star,
        threshold=threshold,
        ifmask=if_mask,
        if_threshold=if_threshold,
        if_self_adaptive=if_self_adaptive,
    )
    hilbert_to_final = res["hilbert_to_final"]
    hilbert_between = res["hilbert_between"]
    hilbert_to_init = res["hilbert_to_init"]

    ratios_to_final = []
    for t in range(len(hilbert_to_final) - 1):
        if hilbert_to_final[t] > 0:
            ratios_to_final.append(hilbert_to_final[t + 1] / hilbert_to_final[t])
        else:
            ratios_to_final.append(float("nan"))

    ratios_between = []
    for t in range(len(hilbert_between) - 1):
        if hilbert_between[t] > 0:
            ratios_between.append(hilbert_between[t + 1] / hilbert_between[t])
        else:
            ratios_between.append(float("nan"))

    return {
        "hilbert_to_final": hilbert_to_final,
        "hilbert_between": hilbert_between,
        "hilbert_to_init": hilbert_to_init,
        "ratios_to_final": ratios_to_final,
        "ratios_between": ratios_between,
    }


# ============================
# 画图函数（只负责画图）
# ============================

def plot_hilbert_distance_smoothed(hilbert_to_final,
                                   batch_size, lr, num_epochs,
                                   result_dir,
                                   suffix="unmasked"):
    """
    TODO #1：用 moving_average_xy 做平滑，
    画 d_H(w_t, w*) 的 log-scale 曲线。
    """
    window_h = max(20, len(hilbert_to_final) // 100)  # 根据长度自适应
    x_h, hilbert_smooth = moving_average_xy(hilbert_to_final, window=window_h)

    plt.figure()
    if len(hilbert_smooth) > 0:
        plt.semilogy(x_h, hilbert_smooth, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("d_H(w_t, w*) (log scale, smoothed)")
    plt.title(
        f"Hilbert distance to w* (smoothed, {suffix})\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    output_path = os.path.join(
        result_dir,
        f"HTF_smooth_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path)
    plt.close()


def plot_ratio_global_smoothed(ratios,
                               batch_size, lr, num_epochs,
                               result_dir,
                               suffix="unmasked"):
    """
    TODO #1/#4：ratio 全程平滑图。
    - 不删除 NaN，而是先替换为 1.0
    - 用 moving_average_xy 对齐 step
    """
    ratio_array = np.array(ratios, dtype=float)
    bad = ~np.isfinite(ratio_array)
    ratio_array[bad] = 1.0

    window_r = max(50, len(ratio_array) // 50)
    x_r, ratio_smooth = moving_average_xy(ratio_array, window=window_r)

    # 下采样防止点太多（masked/unmasked 共用逻辑）
    x_r_ds, ratio_ds = downsample_xy(x_r, ratio_smooth, max_points=1000)

    plt.figure()
    if len(ratio_ds) > 0:
        plt.plot(x_r_ds, ratio_ds, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("smoothed ratio d_H(t+1)/d_H(t)")
    plt.axhline(1.0, linestyle="--")
    plt.title(
        f"Hilbert contraction ratio (smoothed, {suffix})\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    output_path_ratio = os.path.join(
        result_dir,
        f"HR_smooth_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path_ratio)
    plt.close()


def plot_ratio_zoom_last(ratios,
                         batch_size, lr, num_epochs,
                         result_dir,
                         suffix="unmasked",
                         zoom_len=300,
                         window=10):
    """
    TODO #3：只看最后 zoom_len 步的 ratio。
    - 用很小窗口（或者 window=1 基本不平滑）看末端乱跳。
    """
    ratio_array = np.array(ratios, dtype=float)
    bad = ~np.isfinite(ratio_array)
    ratio_array[bad] = 1.0

    n = len(ratio_array)
    if n == 0:
        return

    start_idx = max(0, n - zoom_len)
    steps_zoom = np.arange(start_idx, n)
    ratio_zoom = ratio_array[start_idx:]

    # 在 zoom 里用较小的 window
    window = min(window, len(ratio_zoom)) if window > 1 else 1
    x_zoom, ratio_zoom_smooth = moving_average_xy(ratio_zoom, window=window)

    # x_zoom 是相对 index，所以要平移到真实 step
    x_zoom_real = x_zoom + start_idx

    plt.figure()
    if len(ratio_zoom_smooth) > 0:
        plt.plot(x_zoom_real, ratio_zoom_smooth)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("step t")
    plt.ylabel(f"ratio (zoom, last {zoom_len} steps)")
    plt.title(
        f"Hilbert ratio zoomed (last {zoom_len} steps, {suffix})\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    zoom_path = os.path.join(
        result_dir,
        f"HR_zoom_last{zoom_len}_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(zoom_path)
    plt.close()


def plot_masked_hilbert_and_ratio(hilbert_to_final2,
                                  ratios_to_final2,
                                  batch_size, lr, num_epochs,
                                  result_dir):
    """
    TODO #1/#3/#4：Masked 的 Hilbert + ratio 图。
    - Hilbert：log-scale 原始曲线 + 平滑版（可选）
    - ratio：全程平滑 + last 300/100 步 zoom in
    """
    # Hilbert distance (masked, 原始 log-scale)
    steps = np.arange(len(hilbert_to_final2))
    plt.figure()
    if len(hilbert_to_final2) > 0:
        plt.semilogy(steps, hilbert_to_final2)
    plt.xlabel("step t")
    plt.ylabel("d_H(w_t, w*) (log scale)")
    title = (
        "Hilbert distance to w* during training (Masked Cone)\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.title(title)
    plt.tight_layout()
    output_path2 = os.path.join(
        result_dir,
        f"HTF_masked_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path2)
    plt.close()

    # ratio 全程平滑 + zoom in
    plot_ratio_global_smoothed(
        ratios_to_final2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked",
    )
    plot_ratio_zoom_last(
        ratios_to_final2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked",
        zoom_len=300,
        window=10,
    )
    plot_ratio_zoom_last(
        ratios_to_final2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked_last100",
        zoom_len=100,
        window=5,
    )


# ============================
# 文本写入（统计）函数
# ============================

def write_unmasked_stats(f, ratios_to_final):
    """
    写入无 mask 的 ratio 统计：
    - 前 200
    - 200–400
    - 最后 200
    加上 mean/min/max/Q25/Q75（TODO #5）
    """
    f.write("===== No Masking Analysis Results =====\n")

    front_200 = ratios_to_final[:200]
    medium_200_400 = ratios_to_final[200:400]
    tail_200 = ratios_to_final[-200:]

    # 前 200 步
    f.write("The first 200 Steps ratio_to_final:\n")
    desc_front = describe_segment(front_200)
    if desc_front is not None:
        f.write(f"  Mean ≈ {desc_front['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_front['min']:.4f}, Max ≈ {desc_front['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_front['q25']:.4f}, Q75 ≈ {desc_front['q75']:.4f}\n")

    # 200–400 步
    f.write("The 200 to 400 Steps ratio_to_final:\n")
    desc_mid = describe_segment(medium_200_400)
    if desc_mid is not None:
        f.write(f"  Mean ≈ {desc_mid['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_mid['min']:.4f}, Max ≈ {desc_mid['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_mid['q25']:.4f}, Q75 ≈ {desc_mid['q75']:.4f}\n")

    # 最后 200 步
    f.write("\nThe last 200 Steps ratio_to_final:\n")
    desc_tail = describe_segment(tail_200)
    if desc_tail is not None:
        f.write(f"  Mean ≈ {desc_tail['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_tail['min']:.4f}, Max ≈ {desc_tail['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_tail['q25']:.4f}, Q75 ≈ {desc_tail['q75']:.4f}\n")


def write_masked_stats(f, ratios_to_final2):
    """
    写入 Masked Cone 的 ratio 统计：
    - 前 200
    - 200–400
    - 最后 200
    同样用 mean/min/max/Q25/Q75。
    """
    f.write("===== Masked Cone Analysis Results =====\n")

    front_200_2 = ratios_to_final2[:200]
    medium_200_400_2 = ratios_to_final2[200:400]
    tail_200_2 = ratios_to_final2[-200:]

    f.write("\nFirst 200 Steps ratio_to_final:\n")
    desc_front2 = describe_segment(front_200_2)
    if desc_front2 is not None:
        f.write(f"  Mean ≈ {desc_front2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_front2['min']:.4f}, Max ≈ {desc_front2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_front2['q25']:.4f}, Q75 ≈ {desc_front2['q75']:.4f}\n")

    f.write("200 to 400 Steps ratio_to_final:\n")
    desc_mid2 = describe_segment(medium_200_400_2)
    if desc_mid2 is not None:
        f.write(f"  Mean ≈ {desc_mid2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_mid2['min']:.4f}, Max ≈ {desc_mid2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_mid2['q25']:.4f}, Q75 ≈ {desc_mid2['q75']:.4f}\n")

    f.write("\nLast 200 Steps ratio_to_final:\n")
    desc_tail2 = describe_segment(tail_200_2)
    if desc_tail2 is not None:
        f.write(f"  Mean ≈ {desc_tail2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_tail2['min']:.4f}, Max ≈ {desc_tail2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_tail2['q25']:.4f}, Q75 ≈ {desc_tail2['q75']:.4f}\n")


# ============================
# 顶层分析函数：调用上面的东西
# ============================

def analysis(param_traj, output_log, batch_size, lr, num_epochs=None,
             threshold=1e-3, if_mask=True, steps=None):
    """
    顶层总入口：
      1. 创建 result_dir 和 result_path
      2. 计算 unmasked Hilbert / ratio
      3. 画 unmasked 图（全程 + zoom）
      4. 写 unmasked 文本统计
      5. 如果 if_mask=True：
         - 计算 masked Hilbert / ratio
         - 画 masked 图（全程 + zoom）
         - 写 masked 文本统计
      6. 返回相关结果（方便之后做更复杂分析）

    对于“错误”情况不做 try/except 捕获，直接让错误抛出：
      - num_epochs 和 steps 同时 None 会 raise ValueError（和原来逻辑一致）
    """

    if steps is not None:
        num_epochs = steps
    elif num_epochs is None:
        raise ValueError("Either num_epochs or steps must be provided.")

    # 0. 创建目录和结果文件
    result_dir = './model_result/'+f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}"
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(
        result_dir,
        f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}.txt"
    )

    # 这里保留“已经做过就跳过”的逻辑，这不是错误，只是逻辑控制
    if os.path.exists(result_path):
        print(f"Analysis file {result_path} already exists. Skipping analysis.")
        return

    if sys.path and "swiss_roll_models" in sys.path[0]:
        sys.path.pop(0)

    # 1. 无 mask 的 Hilbert / ratio
    metrics_unmasked = compute_hilbert_metrics(
        param_traj=param_traj,
        threshold=threshold,
        if_mask=False,
        if_threshold=True,
        if_self_adaptive=False,
        w_star=None,
    )
    hilbert_to_final = metrics_unmasked["hilbert_to_final"]
    ratios_to_final = metrics_unmasked["ratios_to_final"]
    ratios_between = metrics_unmasked["ratios_between"]

    # 2. 无 mask 图：Hilbert 平滑 + ratio 全程平滑 + ratio 末端 zoom
    plot_hilbert_distance_smoothed(
        hilbert_to_final,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked",
    )
    plot_ratio_global_smoothed(
        ratios_to_final,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked",
    )
    plot_ratio_zoom_last(
        ratios_to_final,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked",
        zoom_len=300,
        window=10,
    )
    plot_ratio_zoom_last(
        ratios_to_final,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked_last100",
        zoom_len=100,
        window=5,
    )

    # 3. 写文本：模型信息 + training log + 无 mask 统计
    with open(result_path, "a", encoding="utf-8") as f:
        f.write("===== Models Information =====\n\n")
        f.write(f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}\n\n")
        f.write("training log\n")
        f.write(output_log)
        f.write("\n\n")

        write_unmasked_stats(f, ratios_to_final)

        # 4. 如果不需要 masked，直接结束
        if not if_mask:
            # 返回无 mask 的结果
            return {
                "hilbert_to_final": hilbert_to_final,
                "ratios_to_final": ratios_to_final,
                "ratios_between": ratios_between,
            }

        # 5. Masked 分析
        para_traj2 = param_traj
        w_star_raw = para_traj2[-1].clone()

        metrics_masked = compute_hilbert_metrics(
            param_traj=para_traj2,
            threshold=1e-5,
            if_mask=True,
            if_threshold=False,
            if_self_adaptive=False,
            w_star=w_star_raw,
        )
        hilbert_to_final2 = metrics_masked["hilbert_to_final"]
        ratios_to_final2 = metrics_masked["ratios_to_final"]
        ratios_between2 = metrics_masked["ratios_between"]

        # 文本统计（masked）
        write_masked_stats(f, ratios_to_final2)

    # 6. Masked 图（Hilbert + ratio 全程 + ratio zoom）
    plot_masked_hilbert_and_ratio(
        hilbert_to_final2,
        ratios_to_final2,
        batch_size, lr, num_epochs,
        result_dir,
    )

    return {
        "hilbert_to_final": hilbert_to_final,
        "ratios_to_final": ratios_to_final,
        "ratios_between": ratios_between,
        "hilbert_to_final_masked": hilbert_to_final2,
        "ratios_to_final_masked": ratios_to_final2,
        "ratios_between_masked": ratios_between2,
    }
