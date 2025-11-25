import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import os
import json

# 把 swiss_roll_models 加进来（如果存在）
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "swiss_roll_models").exists():
        sys.path.insert(0, str(p))
        break

from environment.hilbert_distance import hilbert_analysis as hda


# ============================
# Tools functions
# ============================

def moving_average_xy(y, window=50):
    """
    Average (step, y)，Return (x_smooth, y_smooth)
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
    Downsample (x, y) evenly to at most max_points points.
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
    Compute statistics for a segment of ratio values
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
    调 hda.analysis_distance_on_cone，拿到：
      - hilbert_to_final, hilbert_between, hilbert_to_init
      - ratios_between（基于 hilbert_between）
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

    # 我们现在只关心 between 的 ratio
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
        "ratios_between": ratios_between,
    }


# ============================
# Plotting functions (between 专用)
# ============================

def plot_between_global_smoothed(ratios_between,
                                 batch_size, lr, num_epochs,
                                 result_dir,
                                 suffix="unmasked"):
    """
    全局平滑后的 between ratio 图
    """
    plt.xscale('log')
    plt.yscale('log')
    ratio_array = np.array(ratios_between, dtype=float)
    bad = ~np.isfinite(ratio_array)
    ratio_array[bad] = 1.0  # NaN/inf 先替成 1

    window_r = max(50, len(ratio_array) // 50)
    x_r, ratio_smooth = moving_average_xy(ratio_array, window=window_r)

    x_r_ds, ratio_ds = downsample_xy(x_r, ratio_smooth, max_points=1000)

    plt.figure()
    if len(ratio_ds) > 0:
        plt.plot(x_r_ds, ratio_ds, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("smoothed ratio_between(t)")
    plt.axhline(1.0, linestyle="--")
    plt.title(
        f"Hilbert contraction between steps (smoothed, {suffix})\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    out_path = os.path.join(
        result_dir,
        f"HB_smooth_between_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png",
    )
    plt.savefig(out_path)
    plt.close()


def plot_between_zoom_last(ratios_between,
                           batch_size, lr, num_epochs,
                           result_dir,
                           suffix="unmasked",
                           zoom_len=300,
                           window=10):
    """
    只看最后 zoom_len 步的 between ratio，窗口小一点看末端震荡
    """
    ratio_array = np.array(ratios_between, dtype=float)
    bad = ~np.isfinite(ratio_array)
    ratio_array[bad] = 1.0
    plt.xscale('log')
    plt.yscale('log')
    n = len(ratio_array)
    if n == 0:
        return

    start_idx = max(0, n - zoom_len)
    ratio_zoom = ratio_array[start_idx:]

    window = min(window, len(ratio_zoom)) if window > 1 else 1
    x_zoom, ratio_zoom_smooth = moving_average_xy(ratio_zoom, window=window)
    x_zoom_real = x_zoom + start_idx

    plt.figure()
    if len(ratio_zoom_smooth) > 0:
        plt.plot(x_zoom_real, ratio_zoom_smooth, linewidth=1.5)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("step t")
    plt.ylabel(f"ratio_between (last {zoom_len} steps)")
    plt.title(
        f"Hilbert contraction between steps (zoom, {suffix})\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    out_path = os.path.join(
        result_dir,
        f"HB_zoom_last{zoom_len}_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png",
    )
    plt.savefig(out_path)
    plt.close()


def plot_masked_between(ratios_between2,
                        batch_size, lr, num_epochs,
                        result_dir):
    """
    masked cone 的 between 图：全局 + 两个 zoom
    """
    plot_between_global_smoothed(
        ratios_between2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked",
    )
    plot_between_zoom_last(
        ratios_between2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked",
        zoom_len=300,
        window=10,
    )
    plot_between_zoom_last(
        ratios_between2,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="masked_last100",
        zoom_len=100,
        window=5,
    )


# ============================
# Text Writing (Statistics) Functions
# ============================

def write_unmasked_between_stats(f, ratios_between):
    """
    只写 between 的统计：
    - 前 200
    - 200–400
    - 最后 200
    """
    f.write("===== No Masking (Between) Analysis Results =====\n")

    front_200 = ratios_between[:200]
    medium_200_400 = ratios_between[200:400]
    tail_200 = ratios_between[-200:]

    f.write("The first 200 steps ratios_between:\n")
    desc_front = describe_segment(front_200)
    if desc_front is not None:
        f.write(f"  Mean ≈ {desc_front['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_front['min']:.4f}, Max ≈ {desc_front['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_front['q25']:.4f}, Q75 ≈ {desc_front['q75']:.4f}\n")

    f.write("The 200 to 400 steps ratios_between:\n")
    desc_mid = describe_segment(medium_200_400)
    if desc_mid is not None:
        f.write(f"  Mean ≈ {desc_mid['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_mid['min']:.4f}, Max ≈ {desc_mid['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_mid['q25']:.4f}, Q75 ≈ {desc_mid['q75']:.4f}\n")

    f.write("\nThe last 200 steps ratios_between:\n")
    desc_tail = describe_segment(tail_200)
    if desc_tail is not None:
        f.write(f"  Mean ≈ {desc_tail['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_tail['min']:.4f}, Max ≈ {desc_tail['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_tail['q25']:.4f}, Q75 ≈ {desc_tail['q75']:.4f}\n")


def write_masked_between_stats(f, ratios_between2):
    """
    Masked Cone 的 between 统计
    """
    f.write("\n===== Masked Cone (Between) Analysis Results =====\n")

    front_200_2 = ratios_between2[:200]
    medium_200_400_2 = ratios_between2[200:400]
    tail_200_2 = ratios_between2[-200:]

    f.write("\nFirst 200 steps ratios_between:\n")
    desc_front2 = describe_segment(front_200_2)
    if desc_front2 is not None:
        f.write(f"  Mean ≈ {desc_front2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_front2['min']:.4f}, Max ≈ {desc_front2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_front2['q25']:.4f}, Q75 ≈ {desc_front2['q75']:.4f}\n")

    f.write("200 to 400 steps ratios_between:\n")
    desc_mid2 = describe_segment(medium_200_400_2)
    if desc_mid2 is not None:
        f.write(f"  Mean ≈ {desc_mid2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_mid2['min']:.4f}, Max ≈ {desc_mid2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_mid2['q25']:.4f}, Q75 ≈ {desc_mid2['q75']:.4f}\n")

    f.write("\nLast 200 steps ratios_between:\n")
    desc_tail2 = describe_segment(tail_200_2)
    if desc_tail2 is not None:
        f.write(f"  Mean ≈ {desc_tail2['mean']:.4f}\n")
        f.write(f"  Min ≈ {desc_tail2['min']:.4f}, Max ≈ {desc_tail2['max']:.4f}\n")
        f.write(f"  Q25 ≈ {desc_tail2['q25']:.4f}, Q75 ≈ {desc_tail2['q75']:.4f}\n")


# ============================
# Top-level Analysis Function
# ============================

def analysis(param_traj,
             output_log,
             batch_size,
             lr,
             path='./analysis_result',
             num_epochs=None,
             threshold=1e-3,
             if_mask=True,
             steps=None,
             name=None):
    """
    全部以 between 为核心的顶层分析函数：
      0. 创建 result_dir / result_path
      1. 计算 unmasked Hilbert / ratios_between
      2. 画 unmasked between 图（global + zoom）
      3. 写 unmasked 文本统计
      4. 如果 if_mask:
         - 计算 masked Hilbert / ratios_between
         - 画 masked between 图
         - 写 masked 文本统计
      5. 把所有轨迹数据保存为 json（包含 masked / unmasked）
    """

    if name is None:
        name = f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}"

    if steps is not None:
        num_epochs = steps
    elif num_epochs is None:
        raise ValueError("Either num_epochs or steps must be provided.")

    # 0. 结果目录
    result_dir = os.path.join(path, name)
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(result_dir, f"{name}.txt")

    # 避免覆盖已有 txt
    index = 1
    base_txt = result_path
    while os.path.exists(result_path):
        result_path = os.path.join(result_dir, f"{name}_v{index}.txt")
        index += 1

    if sys.path and "swiss_roll_models" in sys.path[0]:
        sys.path.pop(0)

    # 1. Unmasked metrics
    metrics_unmasked = compute_hilbert_metrics(
        param_traj=param_traj,
        threshold=threshold,
        if_mask=False,
        if_threshold=True,
        if_self_adaptive=False,
        w_star=None,
    )
    hilbert_to_final = metrics_unmasked["hilbert_to_final"]
    hilbert_between = metrics_unmasked["hilbert_between"]
    hilbert_to_init = metrics_unmasked["hilbert_to_init"]
    ratios_between = metrics_unmasked["ratios_between"]

    # 2. Unmasked between plots
    plot_between_global_smoothed(
        ratios_between,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked",
    )
    plot_between_zoom_last(
        ratios_between,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked",
        zoom_len=300,
        window=10,
    )
    plot_between_zoom_last(
        ratios_between,
        batch_size, lr, num_epochs,
        result_dir,
        suffix="unmasked_last100",
        zoom_len=100,
        window=5,
    )

    # 3. 文本：模型信息 + log + unmasked between 统计
    with open(result_path, "a", encoding="utf-8") as f:
        f.write("===== Models Information =====\n\n")
        f.write(f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}\n\n")
        f.write("training log\n")
        f.write(output_log)
        f.write("\n\n")

        write_unmasked_between_stats(f, ratios_between)

        if not if_mask:
            # 保存 json（unmasked）
            with open(os.path.join(result_dir, "hilbert_to_final.json"), "w", encoding="utf-8") as jf:
                json.dump(hilbert_to_final, jf, ensure_ascii=False)

            with open(os.path.join(result_dir, "hilbert_between.json"), "w", encoding="utf-8") as jf:
                json.dump(hilbert_between, jf, ensure_ascii=False)

            with open(os.path.join(result_dir, "hilbert_to_init.json"), "w", encoding="utf-8") as jf:
                json.dump(hilbert_to_init, jf, ensure_ascii=False)

            with open(os.path.join(result_dir, "ratios_between.json"), "w", encoding="utf-8") as jf:
                json.dump(ratios_between, jf, ensure_ascii=False)

            return {
                "hilbert_to_final": hilbert_to_final,
                "hilbert_between": hilbert_between,
                "hilbert_to_init": hilbert_to_init,
                "ratios_between": ratios_between,
            }

        # 4. Masked metrics
        para_traj2 = param_traj
        w_star_raw = para_traj2[-1].clone()

        metrics_masked = compute_hilbert_metrics(
            param_traj=para_traj2,
            threshold=threshold,
            if_mask=True,
            if_threshold=False,
            if_self_adaptive=False,
            w_star=w_star_raw,
        )
        hilbert_to_final2 = metrics_masked["hilbert_to_final"]
        hilbert_between2 = metrics_masked["hilbert_between"]
        hilbert_to_init2 = metrics_masked["hilbert_to_init"]
        ratios_between2 = metrics_masked["ratios_between"]

        # Masked 文本统计
        write_masked_between_stats(f, ratios_between2)

    # 5. Masked plots
    plot_masked_between(
        ratios_between2,
        batch_size, lr, num_epochs,
        result_dir,
    )

    # 6. 全部轨迹保存为 json
    # --- unmasked ---
    with open(os.path.join(result_dir, "hilbert_to_final.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_to_final, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "hilbert_between.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_between, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "hilbert_to_init.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_to_init, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "ratios_between.json"), "w", encoding="utf-8") as jf:
        json.dump(ratios_between, jf, ensure_ascii=False)

    # --- masked ---
    with open(os.path.join(result_dir, "hilbert_to_final_masked.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_to_final2, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "hilbert_between_masked.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_between2, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "hilbert_to_init_masked.json"), "w", encoding="utf-8") as jf:
        json.dump(hilbert_to_init2, jf, ensure_ascii=False)

    with open(os.path.join(result_dir, "ratios_between_masked.json"), "w", encoding="utf-8") as jf:
        json.dump(ratios_between2, jf, ensure_ascii=False)

    return {
        "hilbert_to_final": hilbert_to_final,
        "hilbert_between": hilbert_between,
        "hilbert_to_init": hilbert_to_init,
        "ratios_between": ratios_between,
        "hilbert_to_final_masked": hilbert_to_final2,
        "hilbert_between_masked": hilbert_between2,
        "hilbert_to_init_masked": hilbert_to_init2,
        "ratios_between_masked": ratios_between2,
    }
