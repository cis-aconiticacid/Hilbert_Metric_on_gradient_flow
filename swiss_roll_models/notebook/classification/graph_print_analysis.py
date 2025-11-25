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
# Tools functions
# ============================

def moving_average_xy(y, window=50):
    """
    Average (step, y) ，Return (x_smooth, y_smooth)，
    x_smooth is the moving average of step, ensuring the x-axis aligns with the actual step.
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
    Used to avoid overly dense plots without relying on a fixed stride.
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
    Compute statistics for a segment of ratio values, returning a dict with mean/min/max/Q25/Q75.
    NaN values are filtered out; if all are NaN, return None.
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
    Call hda.analysis_distance_on_cone, returning:
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
# Plotting functions (only responsible for plotting)
# ============================

def plot_hilbert_distance_smoothed(hilbert_to_final,
                                   batch_size, lr, num_epochs,
                                   result_dir,
                                   suffix="unmasked",save_name=None):
    """
    TODO #1: Use moving_average_xy for smoothing,
    plot the log-scale curve of d_H(w_t, w*).
    """
    window_h = max(20, len(hilbert_to_final) // 100)  # Adapt based on length
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
    if save_name is None:
        output_path = os.path.join(
            result_dir,
            f"HTF_smooth_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
        )
    else:
        output_path = save_name
    plt.savefig(output_path)
    plt.close()


def plot_ratio_global_smoothed(ratios,
                               batch_size, lr, num_epochs,
                               result_dir,
                               suffix="unmasked",
                               save_name=None):
    """
    TODO #1/#4: Plot the globally smoothed ratio.
    - Instead of removing NaN, replace them with 1.0 first.
    - Use moving_average_xy to align steps.
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
    if save_name is None:
        output_path_ratio = os.path.join(
            result_dir,
            f"HR_smooth_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
        )
    else:
        output_path_ratio = save_name
    plt.savefig(output_path_ratio)
    plt.close()


def plot_ratio_zoom_last(ratios,
                         batch_size, lr, num_epochs,
                         result_dir,
                         suffix="unmasked",
                         zoom_len=300,
                         window=10,
                         save_name=None):
    """
    TODO #3: Only look at the last zoom_len steps of the ratio.
    - Use a very small window (or window=1 for almost no smoothing) to observe end fluctuations.
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

    # Use a smaller window within the zoomed section
    window = min(window, len(ratio_zoom)) if window > 1 else 1
    x_zoom, ratio_zoom_smooth = moving_average_xy(ratio_zoom, window=window)

    # x_zoom is relative index, so shift to real step
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
    if save_name is None:
        zoom_path = os.path.join(
            result_dir,
            f"HR_zoom_last{zoom_len}_{suffix}_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
        )
    else:
        zoom_path = save_name
    plt.savefig(zoom_path)
    plt.close()


def plot_masked_hilbert_and_ratio(hilbert_to_final2,
                                  ratios_to_final2,
                                  batch_size, lr, num_epochs,
                                  result_dir):
    """
    TODO #1/#3/#4: Plot the Hilbert and ratio for Masked.
    - Hilbert: log-scale original curve + smoothed version (optional)
    - ratio: globally smoothed + last 300/100 steps zoom in
    """
    # Hilbert distance (masked, Original log-scale)
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

    # ratio globally smoothed + zoom in
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
# Text Writing (Statistics) Functions
# ============================

def write_unmasked_stats(f, ratios_to_final):
    """
    Write statistics for unmasked ratios:
    - First 200 steps
    - Steps 200–400
    - Last 200 steps
    Including mean/min/max/Q25/Q75 (TODO #5)
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


def write_stats_in_range(
    f,
    start: int,
    end: int,
    ratios,
    step: int,
    name: str,
    if_average: bool = False,
):
    """
    Write statistics of ratios in the range [start, end) to the file f.

    - step: The actual training step interval (e.g., if recorded every 10 steps, pass 10)
    - name: The name of this range, such as "no_mask" / "masked"
    - if_average: If True, average this segment first (used for average results of multiple curves)
    """
    # Boundary protection
    n = len(ratios)
    if start < 0:
        start = 0
    if end > n:
        end = n
    if start >= end:
        f.write(f"\n[WARN] Empty range for {name}: start={start}, end={end}\n")
        return

    segment = ratios[start:end]

    # If averaging is needed (e.g., you passed in a list of multiple experimental results)
    if if_average:
        import numpy as np
        segment = np.array(segment, dtype=float)
        # Simple approach: take the mean of this entire segment, other statistics are handled by describe_segment as usual
        avg_value = float(np.nanmean(segment))
        # Record an overall mean
        f.write(
            f"\nSteps {start*step} to {end*step} "
            f"(index {start} to {end}, {name}, averaged):\n"
        )
        f.write(f"  Overall mean ≈ {avg_value:.4f}\n")

    # Call your original statistics function
    desc = describe_segment(segment)
    if desc is None:
        f.write(
            f"\nSteps {start*step} to {end*step} "
            f"(index {start} to {end}, {name}):\n"
        )
        f.write("  [WARN] No valid data in this range.\n")
        return

    # Real step range (considering sampling interval)
    real_start = start * step
    real_end = end * step

    f.write(
        f"\nSteps {real_start} to {real_end} "
        f"(index {start} to {end}, {name}):\n"
    )
    f.write(f"  Mean ≈ {desc['mean']:.4f}\n")
    f.write(f"  Min ≈ {desc['min']:.4f}, Max ≈ {desc['max']:.4f}\n")
    f.write(f"  Q25 ≈ {desc['q25']:.4f}, Q75 ≈ {desc['q75']:.4f}\n")




# ============================
# Top-level Analysis Function
# ============================

def analysis(param_traj, output_log, batch_size, lr, path='./model_result/',num_epochs=None,
             threshold=1e-3, if_mask=True, steps=None,name=None):
    """
    Top-level analysis function:
      0. Create result_dir and result_path
      1. Create result_dir and result_path
      2. Compute unmasked Hilbert / ratio
      3. Plot unmasked graphs (global + zoom)
      4. Write unmasked text statistics
      5. If if_mask=True:
         - Compute masked Hilbert / ratio
         - Plot masked graphs (global + zoom)
         - Write masked text statistics
      6. Return relevant results (for further complex analysis)

    For error cases, do not catch with try/except; let errors raise directly:
      - If both num_epochs and steps are None, raise ValueError (consistent with original logic)
    """

    if name is None:
        name = f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}"

    if steps is not None:
        num_epochs = steps
    elif num_epochs is None:
        raise ValueError("Either num_epochs or steps must be provided.")

    # 0. Create result_dir and result_path
    result_dir = path+f"{name}"
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(
        result_dir,
        f"{name}.txt"
    )

    # Avoid overwriting existing files
    index=1
    while os.path.exists(result_path):
        result_path = os.path.join(
            result_dir,
            f"{name}_v{index}.txt"
        )
        index+=1

    if sys.path and "swiss_roll_models" in sys.path[0]:
        sys.path.pop(0)

    # 1. Unmasked Hilbert / ratio
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

    # 2. non-masked graph：Hilbert smoothing + ratio global smoothing + ratio zoom at the end
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

    # 3. Write text: model information + training log + unmasked statistics
    with open(result_path, "a", encoding="utf-8") as f:
        f.write("===== Models Information =====\n\n")
        f.write(f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}\n\n")
        f.write("training log\n")
        f.write(output_log)
        f.write("\n\n")

        write_unmasked_stats(f, ratios_to_final)

        # 4. If masked analysis is not needed, end here
        if not if_mask:
            # Return unmasked results
            return {
                "hilbert_to_final": hilbert_to_final,
                "ratios_to_final": ratios_to_final,
                "ratios_between": ratios_between,
            }

        # 5. Masked analysis
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
        ratios_to_final2 = metrics_masked["ratios_to_final"]
        ratios_between2 = metrics_masked["ratios_between"]

        # Text statistics (masked)
        write_masked_stats(f, ratios_to_final2)

    # 6. Masked graphs (Hilbert + ratio global + ratio zoom)
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



def compute_hilbert_metrics_to_step(
    param_traj,
    start,
    end,
    threshold=1e-3,
    if_mask=False,
    if_threshold=True,
    if_self_adaptive=False,
    w_star=None,
):
    """
    在给定区间 [start, end] 上计算 Hilbert 度量相关量：
      - hilbert_to_n: 每一步到目标向量 w* 的 Hilbert 距离
      - ratio_to_n: hilbert_to_n 的相邻比值
      - hilbert_between: 相邻两步之间的 Hilbert 距离
      - ratio_between: hilbert_between 的相邻比值

    参数:
      param_traj: 整条参数轨迹 (list / numpy / torch 都行，只要可切片)
      start: 子轨迹起始下标（含）
      end:   子轨迹终止下标（含）
      threshold, if_mask, if_threshold, if_self_adaptive: 直接传给 hda.analysis_distance_on_cone
      w_star: 目标向量；如果为 None，默认用 param_traj[end]（也就是子轨迹最后一步）
    """
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start}).")
    if start < 0 or end >= len(param_traj):
        raise IndexError(
            f"Invalid range [{start}, {end}] for param_traj of length {len(param_traj)}."
        )

    # 取子轨迹 [start, end]
    sub_traj = param_traj[start : end + 1]

    # 默认目标向量用这一段的最后一点 (即第 end 步)
    if w_star is None:
        w_star = sub_traj[-1]

    # 直接复用你已有的 cone 分析函数
    res = hda.analysis_distance_on_cone(
        param_traj=sub_traj,
        w_star=w_star,
        threshold=threshold,
        ifmask=if_mask,
        if_threshold=if_threshold,
        if_self_adaptive=if_self_adaptive,
    )

    hilbert_to_n = res["hilbert_to_final"]
    hilbert_between = res["hilbert_between"]
    hilbert_to_init = res["hilbert_to_init"]  # 如果你想留着也可以一起返回

    # ratio_to_n: 相对于 w* 的 Hilbert 距离的相邻比值
    ratio_to_n = []
    for t in range(len(hilbert_to_n) - 1):
        if hilbert_to_n[t] > 0:
            ratio_to_n.append(hilbert_to_n[t + 1] / hilbert_to_n[t])
        else:
            ratio_to_n.append(float("nan"))

    # ratio_between: 相邻两步之间 Hilbert 距离的相邻比值
    ratio_between = []
    for t in range(len(hilbert_between) - 1):
        if hilbert_between[t] > 0:
            ratio_between.append(hilbert_between[t + 1] / hilbert_between[t])
        else:
            ratio_between.append(float("nan"))

    return {
        "start": start,
        "end": end,
        "hilbert_to_n": hilbert_to_n,
        "ratio_to_n": ratio_to_n,
        "hilbert_between": hilbert_between,
        "ratio_between": ratio_between,
        "hilbert_to_init": hilbert_to_init,  # 可选，但顺手一起给
    }
