import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Hilbert_computation import hilbert_computation as hc


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


# ============================
# New helper functions
# ============================

def plot_hb(
    values=None,
    *,
    if_smooth=True,
    smooth_window=50,
    max_points=1000,
    if_plot=False,
    if_save=True,
    saved_name="hilbert_plot.png",
    saved_path=".",
    if_show=False,
):
    """Plot Hilbert metrics with optional smoothing and saving.

    Args:
        values: 1D array-like Hilbert metric values.
        if_smooth: Whether to smooth the curve with a moving average.
        smooth_window: Window size for smoothing.
        max_points: Maximum number of points to keep after downsampling.
        if_plot: Whether to render the plot (useful in notebooks).
        if_save: Whether to save the plot.
        saved_name: File name for saving.
        saved_path: Directory for saving (relative path is supported).
        if_show: Whether to call ``plt.show`` after plotting.
    """

    values = np.asarray(values, dtype=float)
    plt.figure()

    if if_smooth:
        x_axis, y_axis = moving_average_xy(values, window=max(1, smooth_window))
    else:
        x_axis = np.arange(len(values), dtype=float)
        y_axis = values

    x_axis, y_axis = downsample_xy(x_axis, y_axis, max_points=max_points)

    if len(y_axis) > 0:
        plt.plot(x_axis, y_axis, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("Hilbert metric")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.tight_layout()

    save_dir = Path(saved_path)
    if if_save:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / saved_name)

    if if_plot or if_show:
        plt.show()

    plt.close()


def write_stats(
    values,
    *,
    if_print=False,
    if_save=True,
    saved_name="hilbert_stats.txt",
    saved_path=".",
):
    """Write statistics for a Hilbert metric sequence.

    Returns the generated report string.
    """

    sections = ["===== Hilbert Metric Statistics ====="]
    segments = [
        ("first 200", values[:200]),
        ("200 to 400", values[200:400]),
        ("last 200", values[-200:]),
    ]

    for title, segment in segments:
        desc = describe_segment(segment)
        sections.append(f"\nThe {title} steps:")
        if desc is None:
            sections.append("  No data available.")
            continue
        sections.append(f"  Mean ≈ {desc['mean']:.4f}")
        sections.append(
            f"  Min ≈ {desc['min']:.4f}, Max ≈ {desc['max']:.4f}"
        )
        sections.append(
            f"  Q25 ≈ {desc['q25']:.4f}, Q75 ≈ {desc['q75']:.4f}"
        )

    report = "\n".join(sections) + "\n"

    if if_print:
        print(report)

    if if_save:
        save_dir = Path(saved_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / saved_name, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def _hilbert_to_target_sequence(trajectory, w_star):
    """Compute Hilbert distances between each step and ``w_star``.

    This helper calls ``hc.compute_hilbert_between_steps`` on pairs of
    ``(step, w_star)`` to respect the instruction of using that API.
    """

    distances = []
    for vect in trajectory:
        pair = [vect, w_star]
        res = hc.compute_hilbert_between_steps(pair)
        distances.append(res[0] if res else float("nan"))
    return distances


def analysis_to_w_star(
    trajectory,
    w_star,
    if_plot,
    if_writes,
    *,
    if_save=True,
    save_name="analysis_to_w_star",
    saved_path=".",
    if_show=False,
    if_masked=True,
    if_unmasked=True,
    threshold=1e-10,
):
    """Analyze Hilbert distances between trajectory steps and ``w_star``.

    Masking is applied relative to ``w_star`` via ``hc.mask_by_wstar_support``
    when requested.
    """

    results = {}
    save_dir = Path(saved_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    if if_unmasked:
        unmasked = _hilbert_to_target_sequence(trajectory, w_star)
        results["hilbert_to_w_star"] = unmasked

        if if_plot:
            plot_hb(
                unmasked,
                saved_name=f"{save_name}_to_w_star.png",
                saved_path=save_dir,
                if_show=if_show,
            )

        if if_writes:
            results["unmasked_stats"] = write_stats(
                unmasked,
                saved_name=f"{save_name}_to_w_star_stats.txt",
                saved_path=save_dir,
            )

        if if_save:
            with open(save_dir / f"{save_name}_to_w_star.json", "w", encoding="utf-8") as jf:
                json.dump(unmasked, jf, ensure_ascii=False)

    if if_masked:
        masked_traj, masked_w_star, _ = hc.mask_by_wstar_support(
            trajectory, w_star, threshold
        )
        masked_list = [masked_traj[i] for i in range(masked_traj.shape[0])]
        masked = _hilbert_to_target_sequence(masked_list, masked_w_star)
        results["hilbert_to_w_star_masked"] = masked

        if if_plot:
            plot_hb(
                masked,
                saved_name=f"{save_name}_to_w_star_masked.png",
                saved_path=save_dir,
                if_show=if_show,
            )

        if if_writes:
            results["masked_stats"] = write_stats(
                masked,
                saved_name=f"{save_name}_to_w_star_masked_stats.txt",
                saved_path=save_dir,
            )

        if if_save:
            with open(
                save_dir / f"{save_name}_to_w_star_masked.json",
                "w",
                encoding="utf-8",
            ) as jf:
                json.dump(masked, jf, ensure_ascii=False)

    return results


def analysis_to_w_between(
    trajectory,
    if_plot,
    if_writes,
    *,
    if_save=True,
    save_name="analysis_between",
    saved_path=".",
    if_show=False,
    threshold=1e-10,
    if_masked=True,
    if_self_adaptive=False,
):
    """Analyze Hilbert distances between consecutive trajectory steps."""

    results = {}
    save_dir = Path(saved_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    unmasked_between = hc.compute_hilbert_between_steps(
        trajectory,
        threshold=threshold,
        ifmask=False,
        if_self_adaptive=if_self_adaptive,
        if_threshold=True,
    )
    results["hilbert_between"] = unmasked_between

    if if_plot:
        plot_hb(
            unmasked_between,
            saved_name=f"{save_name}_between.png",
            saved_path=save_dir,
            if_show=if_show,
        )

    if if_writes:
        results["unmasked_stats"] = write_stats(
            unmasked_between,
            saved_name=f"{save_name}_between_stats.txt",
            saved_path=save_dir,
        )

    if if_save:
        with open(save_dir / f"{save_name}_between.json", "w", encoding="utf-8") as jf:
            json.dump(unmasked_between, jf, ensure_ascii=False)

    if if_masked:
        masked_between = hc.compute_hilbert_between_steps(
            trajectory,
            threshold=threshold,
            ifmask=True,
            if_self_adaptive=if_self_adaptive,
            if_threshold=False,
        )
        results["hilbert_between_masked"] = masked_between

        if if_plot:
            plot_hb(
                masked_between,
                saved_name=f"{save_name}_between_masked.png",
                saved_path=save_dir,
                if_show=if_show,
            )

        if if_writes:
            results["masked_stats"] = write_stats(
                masked_between,
                saved_name=f"{save_name}_between_masked_stats.txt",
                saved_path=save_dir,
            )

        if if_save:
            with open(
                save_dir / f"{save_name}_between_masked.json",
                "w",
                encoding="utf-8",
            ) as jf:
                json.dump(masked_between, jf, ensure_ascii=False)

    return results


# ============================
# Top-level Analysis Function
# ============================

def analysis(
    param_traj,
    output_log,
    batch_size,
    lr,
    path="./analysis_result",
    num_epochs=None,
    threshold=1e-3,
    if_mask=True,
    steps=None,
    name=None,
):
    """Run analysis for Hilbert metrics using the new helper functions."""

    if name is None:
        name = f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}"

    if steps is not None:
        num_epochs = steps
    elif num_epochs is None:
        raise ValueError("Either num_epochs or steps must be provided.")

    result_dir = Path(path) / name
    result_dir.mkdir(parents=True, exist_ok=True)

    text_path = result_dir / f"{name}.txt"
    index = 1
    while text_path.exists():
        text_path = result_dir / f"{name}_v{index}.txt"
        index += 1

    # training log
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("===== Models Information =====\n\n")
        f.write(f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}\n\n")
        f.write("training log\n")
        f.write(output_log)
        f.write("\n\n")

    # Between-step analysis
    between_results = analysis_to_w_between(
        param_traj,
        if_plot=True,
        if_writes=True,
        if_save=True,
        save_name=name,
        saved_path=result_dir,
        if_show=False,
        threshold=threshold,
        if_masked=if_mask,
    )

    # To w_star analysis
    w_star = param_traj[-1]
    w_star_results = analysis_to_w_star(
        param_traj,
        w_star,
        if_plot=True,
        if_writes=True,
        if_save=True,
        save_name=name,
        saved_path=result_dir,
        if_show=False,
        if_masked=if_mask,
        if_unmasked=True,
        threshold=threshold,
    )

    with open(text_path, "a", encoding="utf-8") as f:
        f.write("===== Between-step Hilbert Statistics =====\n")
        if "unmasked_stats" in between_results:
            f.write(between_results["unmasked_stats"])
        if "masked_stats" in between_results:
            f.write("\n" + between_results["masked_stats"])

        f.write("\n===== Hilbert to w* Statistics =====\n")
        if "unmasked_stats" in w_star_results:
            f.write(w_star_results["unmasked_stats"])
        if "masked_stats" in w_star_results:
            f.write("\n" + w_star_results["masked_stats"])

    combined_results = {**between_results, **w_star_results}
    return combined_results

