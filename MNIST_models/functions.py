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

def smooth(x, window=50):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")

def analysis(param_traj, output_log, batch_size, lr, num_epochs=None,
             threshold=1e-3, if_mask=True, steps=None):
    if steps is not None:
        num_epochs = steps
    elif num_epochs is None:
        raise ValueError("Either num_epochs or steps must be provided.")
    # ============================
    # 0. 为这一次调用创建单独文件夹
    # ============================
    result_dir = './analysis_result/'+f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}"
    os.makedirs(result_dir, exist_ok=True)

    # 原来的 result_path 现在放到这个文件夹里
    result_path = os.path.join(result_dir,
                               f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}.txt")

    index=1
    while os.path.exists(result_path):
        result_path = os.path.join(
            result_dir,
            f"Analysis_bs{batch_size}_lr{lr}_ep{num_epochs}_v{index}.txt"
        )
        index += 1
    # 不再 chdir，也不再用 original_path 了
    # original_path = os.getcwd()

    #（你原来这里好像是预备用 torch.save？我先留着不动）
    # torch.save

    if sys.path and "swiss_roll_models" in sys.path[0]:
        sys.path.pop(0)

    # ============================
    # 1. 计算 ratio（保留你原来的逻辑）
    # ============================
    res1=hda.analysis_distance_on_cone(
        param_traj=param_traj,
        w_star=param_traj[-1],
        threshold=threshold,
        ifmask=False,
        if_threshold=True,
        if_self_adaptive=False,
    )
    hilbert_to_final = res1["hilbert_to_final"]
    hilbert_between = res1["hilbert_between"]
    hilbert_to_init = res1["hilbert_to_init"]
    ratios_to_final = []
    ratios_between = []
    for t in range(len(hilbert_to_final) - 1):
        if hilbert_to_final[t] > 0:
            ratios_to_final.append(hilbert_to_final[t+1] / hilbert_to_final[t])
        else:
            ratios_to_final.append(float("nan"))
    for t in range(len(hilbert_between) - 1):
        if hilbert_between[t] > 0:
            ratios_between.append(hilbert_between[t+1] / hilbert_between[t])
        else:
            ratios_between.append(float("nan"))
    # ============================
    # 2. 无 mask 的 Hilbert 曲线图
    # ============================
    def smooth(x, window=50):
        """滑动平均，用于平滑视觉趋势"""
        x = np.array(x, dtype=float)
        if len(x) < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="valid")
    window_h = 50  # 你可以调大一点，比如 100，看起来更平滑
    hilbert_smooth = smooth(hilbert_to_final, window=window_h)

    steps_smooth = list(range(len(hilbert_smooth)))

    plt.figure()
    plt.semilogy(steps_smooth, hilbert_smooth, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("d_H(w_t, w*) (log scale, smoothed)")
    plt.title(
        "Hilbert distance to w* (smoothed)\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    output_path = os.path.join(
        result_dir,
        f"HTF_smooth_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path)
    plt.close()

    # ============================
    # 3. 无 mask 的 ratio 曲线图（stride 采样）
    # ============================
    ratio_array = np.array(ratios_to_final, dtype=float)

    # 去掉 NaN
    ratio_clean = ratio_array[np.isfinite(ratio_array)]

    # ratio 趋势更需要平滑
    window_r = 100   # 这个可以调，越大越平滑
    ratio_smooth = smooth(ratio_clean, window=window_r)

    steps_r = list(range(len(ratio_smooth)))

    plt.figure()
    plt.plot(steps_r, ratio_smooth, linewidth=2)
    plt.xlabel("step t")
    plt.ylabel("smoothed ratio d_H(t+1)/d_H(t)")
    plt.axhline(1.0, linestyle="--")
    plt.title(
        f"Hilbert contraction ratio (smoothed)\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    output_path_ratio = os.path.join(
        result_dir,
        f"HR_smooth_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path_ratio)
    plt.close()

    # ============================
    # 4. 写 Analysis 文本（无 mask 分析）
    # ============================
    # 这里我只改了文件路径，内容完全保留
    f = open(result_path, "a", encoding="utf-8")
    f.write("===== Models Information =====\n\n")
    f.write(f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}\n\n")
    f.write("training log\n")
    f.write(output_log)
    f.write("\n")

    f.write("===== No Masking Analysis Results =====\n")
    front_200= ratios_to_final[:200]
    medium_200_400 = ratios_to_final[200:400]
    f.write("The first 200 Steps ratio_to_final:\n")
    if len(front_200) > 0:
        front_200_clean = [r for r in front_200 if not math.isnan(r)]
        f.write(f"  Mean ≈ {sum(front_200_clean)/len(front_200_clean):.4f}\n")
        f.write(f"  Min ≈ {min(front_200_clean):.4f}, Max ≈ {max(front_200_clean):.4f}\n")
    f.write("The 200 to 400 Steps ratio_to_final:\n")
    if len(medium_200_400) > 0:
        medium_200_400_clean = [r for r in medium_200_400 if not math.isnan(r)]
        f.write(f"  Mean ≈ {sum(medium_200_400_clean)/len(medium_200_400_clean):.4f}\n")
        f.write(f"  Min ≈ {min(medium_200_400_clean):.4f}, Max ≈ {max(medium_200_400_clean):.4f}\n")
    tail_200 = ratios_to_final[-200:]
    tail_200_clean = [r for r in tail_200 if not math.isnan(r)]
    f.write("\nThe last 200 Steps ratio_to_final:\n")
    
    if len(tail_200_clean) > 0:
        f.write(f"  Mean ≈ {sum(tail_200_clean)/len(tail_200_clean):.4f}\n")
        f.write(f"  Min ≈ {min(tail_200_clean):.4f}, Max ≈ {max(tail_200_clean):.4f}\n")

    # 这里是你原来的：
    para_traj2 = param_traj   # The original positive trajectory (already abs+1e-6 processed)
    w_star_raw = para_traj2[-1].clone()

    if not if_mask:
        f.close()
        return

    # ============================
    # 5. Masked Cone 分析（完全保留）

    f.write("===== Masked Cone Analysis Results =====\n")

    res2=hda.analysis_distance_on_cone(
        param_traj=para_traj2,
        w_star=w_star_raw,
        threshold=1e-5,
        ifmask=True,
        if_threshold=False,
        if_self_adaptive=False,
    )
    hilbert_to_final2 = res2["hilbert_to_final"]
    hilbert_between2 = res2["hilbert_between"]
    hilbert_to_init2 = res2["hilbert_to_init"]

    ratios_to_final2 = []
    for t in range(len(hilbert_to_final2) - 1):
        if hilbert_to_final2[t] > 0:
            ratios_to_final2.append(hilbert_to_final2[t+1] / hilbert_to_final2[t])
        else:
            ratios_to_final2.append(float("nan"))

    ratios_between2 = []
    for t in range(len(hilbert_between2) - 1):
        if hilbert_between2[t] > 0:
            ratios_between2.append(hilbert_between2[t+1] / hilbert_between2[t])
        else:
            ratios_between2.append(float("nan"))

    front_200_2= ratios_to_final2[:200]
    medium_200_400_2 = ratios_to_final2[200:400]
    f.write("\nFirst 200 Steps ratio_to_final:\n")
    if len(front_200_2) > 0:
        front_200_clean2 = [r for r in front_200_2 if not math.isnan(r)]
        f.write(f"  Mean ≈ {sum(front_200_clean2)/len(front_200_clean2):.4f}\n")
        f.write(f"  Min ≈ {min(front_200_clean2):.4f}, Max ≈ {max(front_200_clean2):.4f}\n")
    f.write("200 to 400 Steps ratio_to_final:\n")
    if len(medium_200_400_2) > 0:
        medium_200_400_clean2 = [r for r in medium_200_400_2 if not math.isnan(r)]
        f.write(f"  Mean ≈ {sum(medium_200_400_clean2)/len(medium_200_400_clean2):.4f}\n")
        f.write(f"  Min ≈ {min(medium_200_400_clean2):.4f}, Max ≈ {max(medium_200_400_clean2):.4f}\n")
    f.write("\nLast 200 Steps ratio_to_final:\n")
    tail_200_2 = ratios_to_final2[-200:]
    tail_200_clean2 = [r for r in tail_200_2 if not math.isnan(r)]
    if len(tail_200_clean2) > 0:
        f.write(f"  Mean ≈ {sum(tail_200_clean2)/len(tail_200_clean2):.4f}\n")
        f.write(f"  Min ≈ {min(tail_200_clean2):.4f}, Max ≈ {max(tail_200_clean2):.4f}\n")

    # ============================
    # 6. Masked 曲线图（d_H / ratio）
    # ============================
    plt.figure()
    steps_r = list(range(len(hilbert_to_final2)))
    plt.semilogy(steps_r, hilbert_to_final2)
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

    stride = 10
    ratio_idx2 = list(range(0, len(ratios_to_final2), stride))
    ratio_vals2 = [ratios_to_final2[i] for i in ratio_idx2]

    plt.figure()
    plt.plot(ratio_idx2, ratio_vals2)
    plt.xlabel("step t")
    plt.ylabel("ratio d_H(w_{t+1}, w*) / d_H(w_t, w*)")
    plt.axhline(1.0, linestyle="--")
    plt.title(
        f"Hilbert contraction ratio (stride={stride}) Masked Cone\n"
        f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}"
    )
    plt.tight_layout()
    output_path_ratio2 = os.path.join(
        result_dir,
        f"HCRF_masked_bs{batch_size}_lr{lr}_ep{num_epochs}.png"
    )
    plt.savefig(output_path_ratio2)
    plt.close()

    f.close()
    return{
        "ratios_to_final": ratios_to_final,
        "ratios_between": ratios_between,
        "ratios_to_final_masked": ratios_to_final2,
    }
    # 不再 os.chdir(original_path)
