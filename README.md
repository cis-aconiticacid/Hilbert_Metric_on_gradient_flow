# cone_dynamics — Hilbert Metric on Gradient Flow

**作者**：Xinyang (Elizabeth) Wen · xwen57@wisc.edu
**主题**：在梯度下降动力学中观察 Hilbert projective metric 的压缩 / plateau-drop 行为
**对应邮件**：`/workspace/conversation_to_fronesco_tudisco/` （与 Francesco Tudisco 的通信）
**对应实验计划**：`/workspace/conversation_to_fronesco_tudisco/refine-logs/EXPERIMENT_PLAN.md`

> 这个 README 是给未来的自己（以及 Claude）写的导航地图。不负责把代码清干净 —— 只负责说清楚"哪里有什么"和"约定是什么"。已知的混乱在最后一节列出，之后可以分批修。

---

## 1. 项目在做什么

用最小化的 FC 模型在 Swiss Roll（回归/分类）和 MNIST 上跑梯度下降，**追踪最后一层权重 w_t 的 Hilbert projective metric 轨迹**：
$$d_H(u, v) = \log \frac{\max_i u_i / v_i}{\min_i u_i / v_i}, \quad u, v \in \mathbb{R}^d_{>0}.$$

观察到的两类现象：
- **Swiss Roll + softplus**：几何收缩（MSE）/ 阶段性（Huber）/ alternating-cone 2-cycle
- **MNIST FC-ReLU**：plateau–drop，drop 与 support indices 变化对齐，深度增大 → drop 更大、plateau 更长

---

## 2. 目录导航

```
cone_dynamics/
├── README.md                    ← 本文件
├── .gitignore                   ← 忽略 MNIST 数据、pyc、venv、LaTeX 中间产物
│
├── functions/                   ← 核心库：metric 计算 + 画图 + IO
│   ├── Hilbert_computation.py   ← 主角：Hilbert metric、mask、trajectory analysis
│   ├── distance.py              ← hilbert_distance 的独立版本（旧版）
│   ├── dataset.py               ← Swiss Roll 数据生成（regression / classification）
│   ├── dataload.py              ← 从 checkpoints 重建 param trajectory
│   ├── graph_print_analysis.py  ← plot_hb / analysis_to_w_star / analysis_to_w_between
│   └── load_my_file.py          ← 早期风格的 analysis pipeline（和 graph_print_analysis 有重叠）
│
├── MNIST_models/
│   ├── Nureon Network MNIST/    ← FC-ReLU MNIST 实验
│   │   ├── train_mnist_with_network.py   ← HBModel_MNIST.train_mnist_with_hilbert，主训练入口
│   │   ├── MNIST_Classification.ipynb    ← 当前的 MNIST 主 notebook
│   │   ├── full_batch_gd_hilbert.ipynb   ← full-batch 版本实验
│   │   ├── picture_handling.ipynb        ← 画图用
│   │   ├── MNIST_Classification_outdated.ipynb
│   │   └── testtool.py                   ← 批量画 ratios_between 的脚本
│   └── svm_MNIST/               ← SVM on MNIST（PF 实验备用方向，见第 6 节）
│       ├── linear_svm.py / kernel_svm.py
│       ├── train_mnist_svm.py / workflow.py
│       ├── analysis_utils.py / data_utils.py / evalutation.py
│       └── experiment_overview.ipynb
│
├── swiss_roll_models/
│   ├── Notes/
│   ├── regression_on_swiss_roll/         ← 给 Tudisco 那版 note 的主实验
│   │   ├── l2_loss_relumask.ipynb        ← MSE + ReLU mask
│   │   ├── swiss_roll_2d_model_l2loss.ipynb
│   │   ├── l2_regulizer_SP.ipynb         ← softplus + L2 reg
│   │   └── huber_loss_SP.ipynb           ← softplus + Huber loss
│   └── classification_on_swiss_roll/
│       ├── logistic_no_regular.ipynb
│       └── logistic_regular.ipynb
│
├── experiments/                 ← 保存下来的 trajectory / checkpoint
│   └── MNIST_network/
│       └── mnist_hilbert_bs256_lr0.01_decayTrue_regTrue_layers5_steps500_traj.pt
│
├── Noised_PageRank/             ← 早期探索，目前只有一个 TODO 文档
│   └── to main.txt
│
├── checkpoints/, outputs/, results/   ← 运行产物，目前为空（.gitignore 会忽略内容）
└── notes/
    └── Elizabeth_s_Research_Topics.pdf
```

---

## 3. 核心数学约定（重要 —— 改变这些会破坏已有观察的复现）

### 3.1 追踪的向量 `w_t`

在 MNIST 上，`w_t` **就是最后一层 `output_layer.weight` 被 flatten 成长度 `10 × hidden` 的向量**。

```python
# dataload.py:42 · train_mnist_with_network.py:271,319,350
w_t = model.output_layer.weight.detach().cpu().reshape(-1).clone()
```

不是按 class row 拆、不是按 hidden column 拆、不是 abs-then-sum。是**原始带符号**的 flat 权重，经过 clamp 进入正锥之后计算 Hilbert。

### 3.2 进正锥的方式：最后一步 `clamp(min=1e-10)`

```python
# Hilbert_computation.py:141-143
ref_traj = torch.clamp(ref_traj, min=eps)
w_init = torch.clamp(w_init, min=eps)
w_star_new = torch.clamp(w_star_new, min=eps)
```

无论 `ifmask` / `if_threshold` 分支走哪条，**最后都 clamp 到 1e-10**。这意味着负权重被截断成 1e-10，而不是取绝对值。

> ⚠️ 这个 clamp 是已有观察的基础。如果改成 `abs()` 或 per-column norm，现象可能变。后续如果要和新实验对齐，要保持这个约定，或明确标注换了。

### 3.3 Hilbert 距离本身

```python
# distance.py / Hilbert_computation.py.distance_func.hilbert_distance
ratio = x / y
d_H = log(ratio.max()) - log(ratio.min())
```

严格正向量要求（`<eps` 时抛 ValueError），在 `_prepare_trajectory` 里已由 clamp 保证。

### 3.4 "Mask by w* support"

```python
# Hilbert_computation.py:201-234 · mask_by_wstar_support
mask = (|w_star| > threshold)    # 注意是绝对值
# 然后 w_star 和 trajectory 都投影到这个 mask 上
```

用 **w* 的高幅值坐标** 定义"支撑锥"，只在这个子锥里看 Hilbert 轨迹。默认 threshold=1e-3。

### 3.5 w* = 谁

默认约定：**w* = trajectory 的最后一项**（`param_traj[-1]`）。
- `analysis()` / `analysis_to_w_star` 里都用这个默认
- 所以 `d_H(w_T, w*) = 0` 是 terminal boundary，现有 note 里的几何衰减是对这个锚点说的

> 这意味着：现有 Swiss Roll / MNIST 数据里的"Hilbert 到 w* 的几何收缩"是 **post-hoc 的同跑锚**。要做更严谨的声明，需要 two-pass anchor protocol（pass 1 训练到头存 anchor，pass 2 用固定 anchor 重训并 log）—— 新 `/workspace/refine-logs/mnist_hilbert.py` 已实现。

---

## 4. 模块职责 · `functions/`

| 文件 | 对外主要 API | 干什么 |
|---|---|---|
| `Hilbert_computation.py` | `hilbert_computation.analysis_distance_on_cone(param_traj, w_star, ...)` · `hilbert_computation.compute_hilbert_to_w_star` · `hilbert_computation.compute_hilbert_between_steps` · `hilbert_computation.mask_by_wstar_support` | 把一条 trajectory 转成 `d_H(w_t, w*)`, `d_H(w_t, w_0)`, `d_H(w_{t+1}, w_t)` 三条序列 |
| `distance.py` | `distance_func.hilbert_distance(x, y, eps)` | 单对向量的 Hilbert 距离（最简版） |
| `dataset.py` | `generate_swiss_roll(n_samples, task, D, noise, device)` | D 维 Swiss Roll（regression: y = sin(u)/(u+1)；classification: u>6π） |
| `dataload.py` | `build_param_traj(batchsize, lr, step_list, ModelClass)` · `loadresults(...)` | 从磁盘 checkpoint 序列重建 param trajectory |
| `graph_print_analysis.py` | `plot_hb(values, ...)` · `analysis_to_w_star(trajectory, w_star, ...)` · `analysis_to_w_between(trajectory, ...)` | 画 Hilbert 曲线 · 写统计文本 · 存 json |
| `load_my_file.py` | `analysis(param_traj, output_log, batch_size, lr, ...)` | 老的一站式分析（做 smoothing + ratio 图 + masked/unmasked 对比）。和 `graph_print_analysis.analysis` 功能重合 |

---

## 5. MNIST 实验入口

主模型：`train_mnist_with_network.py · MNISTNet`
- `number_of_layerss` 个 `Linear + ReLU` 隐层，最后一个 `Linear(hidden, 10)` 作为 `output_layer`
- hidden_dim 默认 256
- **追踪对象永远是 `output_layer.weight`**

主训练函数：`HBModel_MNIST.train_mnist_with_hilbert(...)`
- 参数：`batch_size, lr, num_epochs` 或 `max_steps`（二选一），`loss_type ∈ {ce, huber, mse}`, `if_regularize`, `regularization_coeff`, `if_regularize_all`, `seed`, `check_distance`, `if_record_test`
- 正则默认**只加在 output_layer**（因为在追踪它），想改成全部正则设 `if_regularize_all=True`
- 返回 `{model, param_traj, traj_steps, loss_traj, loss_steps, test_acc_record, output_log, ...}`

附加的两个入口：
- `HBModel_MNIST.train_full_batch_with_hilbert(model, images, labels, num_steps, ...)` — 一个 batch 完整重复 `num_steps` 次
- `HBModel_MNIST.train_with_optional_initialization(num_steps, if_initialize, ...)` — 包一层方便 notebook 调用

Notebook 使用约定：
```python
# 每个 MNIST notebook 都是这样 path hack 后再 import
sys.path.insert(0, str(Path.cwd().resolve().parents[1]))
from functions import graph_print_analysis as gp_tool
from train_mnist_with_network import HBModel_MNIST as hbm
```

---

## 6. Swiss Roll 实验入口

全在 `swiss_roll_models/regression_on_swiss_roll/` 和 `classification_on_swiss_roll/` 的 notebook 里：
- `l2_loss_relumask.ipynb` — MSE + ReLU (mask 版)
- `l2_regulizer_SP.ipynb` — Softplus + MSE + L2 正则（给 Tudisco 那版 note 的主要来源）
- `huber_loss_SP.ipynb` — Softplus + Huber
- `logistic_no_regular.ipynb` / `logistic_regular.ipynb` — classification

> 每个 notebook 都自带模型定义（不是 import 的）—— 以后要整合可以抽成 `swiss_roll_models/model.py`。

数据由 `functions/dataset.py · generate_swiss_roll` 生成（默认 D=32，task ∈ {regression, classification}）。

---

## 7. 下游 / 备用方向

- `MNIST_models/svm_MNIST/` — **SVM on MNIST**。email 里 Elizabeth 提到"future to avoid the difficulties on MNIST"就是这条。包含 `linear_svm.py`, `kernel_svm.py`（RBF），一个 `workflow.py` orchestration。还没接到 Hilbert pipeline 上。
- `Noised_PageRank/` — 早期想过的 PF 在 PageRank 上做噪声分析，目前只有 `to main.txt` 一个 TODO 文档，**暂时搁置**。

---

## 8. 运行环境

`.gitignore` 已经屏蔽：MNIST 原始数据、`__pycache__/`、各种 `venv/`、LaTeX 产物、jupyter checkpoint、`.DS_Store`/`Thumbs.db`。

没有 `requirements.txt`。实际依赖（从代码里推断）：
- `torch` + `torchvision`
- `numpy`, `matplotlib`, `scikit-learn`（svm_MNIST 里用）
- `jupyter`

设备：代码大部分 hard-code `"cuda"` 默认；`hilbert_computation._prepare_trajectory` 在 `device != 'cuda'` 时会打印一条慢速警告但不 crash。

---

## 9. 已知的混乱 · TODO (不急着修，先记下来)

> 列这些**不是为了让你现在去改**，而是给将来动大手术时一个 checklist。

### 代码质量
- [ ] `functions/Hilbert_computation.py` 结尾有一段**重复定义**的 `analysis_distance_on_cone` / `compute_hilbert_to_w_star` / `compute_hilbert_between_steps`（第 278–335 行），而且错用了 `@staticmethod` 装饰模块级函数。应该删掉，或者放到一个显式的 `__all__ = [...]` 里。
- [ ] `MNIST_models/Nureon Network MNIST/testtool.py` 里有 **Windows 绝对路径** `C:\Users\ASUS\Desktop\cone_dynamics\...` —— 换机器就挂。改成相对 `Path(__file__).resolve().parents[2]`。
- [ ] `distance.py` 和 `Hilbert_computation.py.distance_func` 定义了**两个版本的 `hilbert_distance`**（一个带 clamp 开关，一个硬性要求正）。统一掉。
- [ ] Notebook 里也又手写了一版 `hilbert_distance`（`full_batch_gd_hilbert.ipynb`）—— 应该 import。
- [ ] 所有 notebook 的 `sys.path` hack 很脆（`Path.cwd().parents[1]`、`.parents[2]` 混用）。建议把 `functions/` 装成本地包（加一个 `functions/__init__.py` + 项目根 `setup.py` 或 `pyproject.toml`）。
- [ ] `load_my_file.py · analysis()` 和 `graph_print_analysis.py · analysis()` 功能**几乎重叠**，但签名不同、输出格式不同。合并成一个。
- [ ] 文件夹名 `Nureon Network MNIST` 里**有空格**（import 不了），而且 `Nureon` 应该是 `Neuron` 的 typo。建议改成 `mnist_fc_relu`。
- [ ] `svm_MNIST/evalutation.py` typo（应为 `evaluation.py`）。

### 实验可复现
- [ ] 没有 `requirements.txt` / `pyproject.toml` / `environment.yml`。
- [ ] 各 notebook / 脚本的随机种子不统一（`seed=1919810`, `seed=42` 等散落在不同文件）。
- [ ] **没有 deterministic CUDA 配置** —— Hilbert support-switch 对数值噪声敏感，对 reproduce 不利。
- [ ] `experiments/MNIST_network/test/` 为空。

### 科学约定
- [ ] **w\* = 同跑最后一个 iterate** 这个约定在给 Tudisco 的新 note 里值得讨论 → 见 `/workspace/refine-logs/mnist_hilbert.py` 的 two-pass anchor protocol。
- [ ] Hilbert 距离对 `clamp(min=1e-10)` 敏感（见 Codex 审核报告 `/workspace/refine-logs/REVIEW_LOG.md`）。tiny 权重被 clamp 后会放大 ratio → 要么换成 mask + log-space，要么承认并明确报告 clamp 的影响。

### 文档
- [ ] 没有 Swiss Roll 实验的独立 README，全散在 notebook 里。
- [ ] `Noised_PageRank/to main.txt` 的内容应该要么合并到这个 README 要么删掉。

---

## 10. 和新实验的接口（2026-04-20 新增）

新实现的 `/workspace/refine-logs/mnist_hilbert.py` 是按 `EXPERIMENT_PLAN.md` 的 B1–B4 写的，和这里的约定有三处不同，目前需要对齐：

| 约定 | 旧代码（cone_dynamics） | 新代码（mnist_hilbert.py） |
|---|---|---|
| w_t 的定义 | `output_layer.weight.flatten()` (长度 10·H) | `W.abs().sum(dim=0)` (长度 H) —— **Codex 审核推荐** |
| 进正锥 | `clamp(min=1e-10)` | mask + log-space (relative tol) |
| anchor | 同跑最后一步 | two-pass，pass-1 存 anchor，pass-2 用 fixed anchor log |

**下次继续时第一件事**：决定对齐到哪一边。建议：
- 如果**要复现这里已有的观察**（plateau-drop + support switch），对齐到这边（flatten + clamp）—— 新脚本加一个 `--support_reduce flatten_clamp` 开关
- 如果**要给 Tudisco 发更严谨的新 note**，用新脚本的 two-pass + mask + per-hidden-unit —— 但要补一张对照图证明现象不是 clamp 伪影

---

## 11. 代码修复记录（2026-04-20）

根据第 9 节 TODO 执行的一次批量修复，按文件列出。

### `functions/Hilbert_computation.py`

**Bug 1（运行时崩溃）** — `analysis_distance_on_cone` 第 267 行：

```python
# 修复前（错误：_hilbert_distances_between_consecutive 返回 List[float]，不是三元组）
hilbert_between,_,_ = hilbert_computation._hilbert_distances_between_consecutive(ref_traj)

# 修复后
hilbert_between = hilbert_computation._hilbert_distances_between_consecutive(ref_traj)
```

**Bug 2（死代码 + 错误装饰器）** — 删除了原 278–335 行的三个模块级函数：

`@staticmethod` 在类外部使用时，会把函数包成 `staticmethod` 描述符对象（Python < 3.10 下不可直接调用）。此外 298 行的 `@staticmethod` 与下面的 `def` 之间有空行，语义不明确。三个函数本身都只是对 `hilbert_computation` 类方法的转发，而 `graph_print_analysis.py` 已经直接 import 类，从未使用这些包装器，故全部删除。

### `MNIST_models/Nureon Network MNIST/testtool.py`

**Bug 3（路径依赖 CWD）** — `sys.path` 注入改为相对于 `__file__`：

```python
# 修复前（CWD 不同时路径就错）
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root / "src"))   # cone_dynamics/src/ 不存在

# 修复后
project_root = Path(__file__).resolve().parents[2]   # 始终指向 cone_dynamics/
sys.path.insert(0, str(project_root))
```

**Bug 4（dict vs list API 不匹配）** — `graph_print_analysis` 更新后，`analysis_to_w_star` / `analysis_to_w_between` 均返回 dict，而 `testtool.py` 仍把返回值当 list 迭代（`len(hilbert_to_final)`、`hilbert_to_final[t]` 等）。修复：在调用处提取对应 key：

```python
# 无 mask 段（section 1）
hilbert_to_final = gp_tool.analysis_to_w_star(...).get("hilbert_to_w_star", [])
hilbert_between  = gp_tool.analysis_to_w_between(...).get("hilbert_between", [])

# Masked 段（section 5）
hilbert_to_final2 = gp_tool.analysis_to_w_star(...).get("hilbert_to_w_star_masked", [])
hilbert_between2  = gp_tool.analysis_to_w_between(...).get("hilbert_between_masked", [])
```

### `functions/load_my_file.py`

**Bug 5（Windows 绝对路径）** — `BASE_DIR` 改为相对于 `__file__`：

```python
# 修复前
BASE_DIR = Path(r"C:\Users\ASUS\Desktop\cone_dynamics\MNIST_models\analysis_resul")  # 路径有 typo

# 修复后
BASE_DIR = Path(__file__).resolve().parents[1] / "MNIST_models" / "Nureon Network MNIST" / "analysis_result"
```

### `MNIST_models/svm_MNIST/`

**Bug 6（文件名 typo）** — `evalutation.py` → `evaluation.py`（无其他文件 import 它，直接重命名）。

### `functions/__init__.py`（新建）

`graph_print_analysis.py` 使用相对 import（`from .Hilbert_computation import ...`），需要 `functions/` 是一个 Python package。新建 `__init__.py` 并导出常用符号：

```python
from .Hilbert_computation import hilbert_computation, distance_func
from .graph_print_analysis import plot_hb, write_stats, analysis_to_w_star, analysis_to_w_between, analysis
from .dataset import generate_swiss_roll
```

导入冒烟测试通过：`python -c "from functions import hilbert_computation, graph_print_analysis; print('OK')"` → `OK`。
