# MNIST SVM Hilbert-metric experiments

This module packages the three-layer experiment stack requested in the prompt:

1. **Baseline linear SVM** on flattened MNIST pixels.
2. **Random-projection + linear SVM** for embedding-dimension stress tests.
3. **Kernel SVM** (RBF/poly) inspected through dual variables.

## Running the pipelines

```bash
python -m MNIST_models.svm_MNIST.experiments
```

The module uses a lightweight smoke-test configuration so it will download
MNIST, run two short linear trainings, a projected variant, and two kernel
solves. The output includes endpoint singular values for quick sanity checks.

## Key entry points

* `data.load_binary_mnist` – deterministic 3-vs-8 split with ±1 labels.
* `linear_svm.train_linear_svm` – SGD training loop that records the full
  parameter trajectory `(w, b)`.
* `geometry.trajectory_geometry` – Hilbert/Euclidean/cosine traces relative to
  the final iterate plus inter-step Hilbert distances.
* `geometry.endpoint_svd` and `geometry.hilbert_distance_matrix` – tools for
  multi-run spectral gaps and cluster structure.
* `projections.make_random_projection` – Gaussian projection helper for
  expanded (or reduced) feature spaces.
* `kernel_svm.train_kernel_svm_runs` – kernel SVM solver that reconstructs the
  non-negative dual variables `alpha` for Hilbert analysis in the RKHS setting.
* `experiments.run_*` – convenience wrappers assembling the above pieces for
  baseline, projection, and kernel sweeps.
