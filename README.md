# Hilbert Metric on Gradient Flow

This repository implements the Hilbert metric on convex cones and applies it to gradient flow optimization algorithms. The Hilbert metric is a projective metric defined on the interior of a convex cone, providing a natural geometry for optimization problems constrained to such cones.

## Overview

The Hilbert metric between two points x and y in the interior of a convex cone K is defined as:

```
d_H(x, y) = log(M(x,y) / m(x,y))
```

where M(x,y) and m(x,y) are the maximum and minimum values such that:
```
m(x,y) * x ≤ y ≤ M(x,y) * x
```

This implementation provides:
- **HilbertMetric**: Computation of Hilbert distances and geodesics on convex cones
- **HilbertGradientFlow**: Gradient flow optimization using the Hilbert metric geometry

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Hilbert Metric Operations

```python
import numpy as np
from hilbert_metric import HilbertMetric

# Create a Hilbert metric instance
metric = HilbertMetric()

# Define points in the positive orthant
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 4.0])

# Compute Hilbert distance
distance = metric.distance(x, y)
print(f"Hilbert distance: {distance}")

# Compute a point on the geodesic
gamma_t = metric.geodesic(x, y, t=0.5)
print(f"Midpoint on geodesic: {gamma_t}")
```

### Gradient Flow Optimization

```python
import numpy as np
from hilbert_metric import HilbertGradientFlow

# Define objective function and gradient
target = np.array([2.0, 3.0, 1.5])

def objective(x):
    return np.sum((x - target)**2)

def gradient(x):
    return 2 * (x - target)

# Create optimizer
optimizer = HilbertGradientFlow(objective, gradient)

# Initial point
x0 = np.array([1.0, 1.0, 1.0])

# Optimize using Hilbert metric geometry
x_opt, obj_history, dist_history = optimizer.optimize(
    x0, 
    learning_rate=0.1,
    max_iterations=1000,
    use_hilbert_retraction=True,
    verbose=True
)

print(f"Optimal point: {x_opt}")
print(f"Final objective: {objective(x_opt)}")
```

## Examples

Run the included examples:

```bash
python hilbert_metric.py
```

This will demonstrate:
1. Properties of the Hilbert metric (symmetry, distances, geodesics)
2. Quadratic optimization comparing Hilbert retraction vs. standard gradient descent

## Running Tests

Run the test suite:

```bash
pytest test_hilbert_metric.py -v
```

## Theory

The Hilbert metric provides several advantages for optimization on convex cones:

1. **Projective Invariance**: The metric is invariant under projective transformations
2. **Natural Geometry**: Provides a natural Riemannian-like structure on the cone interior
3. **Convergence Properties**: Can improve convergence for certain classes of problems

The gradient flow in Hilbert geometry uses an exponential map retraction:
```
x_new = x ⊙ exp(-α ∇f(x) ⊘ x)
```
where ⊙ and ⊘ are element-wise multiplication and division.

## References

- Nussbaum, R. D. (1988). "Hilbert's projective metric and iterated nonlinear maps"
- Lemmens, B., & Nussbaum, R. (2012). "Nonlinear Perron-Frobenius Theory"
- Alvarez, F., Bolte, J., & Munier, J. (2002). "A unifying local convergence result for Newton's method in Riemannian manifolds"

## License

MIT License