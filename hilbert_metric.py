"""
Hilbert Metric on Gradient Flow

This module implements the Hilbert metric on convex cones and applies it
to gradient flow optimization. The Hilbert metric is a projective metric
defined on the interior of a convex cone.

References:
- Nussbaum, R. D. (1988). "Hilbert's projective metric and iterated nonlinear maps"
- Lemmens, B., & Nussbaum, R. (2012). "Nonlinear Perron-Frobenius Theory"
"""

import numpy as np
from typing import Callable, Tuple, Optional


class HilbertMetric:
    """
    Implementation of the Hilbert metric on convex cones.
    
    The Hilbert metric between two points x, y in the interior of a convex cone K
    is defined as:
        d_H(x, y) = log(M(x,y) / m(x,y))
    
    where M(x,y) and m(x,y) are the maximum and minimum values such that:
        m(x,y) * x ≤ y ≤ M(x,y) * x
    in the cone ordering.
    """
    
    def __init__(self, cone_constraint: Optional[Callable] = None):
        """
        Initialize the Hilbert metric.
        
        Args:
            cone_constraint: Optional function to check if a point is in the cone.
                           If None, uses the positive orthant (x > 0 for all components).
        """
        self.cone_constraint = cone_constraint or self._positive_orthant
    
    @staticmethod
    def _positive_orthant(x: np.ndarray) -> bool:
        """Check if x is in the positive orthant (all components > 0)."""
        return np.all(x > 0)
    
    def distance(self, x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute the Hilbert metric distance between x and y.
        
        Args:
            x: First point in the cone (must be in interior)
            y: Second point in the cone (must be in interior)
            eps: Small value to avoid division by zero
            
        Returns:
            The Hilbert metric distance d_H(x, y)
            
        Raises:
            ValueError: If x or y is not in the cone interior
        """
        if not self.cone_constraint(x) or not self.cone_constraint(y):
            raise ValueError("Both points must be in the interior of the cone")
        
        # Compute ratios y_i / x_i for all components
        ratios = y / (x + eps)
        
        # M(x,y) is the maximum ratio, m(x,y) is the minimum ratio
        M = np.max(ratios)
        m = np.min(ratios)
        
        # Hilbert metric: log(M/m)
        if m <= eps:
            return np.inf
        
        return np.log(M / m)
    
    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute a point on the Hilbert metric geodesic between x and y.
        
        The geodesic in the Hilbert metric is given by:
            gamma(t) = ((1-t) * x^(-1) + t * y^(-1))^(-1)
        for t in [0, 1], in the positive orthant.
        
        Args:
            x: Starting point
            y: Ending point
            t: Parameter in [0, 1]
            
        Returns:
            Point on the geodesic at parameter t
        """
        if not (0 <= t <= 1):
            raise ValueError("Parameter t must be in [0, 1]")
        
        # For positive orthant, use harmonic mean interpolation
        # gamma(t) = 1 / ((1-t)/x + t/y)
        return 1.0 / ((1 - t) / x + t / y)


class HilbertGradientFlow:
    """
    Gradient flow using the Hilbert metric for optimization on convex cones.
    
    This class implements gradient descent on convex cones using the Hilbert
    metric to measure distances and define the gradient flow.
    """
    
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        cone_projection: Optional[Callable] = None
    ):
        """
        Initialize the Hilbert gradient flow optimizer.
        
        Args:
            objective: The objective function to minimize
            gradient: Gradient of the objective function
            cone_projection: Function to project points onto the cone interior.
                           If None, uses positive orthant projection.
        """
        self.objective = objective
        self.gradient = gradient
        self.cone_projection = cone_projection or self._project_to_positive_orthant
        self.hilbert_metric = HilbertMetric()
    
    @staticmethod
    def _project_to_positive_orthant(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Project a point to the interior of the positive orthant."""
        return np.maximum(x, eps)
    
    def step(
        self,
        x: np.ndarray,
        learning_rate: float,
        use_hilbert_retraction: bool = True
    ) -> np.ndarray:
        """
        Perform one step of gradient flow.
        
        Args:
            x: Current point
            learning_rate: Step size
            use_hilbert_retraction: If True, use Hilbert metric geodesic retraction.
                                   If False, use standard gradient step.
            
        Returns:
            Updated point after one gradient step
        """
        # Compute gradient at current point
        grad = self.gradient(x)
        
        if use_hilbert_retraction:
            # Use exponential map in Hilbert geometry
            # For positive orthant: exp_x(v) = x * exp(v/x)
            # where division and exp are element-wise
            direction = -learning_rate * grad
            x_new = x * np.exp(direction / x)
        else:
            # Standard gradient descent
            x_new = x - learning_rate * grad
        
        # Project back to cone interior
        x_new = self.cone_projection(x_new)
        
        return x_new
    
    def optimize(
        self,
        x0: np.ndarray,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        use_hilbert_retraction: bool = True,
        verbose: bool = False
    ) -> Tuple[np.ndarray, list, list]:
        """
        Optimize using gradient flow with Hilbert metric.
        
        Args:
            x0: Initial point
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (on gradient norm)
            use_hilbert_retraction: Whether to use Hilbert metric retraction
            verbose: Whether to print progress
            
        Returns:
            Tuple of (optimal_point, objective_history, distance_history)
        """
        x = self.cone_projection(x0.copy())
        obj_history = []
        dist_history = []
        
        for iteration in range(max_iterations):
            # Compute objective and gradient
            obj_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            obj_history.append(obj_val)
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: obj = {obj_val:.6f}, "
                      f"grad_norm = {grad_norm:.6f}")
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged after {iteration} iterations")
                break
            
            # Perform gradient step
            x_new = self.step(x, learning_rate, use_hilbert_retraction)
            
            # Compute Hilbert distance moved
            try:
                dist = self.hilbert_metric.distance(x, x_new)
                dist_history.append(dist)
            except ValueError:
                dist_history.append(np.nan)
            
            x = x_new
        
        return x, obj_history, dist_history


def example_quadratic_optimization():
    """
    Example: Minimize a quadratic function on the positive orthant
    using Hilbert metric gradient flow.
    """
    print("=" * 70)
    print("Example: Quadratic Optimization with Hilbert Metric Gradient Flow")
    print("=" * 70)
    
    # Define a simple quadratic objective: f(x) = sum((x_i - 2)^2)
    # Minimum is at x = [2, 2, ..., 2]
    target = np.array([2.0, 3.0, 1.5, 2.5])
    
    def objective(x):
        return np.sum((x - target) ** 2)
    
    def gradient(x):
        return 2 * (x - target)
    
    # Initialize optimizer
    optimizer = HilbertGradientFlow(objective, gradient)
    
    # Starting point
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    
    print(f"\nObjective: minimize sum((x_i - target_i)^2)")
    print(f"Target: {target}")
    print(f"Initial point: {x0}")
    print(f"Initial objective: {objective(x0):.6f}")
    
    # Optimize with Hilbert retraction
    print("\n--- Using Hilbert Metric Retraction ---")
    x_opt_hilbert, obj_hist_hilbert, dist_hist_hilbert = optimizer.optimize(
        x0, learning_rate=0.1, max_iterations=100, verbose=True,
        use_hilbert_retraction=True
    )
    
    print(f"\nOptimal point (Hilbert): {x_opt_hilbert}")
    print(f"Final objective: {objective(x_opt_hilbert):.6f}")
    print(f"Distance to target: {np.linalg.norm(x_opt_hilbert - target):.6f}")
    
    # Optimize with standard gradient descent
    print("\n--- Using Standard Gradient Descent ---")
    x_opt_standard, obj_hist_standard, dist_hist_standard = optimizer.optimize(
        x0, learning_rate=0.1, max_iterations=100, verbose=True,
        use_hilbert_retraction=False
    )
    
    print(f"\nOptimal point (Standard): {x_opt_standard}")
    print(f"Final objective: {objective(x_opt_standard):.6f}")
    print(f"Distance to target: {np.linalg.norm(x_opt_standard - target):.6f}")
    
    # Compare convergence
    print("\n--- Convergence Comparison ---")
    print(f"Hilbert method iterations: {len(obj_hist_hilbert)}")
    print(f"Standard method iterations: {len(obj_hist_standard)}")
    print(f"Hilbert final objective: {obj_hist_hilbert[-1]:.8f}")
    print(f"Standard final objective: {obj_hist_standard[-1]:.8f}")


def example_hilbert_metric_properties():
    """
    Demonstrate properties of the Hilbert metric.
    """
    print("\n" + "=" * 70)
    print("Example: Properties of Hilbert Metric")
    print("=" * 70)
    
    metric = HilbertMetric()
    
    # Test points in positive orthant
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    z = np.array([3.0, 4.0, 5.0])
    
    print(f"\nPoints:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    
    # Compute distances
    d_xy = metric.distance(x, y)
    d_xz = metric.distance(x, z)
    d_yz = metric.distance(y, z)
    
    print(f"\nDistances:")
    print(f"d_H(x, y) = {d_xy:.6f}")
    print(f"d_H(x, z) = {d_xz:.6f}")
    print(f"d_H(y, z) = {d_yz:.6f}")
    
    # Test symmetry: d(x, y) = d(y, x)
    d_yx = metric.distance(y, x)
    print(f"\nSymmetry test: d_H(y, x) = {d_yx:.6f}")
    print(f"Symmetric: {np.isclose(d_xy, d_yx)}")
    
    # Test triangle inequality (may not hold exactly, but approximately)
    print(f"\nTriangle inequality: d_H(x,z) <= d_H(x,y) + d_H(y,z)?")
    print(f"{d_xz:.6f} <= {d_xy:.6f} + {d_yz:.6f} = {d_xy + d_yz:.6f}")
    print(f"Satisfied: {d_xz <= d_xy + d_yz + 1e-6}")
    
    # Test geodesic
    print(f"\nGeodesic between x and y:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gamma_t = metric.geodesic(x, y, t)
        print(f"  t = {t:.2f}: gamma(t) = {gamma_t}")


if __name__ == "__main__":
    # Run examples
    example_hilbert_metric_properties()
    example_quadratic_optimization()
