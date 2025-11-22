"""
Tests for Hilbert Metric on Gradient Flow implementation.
"""

import numpy as np
import pytest
from hilbert_metric import HilbertMetric, HilbertGradientFlow


class TestHilbertMetric:
    """Test cases for HilbertMetric class."""
    
    def test_positive_orthant_check(self):
        """Test that positive orthant constraint works correctly."""
        metric = HilbertMetric()
        
        # Valid points
        assert metric.cone_constraint(np.array([1.0, 2.0, 3.0]))
        assert metric.cone_constraint(np.array([0.1, 0.1, 0.1]))
        
        # Invalid points
        assert not metric.cone_constraint(np.array([1.0, -1.0, 3.0]))
        assert not metric.cone_constraint(np.array([0.0, 1.0, 1.0]))
        assert not metric.cone_constraint(np.array([-1.0, -1.0, -1.0]))
    
    def test_distance_symmetry(self):
        """Test that Hilbert distance is symmetric."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        d_xy = metric.distance(x, y)
        d_yx = metric.distance(y, x)
        
        assert np.isclose(d_xy, d_yx, rtol=1e-6)
    
    def test_distance_zero_for_same_point(self):
        """Test that distance from a point to itself is zero."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        d = metric.distance(x, x)
        
        assert np.isclose(d, 0.0, atol=1e-6)
    
    def test_distance_positive(self):
        """Test that distance between different points is positive."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        d = metric.distance(x, y)
        assert d > 0
    
    def test_distance_scaling_invariance(self):
        """Test that Hilbert distance is scale-invariant in the cone."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        
        # y = 2*x, so they are on the same ray from origin
        # Distance should be 0 for points on the same ray
        d = metric.distance(x, y)
        
        # For points on the same ray, max_ratio = min_ratio = 2
        # So log(2/2) = 0
        assert np.isclose(d, 0.0, atol=1e-6)
    
    def test_distance_invalid_points(self):
        """Test that distance raises error for invalid points."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([-1.0, 2.0, 3.0])  # Invalid: has negative component
        
        with pytest.raises(ValueError):
            metric.distance(x, y)
    
    def test_geodesic_endpoints(self):
        """Test that geodesic returns correct endpoints."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        # At t=0, should get x
        gamma_0 = metric.geodesic(x, y, 0.0)
        assert np.allclose(gamma_0, x)
        
        # At t=1, should get y
        gamma_1 = metric.geodesic(x, y, 1.0)
        assert np.allclose(gamma_1, y)
    
    def test_geodesic_midpoint(self):
        """Test that geodesic midpoint is in the cone."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        gamma_mid = metric.geodesic(x, y, 0.5)
        
        # Check that midpoint is in positive orthant
        assert np.all(gamma_mid > 0)
        
        # For harmonic mean: 1/gamma = 0.5/x + 0.5/y
        # So gamma = 2*x*y/(x+y)
        expected = 2 * x * y / (x + y)
        assert np.allclose(gamma_mid, expected)
    
    def test_geodesic_invalid_parameter(self):
        """Test that geodesic raises error for invalid t."""
        metric = HilbertMetric()
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError):
            metric.geodesic(x, y, -0.1)
        
        with pytest.raises(ValueError):
            metric.geodesic(x, y, 1.1)


class TestHilbertGradientFlow:
    """Test cases for HilbertGradientFlow class."""
    
    def test_projection_to_positive_orthant(self):
        """Test that projection moves points to positive orthant."""
        def objective(x):
            return np.sum(x**2)
        
        def gradient(x):
            return 2 * x
        
        optimizer = HilbertGradientFlow(objective, gradient)
        
        # Test projection of negative points
        x_neg = np.array([-1.0, 2.0, -3.0])
        x_proj = optimizer.cone_projection(x_neg)
        
        assert np.all(x_proj > 0)
    
    def test_gradient_step_reduces_objective(self):
        """Test that gradient step reduces objective value."""
        # Simple convex objective
        target = np.array([2.0, 3.0, 1.5])
        
        def objective(x):
            return np.sum((x - target)**2)
        
        def gradient(x):
            return 2 * (x - target)
        
        optimizer = HilbertGradientFlow(objective, gradient)
        
        x0 = np.array([1.0, 1.0, 1.0])
        obj_before = objective(x0)
        
        # Take a step
        x1 = optimizer.step(x0, learning_rate=0.1, use_hilbert_retraction=False)
        obj_after = objective(x1)
        
        # For convex function with appropriate step size, objective should decrease
        assert obj_after < obj_before
    
    def test_optimize_converges(self):
        """Test that optimization converges to a solution."""
        # Simple quadratic with known minimum
        target = np.array([2.0, 3.0])
        
        def objective(x):
            return np.sum((x - target)**2)
        
        def gradient(x):
            return 2 * (x - target)
        
        optimizer = HilbertGradientFlow(objective, gradient)
        
        x0 = np.array([1.0, 1.0])
        x_opt, obj_hist, dist_hist = optimizer.optimize(
            x0, learning_rate=0.1, max_iterations=1000, tolerance=1e-4
        )
        
        # Check that we're close to the target
        assert np.linalg.norm(x_opt - target) < 0.1
        
        # Check that objective decreased
        assert obj_hist[-1] < obj_hist[0]
    
    def test_hilbert_vs_standard_retraction(self):
        """Test that both retraction methods work."""
        target = np.array([2.0, 3.0])
        
        def objective(x):
            return np.sum((x - target)**2)
        
        def gradient(x):
            return 2 * (x - target)
        
        optimizer = HilbertGradientFlow(objective, gradient)
        x0 = np.array([1.0, 1.0])
        
        # Test with Hilbert retraction
        x_hilbert, _, _ = optimizer.optimize(
            x0, learning_rate=0.05, max_iterations=100,
            use_hilbert_retraction=True
        )
        
        # Test with standard retraction
        x_standard, _, _ = optimizer.optimize(
            x0, learning_rate=0.05, max_iterations=100,
            use_hilbert_retraction=False
        )
        
        # Both should converge to reasonable solutions
        assert np.linalg.norm(x_hilbert - target) < 0.5
        assert np.linalg.norm(x_standard - target) < 0.5
    
    def test_optimize_returns_correct_types(self):
        """Test that optimize returns the correct types."""
        def objective(x):
            return np.sum(x**2)
        
        def gradient(x):
            return 2 * x
        
        optimizer = HilbertGradientFlow(objective, gradient)
        x0 = np.array([1.0, 1.0])
        
        x_opt, obj_hist, dist_hist = optimizer.optimize(
            x0, max_iterations=10
        )
        
        assert isinstance(x_opt, np.ndarray)
        assert isinstance(obj_hist, list)
        assert isinstance(dist_hist, list)
        assert len(obj_hist) > 0
        assert len(dist_hist) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
