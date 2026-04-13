"""Tests for fairkl.kernels (exact and approximate)."""

from __future__ import annotations

import functools

import numpy as np

from fairkl.kernels.approximate import (
    nystrom_approximate,
    random_fourier_features,
    random_kitchen_sinks,
)
from fairkl.kernels.exact import linear_kernel, polynomial_kernel, rbf_kernel


# ---------------------------------------------------------------------------
# Exact kernels
# ---------------------------------------------------------------------------


class TestRBFKernel:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = rbf_kernel(X, sigma=1.0)
        assert np.array(K).shape == (20, 20)

    def test_cross_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5)).astype("float32")
        Y = rng.standard_normal((10, 5)).astype("float32")
        K = rbf_kernel(X, Y, sigma=1.0)
        assert np.array(K).shape == (20, 10)

    def test_symmetry(self):
        X = np.random.default_rng(0).standard_normal((15, 4)).astype("float32")
        K = np.array(rbf_kernel(X))
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_diagonal_is_one(self):
        X = np.random.default_rng(0).standard_normal((15, 4)).astype("float32")
        K = np.array(rbf_kernel(X))
        np.testing.assert_allclose(np.diag(K), np.ones(15), atol=1e-6)

    def test_values_between_zero_and_one(self):
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = np.array(rbf_kernel(X, sigma=1.0))
        assert np.all(K >= -1e-7)
        assert np.all(K <= 1.0 + 1e-7)

    def test_analytical_two_points(self):
        X = np.array([[0.0, 0.0]], dtype="float32")
        Y = np.array([[1.0, 0.0]], dtype="float32")
        K = np.array(rbf_kernel(X, Y, sigma=1.0))
        expected = np.exp(-0.5)  # ||[0,0]-[1,0]||^2 / (2*1^2) = 0.5
        np.testing.assert_allclose(K[0, 0], expected, atol=1e-6)


class TestLinearKernel:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = linear_kernel(X)
        assert np.array(K).shape == (20, 20)

    def test_matches_matmul(self):
        X = np.random.default_rng(0).standard_normal((10, 4)).astype("float32")
        K = np.array(linear_kernel(X))
        expected = X @ X.T
        np.testing.assert_allclose(K, expected, atol=1e-5)


class TestPolynomialKernel:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = polynomial_kernel(X, degree=2, coef0=1.0)
        assert np.array(K).shape == (20, 20)

    def test_known_value(self):
        X = np.array([[1.0, 2.0]], dtype="float32")
        Y = np.array([[3.0, 4.0]], dtype="float32")
        K = np.array(polynomial_kernel(X, Y, degree=2, coef0=1.0))
        # (1*3 + 2*4 + 1)^2 = (3+8+1)^2 = 144
        np.testing.assert_allclose(K[0, 0], 144.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Approximate kernels
# ---------------------------------------------------------------------------


class TestNystrom:
    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 5)).astype("float32")
        landmarks = X[:10]
        kernel_fn = functools.partial(rbf_kernel, sigma=1.0)
        K_approx = nystrom_approximate(X, landmarks, kernel_fn)
        assert np.array(K_approx).shape == (30, 30)

    def test_approximation_quality(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3)).astype("float32")
        landmarks = X[:20]  # many landmarks => good approx
        kernel_fn = functools.partial(rbf_kernel, sigma=1.0)
        K_exact = np.array(rbf_kernel(X, sigma=1.0))
        K_approx = np.array(nystrom_approximate(X, landmarks, kernel_fn))
        # Relative Frobenius error should be small with m=20 out of n=30
        rel_error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        assert rel_error < 0.3


class TestRFF:
    def test_shape(self):
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        Z = random_fourier_features(X, n_features=100, sigma=1.0, seed=42)
        assert np.array(Z).shape == (20, 100)

    def test_approximation_quality(self):
        X = np.random.default_rng(0).standard_normal((20, 3)).astype("float32")
        Z = np.array(random_fourier_features(X, n_features=2000, sigma=1.0, seed=42))
        K_approx = Z @ Z.T
        K_exact = np.array(rbf_kernel(X, sigma=1.0))
        rel_error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        assert rel_error < 0.2


class TestRKS:
    def test_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5)).astype("float32")
        W = rng.standard_normal((5, 100)).astype("float32")
        b = rng.uniform(0, 2 * np.pi, size=(1, 100)).astype("float32")
        Z = random_kitchen_sinks(X, W, b)
        assert np.array(Z).shape == (20, 100)
