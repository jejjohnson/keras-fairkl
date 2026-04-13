"""Tests for fairkl.ops (centering and solvers)."""

from __future__ import annotations

import numpy as np

from fairkl.ops.centering import center_kernel, centering_matrix
from fairkl.ops.solvers import solve_cg, solve_cholesky


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------


class TestCenteringMatrix:
    def test_shape(self):
        H = np.array(centering_matrix(10))
        assert H.shape == (10, 10)

    def test_row_sums_zero(self):
        H = np.array(centering_matrix(10))
        np.testing.assert_allclose(H.sum(axis=1), np.zeros(10), atol=1e-6)

    def test_idempotent(self):
        H = np.array(centering_matrix(8))
        H2 = H @ H
        np.testing.assert_allclose(H2, H, atol=1e-6)


class TestCenterKernel:
    def test_centered_has_zero_means(self):
        rng = np.random.default_rng(0)
        K = rng.standard_normal((15, 15)).astype("float32")
        K = K @ K.T  # make PSD
        Kc = np.array(center_kernel(K))
        np.testing.assert_allclose(Kc.mean(axis=0), np.zeros(15), atol=1e-5)
        np.testing.assert_allclose(Kc.mean(axis=1), np.zeros(15), atol=1e-5)

    def test_matches_explicit_H(self):
        rng = np.random.default_rng(0)
        K = rng.standard_normal((10, 10)).astype("float32")
        K = K @ K.T
        H = np.array(centering_matrix(10))
        expected = H @ K @ H
        result = np.array(center_kernel(K))
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


class TestSolveCholesky:
    def test_simple_system(self):
        rng = np.random.default_rng(0)
        A_half = rng.standard_normal((10, 10)).astype("float32")
        A = A_half @ A_half.T + 0.1 * np.eye(10, dtype="float32")
        b = rng.standard_normal(10).astype("float32")
        x = np.array(solve_cholesky(A, b))
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-3

    def test_multi_rhs(self):
        rng = np.random.default_rng(0)
        A_half = rng.standard_normal((8, 8)).astype("float32")
        A = A_half @ A_half.T + 0.1 * np.eye(8, dtype="float32")
        b = rng.standard_normal((8, 3)).astype("float32")
        x = np.array(solve_cholesky(A, b))
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-3


class TestSolveCG:
    def test_diagonal_system(self):
        A = np.diag(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32"))
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32")
        x = np.array(solve_cg(A, b, maxiter=50))
        expected = np.array([1.0, 0.5, 1 / 3, 0.25, 0.2], dtype="float32")
        np.testing.assert_allclose(x, expected, atol=1e-4)

    def test_spd_system(self):
        rng = np.random.default_rng(0)
        A_half = rng.standard_normal((10, 10)).astype("float32")
        A = A_half @ A_half.T + 0.5 * np.eye(10, dtype="float32")
        b = rng.standard_normal(10).astype("float32")
        x = np.array(solve_cg(A, b, maxiter=200))
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-3

    def test_callable_matvec(self):
        A = np.diag(np.array([2.0, 3.0, 4.0], dtype="float32"))
        b = np.array([2.0, 3.0, 4.0], dtype="float32")

        def matvec(v):
            return A @ v

        x = np.array(solve_cg(matvec, b, maxiter=50))
        np.testing.assert_allclose(x, np.ones(3), atol=1e-4)
