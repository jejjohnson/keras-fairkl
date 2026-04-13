"""Tests for fairkl.metrics.cka (CKA and debiased CKA)."""

from __future__ import annotations

import numpy as np

from fairkl.kernels.exact import rbf_kernel
from fairkl.metrics.cka import (
    CKALoss,
    CKAMetric,
    center_gram_unbiased,
    cka_biased,
    cka_debiased,
    cka_linear,
    cka_rbf,
)


# ---------------------------------------------------------------------------
# center_gram_unbiased
# ---------------------------------------------------------------------------


class TestCenterGramUnbiased:
    def test_zero_diagonal(self):
        rng = np.random.default_rng(0)
        K = rng.standard_normal((15, 15)).astype("float32")
        K = K @ K.T
        Kc = np.array(center_gram_unbiased(K))
        np.testing.assert_allclose(np.diag(Kc), np.zeros(15), atol=1e-6)

    def test_shape(self):
        K = np.eye(10, dtype="float32")
        Kc = np.array(center_gram_unbiased(K))
        assert Kc.shape == (10, 10)


# ---------------------------------------------------------------------------
# cka_biased
# ---------------------------------------------------------------------------


class TestCKABiased:
    def test_identical_kernels_equals_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 5)).astype("float32")
        K = np.array(rbf_kernel(X, sigma=1.0))
        val = float(cka_biased(K, K))
        np.testing.assert_allclose(val, 1.0, atol=1e-5)

    def test_bounded_zero_one(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 5)).astype("float32")
        Y = rng.standard_normal((30, 3)).astype("float32")
        K_x = np.array(rbf_kernel(X, sigma=1.0))
        K_y = np.array(rbf_kernel(Y, sigma=1.0))
        val = float(cka_biased(K_x, K_y))
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_independent_data_low_cka(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 1)).astype("float32")
        Y = rng.standard_normal((50, 1)).astype("float32")
        K_x = np.array(rbf_kernel(X, sigma=1.0))
        K_y = np.array(rbf_kernel(Y, sigma=1.0))
        val = float(cka_biased(K_x, K_y))
        assert val < 0.3


# ---------------------------------------------------------------------------
# cka_debiased
# ---------------------------------------------------------------------------


class TestCKADebiased:
    def test_random_data_near_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 1)).astype("float32")
        Y = rng.standard_normal((50, 1)).astype("float32")
        K_x = np.array(rbf_kernel(X, sigma=1.0))
        K_y = np.array(rbf_kernel(Y, sigma=1.0))
        val = float(cka_debiased(K_x, K_y))
        assert abs(val) < 0.3

    def test_correlated_higher_than_independent(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 1)).astype("float32")
        Y_corr = (X + 0.1 * rng.standard_normal((50, 1))).astype("float32")
        Y_indep = rng.standard_normal((50, 1)).astype("float32")
        K_x = np.array(rbf_kernel(X, sigma=1.0))
        K_corr = np.array(rbf_kernel(Y_corr, sigma=1.0))
        K_indep = np.array(rbf_kernel(Y_indep, sigma=1.0))
        val_corr = float(cka_debiased(K_x, K_corr))
        val_indep = float(cka_debiased(K_x, K_indep))
        assert val_corr > val_indep


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestCKARBF:
    def test_scalar_output(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((20, 3)).astype("float32")
        q = rng.standard_normal((20, 1)).astype("float32")
        val = cka_rbf(f, q, sigma_f=1.0, sigma_q=1.0)
        assert np.array(val).shape == ()

    def test_scale_invariance(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((30, 1)).astype("float32")
        q = rng.standard_normal((30, 1)).astype("float32")
        val1 = float(cka_rbf(f, q, sigma_f=1.0, sigma_q=1.0))
        val2 = float(cka_rbf(f * 100.0, q, sigma_f=100.0, sigma_q=1.0))
        np.testing.assert_allclose(val1, val2, atol=0.05)


class TestCKALinear:
    def test_scalar_output(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((20, 3)).astype("float32")
        q = rng.standard_normal((20, 1)).astype("float32")
        val = cka_linear(f, q)
        assert np.array(val).shape == ()

    def test_bounded(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((30, 3)).astype("float32")
        q = rng.standard_normal((30, 1)).astype("float32")
        val = float(cka_linear(f, q))
        assert 0.0 <= val <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Keras Loss
# ---------------------------------------------------------------------------


class TestCKALoss:
    def test_call_bounded(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((20, 1)).astype("float32")
        y_true = rng.standard_normal((20, 1)).astype("float32")
        loss = CKALoss(sigma_f=1.0, sigma_q=1.0)
        val = float(loss(y_true, y_pred))
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_linear_kernel(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((20, 1)).astype("float32")
        y_true = rng.standard_normal((20, 1)).astype("float32")
        loss = CKALoss(kernel="linear")
        val = float(loss(y_true, y_pred))
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Keras Metric
# ---------------------------------------------------------------------------


class TestCKAMetric:
    def test_running_cka(self):
        metric = CKAMetric(sigma_f=1.0, sigma_q=1.0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            y_pred = rng.standard_normal((15, 1)).astype("float32")
            y_true = rng.standard_normal((15, 1)).astype("float32")
            metric.update_state(y_true, y_pred)
        val = float(metric.result())
        assert np.isfinite(val)
        metric.reset_state()
        # After reset, hsic_cross=0, denom=0, result should handle gracefully
        assert np.isfinite(float(metric.result()))
