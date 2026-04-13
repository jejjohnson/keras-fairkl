"""Tests for fairkl.metrics (HSIC, MMD, losses, metrics)."""

from __future__ import annotations

import numpy as np

from fairkl.kernels.exact import rbf_kernel
from fairkl.metrics.hsic import HSICLoss, HSICMetric, hsic_biased, hsic_linear, hsic_rbf
from fairkl.metrics.mmd import MMDLoss, MMDMetric, mmd_rbf


# ---------------------------------------------------------------------------
# HSIC pure functions
# ---------------------------------------------------------------------------


class TestHSICBiased:
    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3)).astype("float32")
        K_f = np.array(rbf_kernel(X, sigma=1.0))
        K_q = np.array(rbf_kernel(X, sigma=1.0))
        val = float(hsic_biased(K_f, K_q))
        assert val >= -1e-7

    def test_independent_low_hsic(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal((50, 1)).astype("float32")
        q = rng.standard_normal((50, 1)).astype("float32")
        K_f = np.array(rbf_kernel(f, sigma=1.0))
        K_q = np.array(rbf_kernel(q, sigma=1.0))
        val = float(hsic_biased(K_f, K_q))
        # Independent => HSIC should be small
        assert val < 0.1

    def test_correlated_higher_hsic(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal((50, 1)).astype("float32")
        q = f + 0.1 * rng.standard_normal((50, 1)).astype("float32")
        K_f = np.array(rbf_kernel(f, sigma=1.0))
        K_q = np.array(rbf_kernel(q, sigma=1.0))
        val_corr = float(hsic_biased(K_f, K_q))

        # Compare with independent
        q_ind = rng.standard_normal((50, 1)).astype("float32")
        K_q_ind = np.array(rbf_kernel(q_ind, sigma=1.0))
        val_ind = float(hsic_biased(K_f, K_q_ind))

        assert val_corr > val_ind


class TestHSICRBF:
    def test_scalar_output(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((20, 3)).astype("float32")
        q = rng.standard_normal((20, 1)).astype("float32")
        val = hsic_rbf(f, q, sigma_f=1.0, sigma_q=1.0)
        assert np.array(val).shape == ()


class TestHSICLinear:
    def test_scalar_output(self):
        rng = np.random.default_rng(0)
        f = rng.standard_normal((20, 3)).astype("float32")
        q = rng.standard_normal((20, 1)).astype("float32")
        val = hsic_linear(f, q)
        assert np.array(val).shape == ()


# ---------------------------------------------------------------------------
# MMD pure function
# ---------------------------------------------------------------------------


class TestMMDRBF:
    def test_zero_for_same_data(self):
        X = np.random.default_rng(0).standard_normal((30, 3)).astype("float32")
        val = float(mmd_rbf(X, X, sigma=1.0))
        np.testing.assert_allclose(val, 0.0, atol=1e-6)

    def test_positive_for_different_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 3)).astype("float32")
        Y = rng.standard_normal((30, 3)).astype("float32") + 3.0
        val = float(mmd_rbf(X, Y, sigma=1.0))
        assert val > 0.0


# ---------------------------------------------------------------------------
# Keras Loss classes
# ---------------------------------------------------------------------------


class TestHSICLoss:
    def test_call(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((20, 1)).astype("float32")
        y_true = rng.standard_normal((20, 1)).astype("float32")
        loss = HSICLoss(sigma_f=1.0, sigma_q=1.0)
        val = float(loss(y_true, y_pred))
        assert val >= -1e-7

    def test_linear_kernel(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((20, 1)).astype("float32")
        y_true = rng.standard_normal((20, 1)).astype("float32")
        loss = HSICLoss(kernel="linear")
        val = float(loss(y_true, y_pred))
        assert np.isfinite(val)


class TestMMDLoss:
    def test_call(self):
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((20, 1)).astype("float32")
        y_true = rng.standard_normal((20, 1)).astype("float32") + 2.0
        loss = MMDLoss(sigma=1.0)
        val = float(loss(y_true, y_pred))
        assert val > 0.0


# ---------------------------------------------------------------------------
# Keras Metric classes
# ---------------------------------------------------------------------------


class TestHSICMetric:
    def test_running_average(self):
        metric = HSICMetric(sigma_f=1.0, sigma_q=1.0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            y_pred = rng.standard_normal((15, 1)).astype("float32")
            y_true = rng.standard_normal((15, 1)).astype("float32")
            metric.update_state(y_true, y_pred)
        val = float(metric.result())
        assert np.isfinite(val)
        metric.reset_state()
        assert float(metric.result()) == 0.0


class TestMMDMetric:
    def test_running_average(self):
        metric = MMDMetric(sigma=1.0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            y_pred = rng.standard_normal((15, 1)).astype("float32")
            y_true = rng.standard_normal((15, 1)).astype("float32") + 1.0
            metric.update_state(y_true, y_pred)
        val = float(metric.result())
        assert val > 0.0
