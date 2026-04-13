"""Tests for fairkl.models (FairKernelRidge, FairLinear, FairPCA, FairKernelPCA)."""

from __future__ import annotations

import numpy as np

from fairkl.models.fair_kernel_pca import FairKernelPCA
from fairkl.models.fair_kernel_ridge import FairKernelRidge
from fairkl.models.fair_linear import FairLinear
from fairkl.models.fair_pca import FairPCA


# Shared synthetic data
def _make_data(n=50, d=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    w_true = rng.standard_normal(d).astype("float32")
    y = (X @ w_true + 0.1 * rng.standard_normal(n)).astype("float32")
    q = rng.standard_normal((n, 1)).astype("float32")
    return X, y, q


# ---------------------------------------------------------------------------
# FairKernelRidge
# ---------------------------------------------------------------------------


class TestFairKernelRidge:
    def test_fit_predict_shapes(self):
        X, y, _q = _make_data()
        model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
        model.fit(X, y)
        y_pred = np.array(model.predict(X))
        assert y_pred.shape == (50, 1)

    def test_fit_with_fairness(self):
        X, y, q = _make_data()
        model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=1.0, sigma_q=1.0)
        model.fit(X, y, q=q)
        y_pred = np.array(model.predict(X))
        assert y_pred.shape == (50, 1)

    def test_get_alpha(self):
        X, y, _q = _make_data()
        model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
        model.fit(X, y)
        alpha = np.array(model.get_alpha())
        assert alpha.shape == (50, 1)

    def test_get_kernel_matrix(self):
        X, y, _q = _make_data()
        model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
        model.fit(X, y)
        K = np.array(model.get_kernel_matrix(X[:10]))
        assert K.shape == (10, 50)

    def test_mu_zero_recovers_krr(self):
        X, y, _ = _make_data(n=30, d=3)
        # mu=0 should give standard KRR
        model = FairKernelRidge(sigma=1.0, lam=0.1, mu=0.0)
        model.fit(X, y)
        y_pred = np.array(model.predict(X)).ravel()
        # Should fit reasonably well
        mse = np.mean((y_pred - y) ** 2)
        assert mse < np.var(y)  # better than predicting the mean

    def test_linear_kernel(self):
        X, y, _q = _make_data()
        model = FairKernelRidge(lam=1e-2, mu=0.0, kernel="linear")
        model.fit(X, y)
        y_pred = np.array(model.predict(X))
        assert y_pred.shape == (50, 1)


# ---------------------------------------------------------------------------
# FairLinear
# ---------------------------------------------------------------------------


class TestFairLinear:
    def test_fit_predict_shapes(self):
        X, y, _q = _make_data()
        model = FairLinear(lam=1e-3, mu=0.0)
        model.fit(X, y, epochs=20, lr=1e-2)
        y_pred = np.array(model.predict(X))
        assert y_pred.shape == (50, 1)

    def test_fit_with_fairness(self):
        X, y, q = _make_data()
        model = FairLinear(lam=1e-3, mu=0.5, sigma_q=1.0)
        model.fit(X, y, q=q, epochs=20, lr=1e-2)
        y_pred = np.array(model.predict(X))
        assert y_pred.shape == (50, 1)

    def test_decision_function(self):
        X, y, _q = _make_data()
        model = FairLinear(lam=1e-3, mu=0.0)
        model.fit(X, y, epochs=10)
        scores = np.array(model.decision_function(X))
        y_pred = np.array(model.predict(X))
        np.testing.assert_allclose(scores, y_pred, atol=1e-6)


# ---------------------------------------------------------------------------
# FairPCA
# ---------------------------------------------------------------------------


class TestFairPCA:
    def test_transform_shape(self):
        X, _, _q = _make_data(n=40, d=8)
        model = FairPCA(n_components=3, mu=0.0)
        model.fit(X, epochs=50, lr=1e-2)
        Z = np.array(model.transform(X))
        assert Z.shape == (40, 3)

    def test_fit_with_fairness(self):
        X, _, q = _make_data(n=40, d=8)
        model = FairPCA(n_components=3, mu=1.0, sigma_q=1.0)
        model.fit(X, q=q, epochs=50, lr=1e-2)
        Z = np.array(model.transform(X))
        assert Z.shape == (40, 3)

    def test_inverse_transform(self):
        X, _, _q = _make_data(n=40, d=8)
        model = FairPCA(n_components=3, mu=0.0)
        model.fit(X, epochs=50)
        Z = model.transform(X)
        X_hat = np.array(model.inverse_transform(Z))
        assert X_hat.shape == (40, 8)

    def test_fit_transform(self):
        X, _, _q = _make_data(n=40, d=8)
        model = FairPCA(n_components=3, mu=0.0)
        Z = np.array(model.fit_transform(X, epochs=50))
        assert Z.shape == (40, 3)


# ---------------------------------------------------------------------------
# FairKernelPCA
# ---------------------------------------------------------------------------


class TestFairKernelPCA:
    def test_fit_transform_shape(self):
        X, _, _q = _make_data(n=30, d=5)
        model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.0)
        Z = np.array(model.fit_transform(X, epochs=30, lr=1e-2))
        assert Z.shape == (30, 3)

    def test_transform_new_data(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((30, 5)).astype("float32")
        X_test = rng.standard_normal((10, 5)).astype("float32")
        model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.0)
        model.fit(X_train, epochs=30, lr=1e-2)
        Z_test = np.array(model.transform(X_test))
        assert Z_test.shape == (10, 3)

    def test_inverse_transform(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((30, 5)).astype("float32")
        model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.0)
        model.fit(X_train, epochs=30, lr=1e-2)
        Z = np.array(model.transform(X_train))
        X_hat = np.array(model.inverse_transform(Z))
        assert X_hat.shape == (30, 5)

    def test_fit_with_fairness(self):
        X, _, q = _make_data(n=30, d=5)
        model = FairKernelPCA(n_components=3, sigma=1.0, mu=1.0, sigma_q=1.0)
        model.fit(X, q=q, epochs=30, lr=1e-2)
        Z = np.array(model.transform(X))
        assert Z.shape == (30, 3)
