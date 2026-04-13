"""Tests for fairkl.sklearn_compat (FairKRREstimator)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from fairkl.sklearn_compat import FairKRREstimator


def _make_data(n=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype("float32")
    q = rng.standard_normal((n, 1)).astype("float32")
    y = (X[:, 0] + 2.0 * q.ravel() + 0.1 * rng.standard_normal(n)).astype("float32")
    return X, y, q


class TestFairKRREstimator:
    def test_get_set_params(self):
        est = FairKRREstimator(sigma=2.0, lam=0.1, mu=5.0)
        params = est.get_params()
        assert params["sigma"] == 2.0
        assert params["lam"] == 0.1
        assert params["mu"] == 5.0

        est.set_params(mu=10.0)
        assert est.mu == 10.0

    def test_fit_predict_shapes(self):
        X, y, _q = _make_data()
        est = FairKRREstimator(sigma=1.0, lam=0.01, mu=0.0)
        est.fit(X, y)
        y_pred = est.predict(X)
        assert y_pred.shape == (50,)
        assert isinstance(y_pred, np.ndarray)

    def test_fit_with_fairness(self):
        X, y, q = _make_data()
        est = FairKRREstimator(sigma=1.0, lam=0.01, mu=1.0, sigma_q=1.0, epochs=20)
        est.fit(X, y, q=q)
        y_pred = est.predict(X)
        assert y_pred.shape == (50,)

    def test_score(self):
        X, y, _q = _make_data()
        est = FairKRREstimator(sigma=1.0, lam=0.01, mu=0.0)
        est.fit(X, y)
        r2 = est.score(X, y)
        # Should fit better than a constant predictor
        assert r2 > 0.0

    def test_cross_val_score(self):
        X, y, _q = _make_data(n=80)
        est = FairKRREstimator(sigma=1.0, lam=0.01, mu=0.0)
        scores = cross_val_score(est, X, y, cv=3, scoring="neg_mean_squared_error")
        assert scores.shape == (3,)
        # All folds should produce finite scores
        assert np.all(np.isfinite(scores))

    def test_manual_kfold_with_q(self):
        X, y, q = _make_data(n=80)
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        mses = []
        for train_idx, test_idx in kf.split(X):
            est = FairKRREstimator(sigma=1.0, lam=0.01, mu=1.0, epochs=20)
            est.fit(X[train_idx], y[train_idx], q=q[train_idx])
            y_pred = est.predict(X[test_idx])
            mses.append(float(np.mean((y_pred - y[test_idx]) ** 2)))
        assert len(mses) == 3
        assert all(np.isfinite(m) for m in mses)
