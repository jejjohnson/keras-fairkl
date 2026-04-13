"""Tests for fairkl.tuning (FairKernelRidgeHyperModel)."""

from __future__ import annotations

import keras_tuner as kt
import numpy as np

from fairkl.tuning import FairKernelRidgeHyperModel


def _make_data(n=40, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2)).astype("float32")
    q = rng.standard_normal((n, 1)).astype("float32")
    y = (X[:, 0] + 2.0 * q.ravel() + 0.1 * rng.standard_normal(n)).astype("float32")
    return X, y, q


class TestFairKernelRidgeHyperModel:
    def test_build_returns_model(self):
        X, y, q = _make_data()
        hm = FairKernelRidgeHyperModel(X_train=X, y_train=y, q_train=q)
        hp = kt.HyperParameters()
        model = hm.build(hp)
        # Should return a FairKernelRidge instance
        from fairkl.models.fair_kernel_ridge import FairKernelRidge

        assert isinstance(model, FairKernelRidge)

    def test_fit_returns_metrics_dict(self):
        X, y, q = _make_data()
        hm = FairKernelRidgeHyperModel(X_train=X, y_train=y, q_train=q)
        hp = kt.HyperParameters()
        model = hm.build(hp)
        result = hm.fit(hp, model)
        assert isinstance(result, dict)
        assert "val_mse" in result
        assert "val_cka" in result
        assert np.isfinite(result["val_mse"])
        assert np.isfinite(result["val_cka"])

    def test_fit_with_validation_data(self):
        X, y, q = _make_data(n=60)
        hm = FairKernelRidgeHyperModel(
            X_train=X[:40],
            y_train=y[:40],
            q_train=q[:40],
            X_val=X[40:],
            y_val=y[40:],
            q_val=q[40:],
        )
        hp = kt.HyperParameters()
        model = hm.build(hp)
        result = hm.fit(hp, model)
        assert np.isfinite(result["val_mse"])
        assert np.isfinite(result["val_cka"])

    def test_fit_without_q(self):
        X, y, _q = _make_data()
        hm = FairKernelRidgeHyperModel(X_train=X, y_train=y, q_train=None)
        hp = kt.HyperParameters()
        model = hm.build(hp)
        result = hm.fit(hp, model)
        assert result["val_cka"] == 0.0
        assert np.isfinite(result["val_mse"])

    def test_random_search_runs(self, tmp_path):
        X, y, q = _make_data()
        hm = FairKernelRidgeHyperModel(X_train=X, y_train=y, q_train=q)
        tuner = kt.RandomSearch(
            hm,
            objective=kt.Objective("val_mse", direction="min"),
            max_trials=2,
            overwrite=True,
            directory=str(tmp_path),
            project_name="test_fair_krr",
        )
        tuner.search()
        best_hp = tuner.get_best_hyperparameters(1)
        assert len(best_hp) == 1
        assert "sigma" in best_hp[0].values
