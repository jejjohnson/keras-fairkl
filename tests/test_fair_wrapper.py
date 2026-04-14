"""Tests for fairkl.models.fair_wrapper.FairModelWrapper."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from fairkl.metrics.cka import CKALoss, cka_rbf
from fairkl.metrics.hsic import HSICLoss
from fairkl.models.fair_wrapper import FairModelWrapper


def _make_data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    # q is a single sensitive attribute correlated with one feature of X
    q = X[:, :1] + 0.2 * rng.standard_normal((n, 1)).astype("float32")
    w_true = rng.standard_normal((d, 1)).astype("float32")
    y = (X @ w_true + 0.1 * rng.standard_normal((n, 1))).astype("float32")
    return X, y.astype("float32"), q.astype("float32")


def _small_mlp(d: int = 4) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(d,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


class TestFairModelWrapper:
    def test_fit_runs_with_q(self):
        X, y, q = _make_data()
        model = FairModelWrapper(_small_mlp(), mu=0.5)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        history = model.fit(X, y, q=q, epochs=2, batch_size=64, verbose=0)
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2

    def test_fit_runs_without_q(self):
        X, y, _ = _make_data()
        model = FairModelWrapper(_small_mlp(), mu=0.5)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(X, y, epochs=1, batch_size=64, verbose=0)

    def test_predict_shape(self):
        X, y, q = _make_data(n=100)
        model = FairModelWrapper(_small_mlp(), mu=0.5)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, q=q, epochs=1, batch_size=64, verbose=0)
        preds = np.asarray(model.predict(X, verbose=0))
        assert preds.shape == (100, 1)

    def test_mu_zero_matches_base_model_shape(self):
        X, y, q = _make_data()
        model = FairModelWrapper(_small_mlp(), mu=0.0)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, q=q, epochs=1, batch_size=64, verbose=0)
        preds = np.asarray(model.predict(X, verbose=0))
        assert preds.shape == y.shape

    def test_penalty_reduces_cka(self):
        """With a q strongly correlated to X, mu>0 should reduce held-out CKA."""
        X, y, q = _make_data(n=400, seed=1)
        # Split
        Xtr, Xte = X[:300], X[300:]
        ytr, _yte = y[:300], y[300:]
        qtr, qte = q[:300], q[300:]

        keras.utils.set_random_seed(0)
        m0 = FairModelWrapper(_small_mlp(), mu=0.0)
        m0.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse")
        m0.fit(Xtr, ytr, q=qtr, epochs=30, batch_size=128, verbose=0)

        keras.utils.set_random_seed(0)
        m1 = FairModelWrapper(_small_mlp(), mu=5.0)
        m1.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse")
        m1.fit(Xtr, ytr, q=qtr, epochs=30, batch_size=128, verbose=0)

        p0 = np.asarray(m0.predict(Xte, verbose=0))
        p1 = np.asarray(m1.predict(Xte, verbose=0))
        cka0 = float(cka_rbf(p0, qte, sigma_f=1.0, sigma_q=1.0))
        cka1 = float(cka_rbf(p1, qte, sigma_f=1.0, sigma_q=1.0))
        assert cka1 < cka0, f"penalty failed to reduce CKA: {cka0=} {cka1=}"

    def test_accepts_custom_fairness_loss(self):
        X, y, q = _make_data()
        model = FairModelWrapper(
            _small_mlp(), mu=0.1, fairness_loss=HSICLoss(sigma_f=1.0, sigma_q=1.0)
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, q=q, epochs=1, batch_size=64, verbose=0)

    def test_rejects_negative_mu(self):
        with pytest.raises(ValueError):
            FairModelWrapper(_small_mlp(), mu=-0.1)

    def test_q_1d_is_expanded(self):
        X, y, q = _make_data()
        model = FairModelWrapper(_small_mlp(), mu=0.1)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, q=q[:, 0], epochs=1, batch_size=64, verbose=0)

    def test_get_config_roundtrips_hyperparameters(self):
        loss = CKALoss(sigma_f=0.7, sigma_q=1.2, debiased=True)
        model = FairModelWrapper(_small_mlp(), mu=0.3, fairness_loss=loss)
        config = model.get_config()
        assert config["mu"] == 0.3
        assert "base_model" in config
        assert "fairness_loss" in config
        # The fairness loss config preserves its own hyperparameters
        fl_cfg = config["fairness_loss"]["config"]
        assert fl_cfg["sigma_f"] == 0.7
        assert fl_cfg["sigma_q"] == 1.2
        assert fl_cfg["debiased"] is True

    def test_default_fairness_loss_is_cka(self):
        model = FairModelWrapper(_small_mlp(), mu=0.5)
        assert isinstance(model.fairness_loss, CKALoss)
