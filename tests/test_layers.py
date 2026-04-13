"""Tests for fairkl.layers (kernel layers)."""

from __future__ import annotations

import numpy as np

from fairkl.layers.kernel_layers import (
    LinearKernelLayer,
    NystromLayer,
    RBFKernelLayer,
    RFFLayer,
    RKSLayer,
)


class TestRBFKernelLayer:
    def test_output_shape(self):
        layer = RBFKernelLayer(sigma_init=1.0)
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = np.array(layer(X))
        assert K.shape == (20, 20)

    def test_cross_output_shape(self):
        layer = RBFKernelLayer(sigma_init=1.0)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5)).astype("float32")
        Y = rng.standard_normal((10, 5)).astype("float32")
        K = np.array(layer(X, Y))
        assert K.shape == (20, 10)

    def test_has_trainable_variable(self):
        layer = RBFKernelLayer(sigma_init=1.0, trainable_sigma=True)
        X = np.random.default_rng(0).standard_normal((5, 3)).astype("float32")
        layer(X)  # build
        trainable_names = [w.name for w in layer.trainable_weights]
        assert any("log_sigma" in n for n in trainable_names)

    def test_non_trainable(self):
        layer = RBFKernelLayer(sigma_init=1.0, trainable_sigma=False)
        X = np.random.default_rng(0).standard_normal((5, 3)).astype("float32")
        layer(X)
        assert len(layer.trainable_weights) == 0


class TestLinearKernelLayer:
    def test_output_shape(self):
        layer = LinearKernelLayer()
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        K = np.array(layer(X))
        assert K.shape == (20, 20)


class TestNystromLayer:
    def test_output_shape(self):
        layer = NystromLayer(n_landmarks=10, sigma_init=1.0)
        X = np.random.default_rng(0).standard_normal((30, 5)).astype("float32")
        Z = np.array(layer(X))
        assert Z.shape == (30, 10)

    def test_approximate_kernel_shape(self):
        layer = NystromLayer(n_landmarks=10, sigma_init=1.0)
        X = np.random.default_rng(0).standard_normal((30, 5)).astype("float32")
        Z = np.array(layer(X))
        K_approx = Z @ Z.T
        assert K_approx.shape == (30, 30)


class TestRFFLayer:
    def test_output_shape(self):
        layer = RFFLayer(n_features=100, sigma_init=1.0, seed=42)
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        Z = np.array(layer(X))
        assert Z.shape == (20, 100)

    def test_deterministic_with_seed(self):
        X = np.random.default_rng(0).standard_normal((10, 3)).astype("float32")
        layer1 = RFFLayer(n_features=50, sigma_init=1.0, seed=42)
        layer2 = RFFLayer(n_features=50, sigma_init=1.0, seed=42)
        Z1 = np.array(layer1(X))
        Z2 = np.array(layer2(X))
        np.testing.assert_allclose(Z1, Z2, atol=1e-6)


class TestRKSLayer:
    def test_output_shape(self):
        layer = RKSLayer(n_features=100, sigma_init=1.0, seed=42)
        X = np.random.default_rng(0).standard_normal((20, 5)).astype("float32")
        Z = np.array(layer(X))
        assert Z.shape == (20, 100)
