"""Keras Layer subclasses for kernel computation.

Each layer wraps a Layer 0 primitive and adds optional trainable
parameters (bandwidths, landmarks).
"""

from __future__ import annotations

import math

import keras
import keras.ops as ops

from fairkl.kernels.exact import linear_kernel, rbf_kernel


# ---------------------------------------------------------------------------
# RBFKernelLayer
# ---------------------------------------------------------------------------


class RBFKernelLayer(keras.Layer):
    """RBF kernel layer with (optionally trainable) bandwidth.

    Args:
        sigma_init: Initial bandwidth.
        trainable_sigma: Whether ``log_sigma`` is trainable.
    """

    def __init__(
        self,
        sigma_init: float = 1.0,
        trainable_sigma: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_init = sigma_init
        self.trainable_sigma = trainable_sigma
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=keras.initializers.Constant(math.log(sigma_init)),
            trainable=trainable_sigma,
        )

    def call(self, X, Y=None):
        sigma = ops.exp(self.log_sigma)
        return rbf_kernel(X, Y, sigma=sigma)

    def get_config(self):
        config = super().get_config()
        config.update(
            sigma_init=self.sigma_init,
            trainable_sigma=self.trainable_sigma,
        )
        return config


# ---------------------------------------------------------------------------
# LinearKernelLayer
# ---------------------------------------------------------------------------


class LinearKernelLayer(keras.Layer):
    """Linear kernel layer. No trainable parameters."""

    def call(self, X, Y=None):
        return linear_kernel(X, Y)


# ---------------------------------------------------------------------------
# NystromLayer
# ---------------------------------------------------------------------------


class NystromLayer(keras.Layer):
    """Nystrom approximation layer.

    Returns a low-rank feature matrix of shape ``(n, n_landmarks)``.
    The approximate kernel is ``Z @ Z.T``.

    Args:
        n_landmarks: Number of landmark (inducing) points.
        sigma_init: Initial RBF bandwidth.
        trainable_sigma: Whether bandwidth is trainable.
        trainable_landmarks: Whether landmark positions are trainable.
    """

    def __init__(
        self,
        n_landmarks: int,
        sigma_init: float = 1.0,
        trainable_sigma: bool = True,
        trainable_landmarks: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_landmarks = n_landmarks
        self.sigma_init = sigma_init
        self.trainable_sigma = trainable_sigma
        self.trainable_landmarks = trainable_landmarks
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=keras.initializers.Constant(math.log(sigma_init)),
            trainable=trainable_sigma,
        )
        self._landmarks = None

    def build(self, input_shape):
        d = input_shape[-1]
        self._landmarks = self.add_weight(
            name="landmarks",
            shape=(self.n_landmarks, d),
            initializer="glorot_uniform",
            trainable=self.trainable_landmarks,
        )
        super().build(input_shape)

    def call(self, X):
        sigma = ops.exp(self.log_sigma)
        landmarks = self._landmarks
        K_xz = rbf_kernel(X, landmarks, sigma=sigma)  # (n, m)
        K_zz = rbf_kernel(landmarks, landmarks, sigma=sigma)  # (m, m)
        m = ops.shape(K_zz)[0]
        K_zz_reg = K_zz + 1e-6 * ops.eye(m)
        L = ops.cholesky(K_zz_reg)
        # Feature matrix: Z = K_xz @ L^{-T}  so that Z @ Z.T ≈ K
        # Solve L @ A = K_xz^T  =>  A = L^{-1} K_xz^T  =>  Z = A^T
        A = ops.solve(L, ops.transpose(K_xz))  # (m, n)
        return ops.transpose(A)  # (n, m)

    def get_config(self):
        config = super().get_config()
        config.update(
            n_landmarks=self.n_landmarks,
            sigma_init=self.sigma_init,
            trainable_sigma=self.trainable_sigma,
            trainable_landmarks=self.trainable_landmarks,
        )
        return config


# ---------------------------------------------------------------------------
# RFFLayer
# ---------------------------------------------------------------------------


class RFFLayer(keras.Layer):
    """Random Fourier Features projection layer.

    Returns a feature matrix of shape ``(n, n_features)`` whose inner
    products approximate the RBF kernel.

    Args:
        n_features: Output feature dimensionality.
        sigma_init: RBF bandwidth for frequency sampling.
        trainable_sigma: Whether bandwidth is trainable.
        seed: Random seed.
    """

    def __init__(
        self,
        n_features: int,
        sigma_init: float = 1.0,
        trainable_sigma: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.sigma_init = sigma_init
        self.trainable_sigma = trainable_sigma
        self.seed = seed
        self.log_sigma = self.add_weight(
            name="log_sigma",
            shape=(),
            initializer=keras.initializers.Constant(math.log(sigma_init)),
            trainable=trainable_sigma,
        )
        self._omega = None
        self._bias = None

    def build(self, input_shape):
        d = input_shape[-1]
        seed_gen = keras.random.SeedGenerator(self.seed)
        # Sample frequencies once at build time
        omega_init = keras.random.normal(shape=(d, self.n_features), seed=seed_gen)
        bias_init = keras.random.uniform(
            shape=(1, self.n_features),
            minval=0.0,
            maxval=2.0 * math.pi,
            seed=seed_gen,
        )
        self._omega = self.add_weight(
            name="omega",
            shape=(d, self.n_features),
            initializer=keras.initializers.Constant(omega_init),
            trainable=False,
        )
        self._bias = self.add_weight(
            name="bias",
            shape=(1, self.n_features),
            initializer=keras.initializers.Constant(bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, X):
        sigma = ops.exp(self.log_sigma)
        # Scale frequencies by 1/sigma
        omega_scaled = self._omega / sigma
        projection = ops.matmul(X, omega_scaled) + self._bias
        return ops.sqrt(2.0 / self.n_features) * ops.cos(projection)

    def get_config(self):
        config = super().get_config()
        config.update(
            n_features=self.n_features,
            sigma_init=self.sigma_init,
            trainable_sigma=self.trainable_sigma,
            seed=self.seed,
        )
        return config


# ---------------------------------------------------------------------------
# RKSLayer
# ---------------------------------------------------------------------------


class RKSLayer(keras.Layer):
    """Random Kitchen Sinks layer.

    Identical to ``RFFLayer`` with fixed bandwidth, following the original
    Rahimi & Recht (2007) formulation.

    Args:
        n_features: Output feature dimensionality.
        sigma_init: RBF bandwidth.
        seed: Random seed.
    """

    def __init__(
        self,
        n_features: int,
        sigma_init: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.sigma_init = sigma_init
        self.seed = seed
        self._omega = None
        self._bias = None

    def build(self, input_shape):
        d = input_shape[-1]
        seed_gen = keras.random.SeedGenerator(self.seed)
        omega_init = (
            keras.random.normal(shape=(d, self.n_features), seed=seed_gen)
            / self.sigma_init
        )
        bias_init = keras.random.uniform(
            shape=(1, self.n_features),
            minval=0.0,
            maxval=2.0 * math.pi,
            seed=seed_gen,
        )
        self._omega = self.add_weight(
            name="omega",
            shape=(d, self.n_features),
            initializer=keras.initializers.Constant(omega_init),
            trainable=False,
        )
        self._bias = self.add_weight(
            name="bias",
            shape=(1, self.n_features),
            initializer=keras.initializers.Constant(bias_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, X):
        projection = ops.matmul(X, self._omega) + self._bias
        return ops.sqrt(2.0 / self.n_features) * ops.cos(projection)

    def get_config(self):
        config = super().get_config()
        config.update(
            n_features=self.n_features,
            sigma_init=self.sigma_init,
            seed=self.seed,
        )
        return config
