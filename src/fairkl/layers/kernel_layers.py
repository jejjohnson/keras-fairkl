"""Keras Layer subclasses for kernel computation.

Each layer wraps a Layer 0 kernel primitive (from ``fairkl.kernels.exact``)
and adds optional trainable parameters (bandwidths, landmarks, random
frequencies).  All layers accept 2-D input tensors of shape ``(n, d)`` and
return either a Gram matrix of shape ``(n, n)`` (exact layers) or a
low-rank feature matrix of shape ``(n, n_features)`` (approximate layers).

Exact layers:
    * :class:`RBFKernelLayer` -- trainable Gaussian RBF kernel
    * :class:`LinearKernelLayer` -- parameterless linear kernel

Approximate layers:
    * :class:`NystromLayer` -- landmark-based Nystrom features
    * :class:`RFFLayer` -- Random Fourier Features with optional trainable sigma
    * :class:`RKSLayer` -- Random Kitchen Sinks (fixed-bandwidth RFF)
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
    """Gaussian RBF kernel layer with (optionally trainable) bandwidth.

    Computes the Gram matrix ``K`` where
    ``K[i, j] = exp(-||X[i] - Y[j]||^2 / (2 * sigma^2))``.  The bandwidth
    ``sigma`` is stored internally as ``log_sigma`` (in log-space) so that
    the parameter is unconstrained during gradient-based optimization while
    guaranteeing ``sigma > 0`` at all times.  This reparameterization also
    produces smoother gradients compared to optimizing ``sigma`` directly.

    When ``trainable_sigma=True`` (the default), the layer learns an
    optimal bandwidth end-to-end via back-propagation.

    Args:
        sigma_init (float): Initial bandwidth value.  Must be positive.
            The weight ``log_sigma`` is initialized to ``log(sigma_init)``.
            Defaults to ``1.0``.
        trainable_sigma (bool): If ``True``, ``log_sigma`` is a trainable
            weight updated by the optimizer.  If ``False``, the bandwidth
            is fixed for the lifetime of the layer.  Defaults to ``True``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.Layer.__init__`` (e.g., ``name``, ``dtype``).

    Examples:
        >>> layer = RBFKernelLayer(sigma_init=0.5, trainable_sigma=True)
        >>> X = keras.random.normal(shape=(32, 4), seed=0)
        >>> K = layer(X)           # shape (32, 32)
        >>> K.shape
        (32, 32)

    Note:
        Computational complexity is ``O(n * m * d)`` where *n* and *m* are
        the number of rows in *X* and *Y* respectively, and *d* is the
        feature dimensionality.  The full Gram matrix is materialized in
        memory, so this layer is best suited for datasets with ``n < ~10k``.

    See Also:
        :func:`fairkl.kernels.exact.rbf_kernel`: Stateless primitive used
            internally.
        :class:`RFFLayer`: Random feature approximation of the RBF kernel
            for large-scale problems.
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
        """Compute the RBF Gram matrix.

        Args:
            X: Input tensor of shape ``(n, d)`` with dtype ``float32``
                or ``float64``.
            Y: Optional second input tensor of shape ``(m, d)``.
                If ``None``, computes the self-kernel ``K(X, X)``.

        Returns:
            Gram matrix of shape ``(n, m)`` (or ``(n, n)`` when
            ``Y is None``), same dtype as *X*.
        """
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
    """Linear (dot-product) kernel layer with no trainable parameters.

    Computes the Gram matrix ``K = X @ Y.T``.  This is the simplest
    kernel, equivalent to operating in the original feature space without
    any non-linear mapping.  Because there are no learnable weights, the
    layer is fully stateless and can be shared across multiple call sites
    without side-effects.

    Examples:
        >>> layer = LinearKernelLayer()
        >>> X = keras.random.normal(shape=(16, 5), seed=0)
        >>> K = layer(X)           # shape (16, 16)
        >>> K.shape
        (16, 16)

    Note:
        Computational complexity is ``O(n * m * d)``.  The resulting Gram
        matrix is positive semi-definite but not strictly positive definite
        unless the input rows are linearly independent.

    See Also:
        :func:`fairkl.kernels.exact.linear_kernel`: Stateless primitive
            used internally.
        :class:`RBFKernelLayer`: Non-linear alternative with a trainable
            bandwidth.
    """

    def call(self, X, Y=None):
        """Compute the linear Gram matrix.

        Args:
            X: Input tensor of shape ``(n, d)`` with dtype ``float32``
                or ``float64``.
            Y: Optional second input tensor of shape ``(m, d)``.
                If ``None``, computes the self-kernel ``K(X, X)``.

        Returns:
            Gram matrix of shape ``(n, m)`` (or ``(n, n)`` when
            ``Y is None``), same dtype as *X*.
        """
        return linear_kernel(X, Y)


# ---------------------------------------------------------------------------
# NystromLayer
# ---------------------------------------------------------------------------


class NystromLayer(keras.Layer):
    """Nystrom low-rank kernel approximation layer.

    Produces a feature matrix ``Z`` of shape ``(n, n_landmarks)`` such that
    ``Z @ Z.T`` approximates the full RBF Gram matrix ``K``.  The
    approximation is constructed from a small set of *landmark* (inducing)
    points: the cross-kernel ``K_xz`` between the input and the landmarks
    is projected through the inverse square-root of the landmark-landmark
    kernel ``K_zz`` via a Cholesky solve::

        L = cholesky(K_zz + 1e-6 * I)
        Z = (L^{-1} @ K_xz^T)^T

    This yields a rank-``n_landmarks`` approximation with
    ``O(n * m * d + m^3)`` time complexity (where ``m = n_landmarks``),
    making it suitable when ``n`` is large but ``m`` is moderate
    (hundreds to low thousands).

    Args:
        n_landmarks (int): Number of landmark (inducing) points.  A larger
            value improves approximation quality at the cost of a bigger
            ``(n_landmarks, n_landmarks)`` Cholesky factorization.
        sigma_init (float): Initial RBF bandwidth.  Stored in log-space
            for unconstrained optimization.  Defaults to ``1.0``.
        trainable_sigma (bool): Whether the log-bandwidth weight is
            trainable.  Defaults to ``True``.
        trainable_landmarks (bool): Whether landmark positions are
            optimized during training.  When ``False`` (the default),
            landmarks are initialized via Glorot uniform and held fixed.
            Set to ``True`` to learn landmark locations end-to-end (useful
            for small landmark budgets).
        **kwargs: Additional keyword arguments forwarded to
            ``keras.Layer.__init__``.

    Examples:
        >>> layer = NystromLayer(n_landmarks=50, sigma_init=1.0)
        >>> X = keras.random.normal(shape=(200, 10), seed=0)
        >>> Z = layer(X)          # shape (200, 50)
        >>> K_approx = Z @ Z.T   # approximates the full 200x200 Gram matrix

    Note:
        A small diagonal jitter of ``1e-6`` is added to ``K_zz`` before
        the Cholesky factorization to ensure numerical stability.  The
        Cholesky solve dominates cost at ``O(m^3)``; for very large
        ``n_landmarks`` consider :class:`RFFLayer` instead.

    References:
        Williams, C. K. I. & Seeger, M. (2001). "Using the Nystrom method
        to speed up kernel machines." *Advances in Neural Information
        Processing Systems 13* (NIPS 2000), pp. 682--688.

    See Also:
        :class:`RBFKernelLayer`: Exact RBF Gram matrix (no approximation).
        :class:`RFFLayer`: Random Fourier Feature approximation as an
            alternative low-rank strategy.
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
        """Create the landmark weight from the input feature dimension.

        Args:
            input_shape: Shape tuple ``(n, d)`` of the expected input.
                The last element ``d`` determines the landmark
                dimensionality.
        """
        d = input_shape[-1]
        self._landmarks = self.add_weight(
            name="landmarks",
            shape=(self.n_landmarks, d),
            initializer="glorot_uniform",
            trainable=self.trainable_landmarks,
        )
        super().build(input_shape)

    def call(self, X):
        """Compute the Nystrom feature matrix.

        Args:
            X: Input tensor of shape ``(n, d)`` with dtype ``float32``
                or ``float64``.

        Returns:
            Feature matrix ``Z`` of shape ``(n, n_landmarks)`` such that
            ``Z @ Z.T`` approximates the full ``(n, n)`` RBF Gram matrix.
            Same dtype as *X*.
        """
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
    """Random Fourier Features (RFF) projection layer.

    Maps inputs to a randomized feature space of dimensionality
    ``n_features`` such that inner products in the feature space
    approximate the RBF kernel:

    .. math::

        Z(x) = \\sqrt{\\frac{2}{D}} \\cos\\!\\bigl(\\omega^\\top x / \\sigma + b\\bigr)

    where ``omega ~ N(0, I)`` and ``b ~ Uniform(0, 2*pi)`` are sampled
    once at layer build time and held fixed thereafter.  The bandwidth
    ``sigma`` is stored in log-space and can optionally be trained.

    The approximation satisfies ``E[Z(x)^T Z(y)] = k(x, y)`` where
    ``k`` is the RBF kernel with bandwidth ``sigma``.  Accuracy improves
    as ``n_features`` grows, with an ``O(1/sqrt(n_features))``
    approximation error rate.

    Args:
        n_features (int): Dimensionality of the random feature output.
            Larger values give a better kernel approximation at the cost
            of increased memory and compute.
        sigma_init (float): Initial RBF bandwidth used to scale the
            sampled frequencies.  Stored as ``log_sigma``.
            Defaults to ``1.0``.
        trainable_sigma (bool): Whether to optimize the bandwidth during
            training.  Defaults to ``False`` (fixed bandwidth), matching
            the original RFF formulation.
        seed (int | None): Random seed for reproducible frequency and bias
            sampling.  Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.Layer.__init__``.

    Examples:
        >>> layer = RFFLayer(n_features=128, sigma_init=1.0, seed=42)
        >>> X = keras.random.normal(shape=(64, 10), seed=0)
        >>> Z = layer(X)              # shape (64, 128)
        >>> K_approx = Z @ Z.T       # approximates 64x64 RBF Gram matrix

    Note:
        Computational complexity of the forward pass is ``O(n * d * D)``
        where ``D = n_features``.  Unlike :class:`NystromLayer`, there is
        no cubic Cholesky cost, making RFF preferable for very large
        landmark counts.  The frequencies ``omega`` and biases ``b`` are
        non-trainable weights to ensure the random projection is
        deterministic across calls.

    References:
        Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale
        Kernel Machines." *Advances in Neural Information Processing
        Systems 20* (NIPS 2007), pp. 1177--1184.

    See Also:
        :class:`RKSLayer`: Fixed-bandwidth variant (no ``trainable_sigma``
            option) following the original Rahimi & Recht formulation.
        :class:`NystromLayer`: Alternative low-rank approximation based
            on landmark points.
        :class:`RBFKernelLayer`: Exact RBF Gram matrix computation.
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
        """Sample random frequencies and biases, stored as fixed weights.

        Args:
            input_shape: Shape tuple ``(n, d)`` of the expected input.
                The last element ``d`` determines the frequency matrix
                shape ``(d, n_features)``.
        """
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
        """Project inputs into the random Fourier feature space.

        Applies the mapping
        ``Z = sqrt(2/D) * cos(X @ (omega / sigma) + b)``
        where ``D = n_features``.

        Args:
            X: Input tensor of shape ``(n, d)`` with dtype ``float32``
                or ``float64``.

        Returns:
            Feature matrix ``Z`` of shape ``(n, n_features)`` such that
            ``Z @ Z.T`` approximates the ``(n, n)`` RBF Gram matrix.
            Same dtype as *X*.
        """
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
    """Random Kitchen Sinks (RKS) layer with fixed bandwidth.

    A simplified variant of :class:`RFFLayer` where the RBF bandwidth is
    always fixed (never trainable).  The frequency matrix ``omega`` is
    drawn from ``N(0, 1/sigma^2 * I)`` at build time and baked into a
    non-trainable weight, so no per-forward-pass rescaling is needed.

    This matches the original "Random Kitchen Sinks" formulation by
    Rahimi & Recht (2007), where all randomness is resolved once and
    the resulting feature map is used as a drop-in replacement for an
    explicit kernel.

    The mapping is identical to :class:`RFFLayer` with
    ``trainable_sigma=False``, but ``RKSLayer`` pre-scales the
    frequencies at construction time, avoiding a division at every
    forward call.

    Args:
        n_features (int): Dimensionality of the random feature output.
            Larger values give a better kernel approximation.
        sigma_init (float): RBF bandwidth.  Used to scale the sampled
            frequencies at build time as ``omega / sigma``.  This value
            is fixed for the lifetime of the layer.  Defaults to ``1.0``.
        seed (int | None): Random seed for reproducible frequency and
            bias sampling.  Defaults to ``None``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.Layer.__init__``.

    Examples:
        >>> layer = RKSLayer(n_features=256, sigma_init=2.0, seed=0)
        >>> X = keras.random.normal(shape=(100, 8), seed=1)
        >>> Z = layer(X)              # shape (100, 256)
        >>> K_approx = Z @ Z.T       # approximates 100x100 RBF Gram matrix

    Note:
        Because the frequencies are pre-scaled, the forward pass is a
        single matrix multiply + cosine, with ``O(n * d * D)``
        complexity (``D = n_features``).  There is no log-sigma weight
        and no ``exp`` call, making this marginally faster than
        :class:`RFFLayer` at the cost of losing bandwidth tunability.

    References:
        Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale
        Kernel Machines." *Advances in Neural Information Processing
        Systems 20* (NIPS 2007), pp. 1177--1184.

    See Also:
        :class:`RFFLayer`: Generalized variant with an optional trainable
            bandwidth.
        :class:`NystromLayer`: Alternative low-rank approximation based
            on landmark points.
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
        """Sample pre-scaled random frequencies and biases.

        Frequencies are drawn from ``N(0, I)`` and divided by
        ``sigma_init`` at build time, so no rescaling is needed in
        the forward pass.

        Args:
            input_shape: Shape tuple ``(n, d)`` of the expected input.
                The last element ``d`` determines the frequency matrix
                shape ``(d, n_features)``.
        """
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
        """Project inputs into the random kitchen sinks feature space.

        Applies the mapping
        ``Z = sqrt(2/D) * cos(X @ omega + b)``
        where ``omega`` already incorporates the ``1/sigma`` scaling.

        Args:
            X: Input tensor of shape ``(n, d)`` with dtype ``float32``
                or ``float64``.

        Returns:
            Feature matrix ``Z`` of shape ``(n, n_features)`` such that
            ``Z @ Z.T`` approximates the ``(n, n)`` RBF Gram matrix
            with bandwidth ``sigma_init``.  Same dtype as *X*.
        """
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
