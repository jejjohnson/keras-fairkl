"""Approximate kernel functions: Nystrom, RFF, RKS.

This module provides **low-rank / random-feature** approximations that
trade a controllable amount of accuracy for dramatically lower memory and
compute.  Use these when the exact Gram matrix from
:mod:`fairkl.kernels.exact` is too expensive to materialize
(roughly ``n > 10 000``).

All functions are pure, stateless, and use only ``keras.ops``.
"""

from __future__ import annotations

import math

import keras
import keras.ops as ops


def nystrom_approximate(X, landmarks, kernel_fn):
    """Low-rank Nystrom approximation of a kernel matrix.

    Approximates the full ``n x n`` Gram matrix by projecting through a
    small set of ``m`` landmark (inducing) points:

    $$\\tilde{K} = K_{XZ}\\, K_{ZZ}^{-1}\\, K_{ZX}$$

    where ``Z`` denotes the landmarks, ``K_{XZ}`` is the ``(n, m)``
    cross-kernel, and ``K_{ZZ}`` is the ``(m, m)`` landmark kernel.
    This yields a rank-``m`` positive semi-definite approximation to
    ``K(X, X)`` that becomes exact when ``m = n``.

    The implementation solves the inner system via Cholesky factorization
    of ``K_{ZZ}`` (with a small diagonal jitter of ``1e-6`` for
    numerical stability), giving a symmetric factorized form
    ``L^{-1} K_{ZX}`` whose outer product is the result.

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        landmarks: Landmark (inducing) points of shape ``(m, d)`` with
            ``m << n``.  These are typically selected by uniform random
            sub-sampling or k-means clustering on the training set.
        kernel_fn: A callable ``(A, B) -> K`` that computes a kernel
            matrix between two sets of points, e.g. :func:`rbf_kernel`.

    Returns:
        Approximate kernel matrix of shape ``(n, n)`` and dtype matching
        the input.  The result is symmetric positive semi-definite with
        rank at most ``m``.

    Example:
        >>> import keras
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> from fairkl.kernels.approximate import nystrom_approximate
        >>> X = keras.random.normal((200, 5), seed=0)
        >>> landmarks = X[:20]  # use first 20 points
        >>> K_approx = nystrom_approximate(X, landmarks, rbf_kernel)
        >>> K_approx.shape
        (200, 200)

    Note:
        Computational complexity is dominated by computing ``K_{XZ}``
        (``O(n * m * d)``) and the Cholesky solve (``O(m^3 / 3)``),
        giving an overall cost of ``O(n * m * d + m^3)`` -- a large
        saving over the ``O(n^2 * d)`` cost of the exact Gram matrix
        when ``m << n``.

    References:
        Williams, C. K. I. & Seeger, M. (2001). Using the Nystrom method
        to speed up kernel machines. *Advances in Neural Information
        Processing Systems 13*, pp. 682--688.

    See Also:
        :func:`random_fourier_features`: An alternative low-rank
            approximation via random sampling in the spectral domain.
    """
    K_xz = kernel_fn(X, landmarks)  # (n, m)
    K_zz = kernel_fn(landmarks, landmarks)  # (m, m)
    # Solve K_zz @ A = K_zx via Cholesky for stability
    # A = K_zz^{-1} @ K_xz^T, then result = K_xz @ A
    n = ops.shape(K_zz)[0]
    K_zz_reg = K_zz + 1e-6 * ops.eye(n)
    L = ops.cholesky(K_zz_reg)
    # Solve L @ tmp = K_xz^T
    K_zx = ops.transpose(K_xz)  # (m, n)
    tmp = ops.solve(L, K_zx)  # (m, n)
    return ops.matmul(ops.transpose(tmp), tmp)  # (n, n)


def random_fourier_features(X, n_features: int, sigma: float, seed=None):
    """Random Fourier Features (RFF) approximation of the RBF kernel.

    Constructs an explicit low-dimensional feature map ``Z`` such that
    inner products in feature space approximate the RBF kernel:

    $$Z Z^\\top \\approx K$$

    where ``K_{ij} = \\exp(-\\|x_i - x_j\\|^2 / 2\\sigma^2)``.

    Each column of ``Z`` is computed as

    $$z_j(x) = \\sqrt{\\frac{2}{D}} \\cos(\\omega_j^\\top x + b_j)$$

    where ``\\omega_j \\sim \\mathcal{N}(0, \\sigma^{-2} I_d)`` and
    ``b_j \\sim \\mathrm{Uniform}(0, 2\\pi)``, and ``D`` is
    ``n_features``.

    This exploits Bochner's theorem: the Fourier transform of a
    shift-invariant positive-definite kernel is a non-negative measure,
    so we can Monte-Carlo sample from it to build unbiased feature maps.

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        n_features: Dimensionality ``D`` of the random feature map.
            Larger values give a better approximation at higher cost.
            A common rule of thumb is ``D ~ 10 * d`` to ``100 * d``.
        sigma: RBF bandwidth (length-scale).  Must be positive.  Should
            match the ``sigma`` used in :func:`~fairkl.kernels.exact.rbf_kernel`.
        seed: Random seed (integer or ``None``) for reproducibility.
            Passed to ``keras.random.SeedGenerator``.

    Returns:
        Feature matrix ``Z`` of shape ``(n, n_features)`` with the same
        dtype as the input.  The approximate kernel is recovered as
        ``Z @ Z.T``.

    Example:
        >>> import keras
        >>> from fairkl.kernels.approximate import random_fourier_features
        >>> X = keras.random.normal((500, 10), seed=0)
        >>> Z = random_fourier_features(X, n_features=200, sigma=1.0, seed=42)
        >>> Z.shape
        (500, 200)
        >>> K_approx = Z @ Z.T  # approximate RBF Gram matrix
        >>> K_approx.shape
        (500, 500)

    Note:
        Computational complexity is ``O(n * D * d)`` for the projection
        and ``O(n * D)`` for the cosine, where ``D = n_features``.
        Storing the feature matrix costs ``O(n * D)`` memory, which is
        much cheaper than the ``O(n^2)`` exact Gram matrix when
        ``D << n``.

    References:
        Rahimi, A. & Recht, B. (2007). Random features for large-scale
        kernel machines. *Advances in Neural Information Processing
        Systems 20*, pp. 1177--1184.

    See Also:
        :func:`random_kitchen_sinks`: A variant that accepts
            pre-sampled weights, useful when the same projection is
            reused across calls.
        :func:`nystrom_approximate`: A data-dependent low-rank
            approximation via landmark points.
    """
    d = ops.shape(X)[-1]
    seed_gen = keras.random.SeedGenerator(seed)
    # omega ~ N(0, 1/sigma^2 I) => scale standard normal by 1/sigma
    omega = keras.random.normal(shape=(d, n_features), seed=seed_gen) / sigma
    b = keras.random.uniform(
        shape=(1, n_features), minval=0.0, maxval=2.0 * math.pi, seed=seed_gen
    )
    projection = ops.matmul(X, omega) + b  # (n, n_features)
    return ops.sqrt(2.0 / n_features) * ops.cos(projection)


def random_kitchen_sinks(X, W, b):
    """Random Kitchen Sinks (RKS) projection with pre-sampled weights.

    Computes the same random Fourier feature map as
    :func:`random_fourier_features`, but accepts **pre-sampled** weights
    ``W`` and biases ``b`` instead of generating them internally.  This
    "fixed-weight" variant is useful when:

    * The same random projection must be applied to multiple batches or
      datasets (e.g., training vs. test splits).
    * The weights are stored as part of a model checkpoint and must be
      re-used exactly at inference time.

    The feature map is

    $$z(x) = \\sqrt{\\frac{2}{D}} \\cos(W^\\top x + b)$$

    and the approximate kernel is ``Z @ Z^T``.

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        W: Random projection weight matrix of shape ``(d, n_features)``.
            For an RBF kernel with bandwidth ``sigma``, sample each entry
            i.i.d. from ``N(0, 1/sigma^2)``.
        b: Random bias vector of shape ``(n_features,)`` or
            ``(1, n_features)``.  Should be sampled uniformly from
            ``[0, 2*pi)``.

    Returns:
        Feature matrix ``Z`` of shape ``(n, n_features)`` with the same
        dtype as the input.  The approximate kernel is recovered as
        ``Z @ Z.T``.

    Example:
        >>> import keras
        >>> from fairkl.kernels.approximate import random_kitchen_sinks
        >>> X = keras.random.normal((100, 5), seed=0)
        >>> W = keras.random.normal((5, 50), seed=1) / 1.0  # sigma=1.0
        >>> b = keras.random.uniform((1, 50), minval=0.0, maxval=6.2832, seed=2)
        >>> Z = random_kitchen_sinks(X, W, b)
        >>> Z.shape
        (100, 50)

    Note:
        Computational complexity is ``O(n * D * d)`` for the matrix
        multiplication plus ``O(n * D)`` for the cosine, where
        ``D = n_features``.  This is identical to
        :func:`random_fourier_features`; the only difference is that
        weight sampling is handled externally.

    References:
        Rahimi, A. & Recht, B. (2007). Random features for large-scale
        kernel machines. *Advances in Neural Information Processing
        Systems 20*, pp. 1177--1184.

    See Also:
        :func:`random_fourier_features`: Convenience wrapper that
            samples ``W`` and ``b`` internally.
    """
    n_features = ops.shape(W)[-1]
    projection = ops.matmul(X, W) + b  # (n, n_features)
    return ops.sqrt(2.0 / n_features) * ops.cos(projection)
