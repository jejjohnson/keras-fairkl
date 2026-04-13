"""Exact kernel functions: RBF, linear, polynomial.

This module provides the **exact kernel** building blocks for FairKL.
Each function computes a full Gram matrix from raw feature tensors using
only ``keras.ops``, so it works unchanged on every Keras backend
(TensorFlow, JAX, PyTorch, NumPy).

All functions are pure, stateless, and return dense ``(n, m)`` kernel
matrices.  For large-scale problems where materializing the full Gram
matrix is infeasible, see :mod:`fairkl.kernels.approximate`.
"""

from __future__ import annotations

import keras.ops as ops


def rbf_kernel(X, Y=None, sigma: float = 1.0):
    """Gaussian radial basis function (RBF) kernel matrix.

    Computes the Gram matrix ``K`` where each entry is

    $$K_{ij} = \\exp\\!\\Bigl(-\\frac{\\|x_i - y_j\\|^2}{2\\sigma^2}\\Bigr)$$

    The RBF kernel (also called the *squared-exponential* or *Gaussian*
    kernel) is a **universal** kernel, meaning it can approximate any
    continuous target function on a compact domain given enough data.  It
    is also **positive definite** for every ``sigma > 0``, so the
    resulting Gram matrix is always SPD when evaluated on distinct points.

    The bandwidth ``sigma`` controls the length-scale: small values yield
    a spiky kernel (more local), large values yield a smoother kernel.
    A common heuristic is to set ``sigma`` to the median pairwise
    distance (the *median heuristic*).

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        Y: Optional second input tensor of shape ``(m, d)``.  When
            ``None`` (the default) the kernel is evaluated as ``K(X, X)``,
            producing a symmetric ``(n, n)`` Gram matrix.
        sigma: Bandwidth (length-scale) parameter.  Must be positive.
            Controls the width of the Gaussian; larger values produce
            smoother kernels.

    Returns:
        Kernel matrix of shape ``(n, m)`` with dtype matching the input.
        All entries lie in the interval ``(0, 1]``.

    Example:
        >>> import keras
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> X = keras.random.normal((100, 5), seed=0)
        >>> K = rbf_kernel(X, sigma=1.0)
        >>> K.shape
        (100, 100)

    Note:
        The squared distances are computed via the expansion
        ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 x . y`` and clamped to
        zero to avoid small negative values from floating-point error.
        Computational complexity is ``O(n * m * d)`` for the distance
        calculation plus ``O(n * m)`` for the element-wise exponential.

    References:
        Scholkopf, B. & Smola, A. J. (2002). *Learning with Kernels:
        Support Vector Machines, Regularization, Optimization, and
        Beyond*. MIT Press.

    See Also:
        :func:`linear_kernel`: A simpler, non-universal kernel.
        :func:`~fairkl.kernels.approximate.random_fourier_features`:
            Scalable random-feature approximation of the RBF kernel.
    """
    if Y is None:
        Y = X
    # Squared Euclidean distances via expansion:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X_sqnorms = ops.sum(X * X, axis=-1, keepdims=True)  # (n, 1)
    Y_sqnorms = ops.sum(Y * Y, axis=-1, keepdims=True)  # (m, 1)
    dist_sq = (
        X_sqnorms + ops.transpose(Y_sqnorms) - 2.0 * ops.matmul(X, ops.transpose(Y))
    )
    # Clamp to avoid negative values from numerical error
    dist_sq = ops.maximum(dist_sq, 0.0)
    return ops.exp(-dist_sq / (2.0 * sigma * sigma))


def linear_kernel(X, Y=None):
    """Linear kernel matrix.

    Computes the Gram matrix

    $$K_{ij} = \\langle x_i,\\, y_j \\rangle = x_i^\\top y_j$$

    which is equivalent to ``K = X @ Y^T``.

    The linear kernel corresponds to an identity feature map
    (``\\phi(x) = x``), so the RKHS is simply the input space itself.
    It is **not** shift-invariant and **not** universal: it can only
    express linear decision boundaries.

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        Y: Optional second input tensor of shape ``(m, d)``.  When
            ``None`` (the default) the kernel is evaluated as ``K(X, X)``,
            producing a symmetric ``(n, n)`` Gram matrix.

    Returns:
        Kernel matrix of shape ``(n, m)`` with dtype matching the input.
        Entries are unbounded and may be negative.

    Example:
        >>> import keras
        >>> from fairkl.kernels.exact import linear_kernel
        >>> X = keras.random.normal((50, 10), seed=0)
        >>> K = linear_kernel(X)
        >>> K.shape
        (50, 50)

    Note:
        Computational complexity is ``O(n * m * d)`` from a single matrix
        multiplication.  When ``Y is None`` the result is guaranteed to be
        symmetric positive semi-definite.

    See Also:
        :func:`polynomial_kernel`: Generalizes the linear kernel by
            raising the inner product to a power.
        :func:`rbf_kernel`: A universal, shift-invariant alternative.
    """
    if Y is None:
        Y = X
    return ops.matmul(X, ops.transpose(Y))


def polynomial_kernel(X, Y=None, degree: int = 3, coef0: float = 1.0):
    """Polynomial kernel matrix.

    Computes the Gram matrix

    $$K_{ij} = \\bigl(x_i^\\top y_j + c\\bigr)^p$$

    where ``c`` is ``coef0`` and ``p`` is ``degree``.

    The polynomial kernel maps data into a feature space spanned by all
    monomials of the input features up to degree ``p``.  The ``coef0``
    parameter trades off the influence of higher-order versus lower-order
    terms: when ``coef0 = 0`` the kernel is *homogeneous* and only
    contains monomials of exactly degree ``p``.

    The kernel is positive semi-definite for all non-negative integer
    degrees and ``coef0 >= 0``.  Higher degrees increase expressiveness
    but can also amplify overfitting and numerical instability.

    Args:
        X: Input tensor of shape ``(n, d)`` where ``n`` is the number of
            samples and ``d`` is the feature dimensionality.
        Y: Optional second input tensor of shape ``(m, d)``.  When
            ``None`` (the default) the kernel is evaluated as ``K(X, X)``,
            producing a symmetric ``(n, n)`` Gram matrix.
        degree: Polynomial degree ``p``.  Must be a positive integer.
            ``degree=1`` recovers an affine variant of the linear kernel.
        coef0: Additive (bias) constant ``c``.  Must be non-negative to
            guarantee positive semi-definiteness.

    Returns:
        Kernel matrix of shape ``(n, m)`` with dtype matching the input.

    Example:
        >>> import keras
        >>> from fairkl.kernels.exact import polynomial_kernel
        >>> X = keras.random.normal((50, 10), seed=0)
        >>> K = polynomial_kernel(X, degree=3, coef0=1.0)
        >>> K.shape
        (50, 50)

    Note:
        Computational complexity is ``O(n * m * d)`` for the inner
        products plus ``O(n * m)`` for the element-wise power.  For high
        degrees, consider using the RBF kernel instead, which implicitly
        maps to an infinite-dimensional feature space without numerical
        blow-up.

    References:
        Scholkopf, B. & Smola, A. J. (2002). *Learning with Kernels:
        Support Vector Machines, Regularization, Optimization, and
        Beyond*. MIT Press, Section 2.3.

    See Also:
        :func:`linear_kernel`: The special case ``degree=1, coef0=0``.
        :func:`rbf_kernel`: A universal kernel with implicit
            infinite-degree feature map.
    """
    if Y is None:
        Y = X
    return ops.power(ops.matmul(X, ops.transpose(Y)) + coef0, degree)
