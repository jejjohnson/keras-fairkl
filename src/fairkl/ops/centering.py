"""Centering operations for kernel matrices.

This module provides utilities to center kernel matrices in the
reproducing kernel Hilbert space (RKHS).  Centering is a prerequisite
for kernel PCA, kernel alignment scores, and the HSIC-based fairness
objectives used throughout FairKL.

All functions are pure and use only ``keras.ops``.
"""

from __future__ import annotations

import keras.ops as ops


def centering_matrix(n: int):
    """Build the centering matrix ``H``.

    Constructs the ``n x n`` centering (hat) matrix

    $$H = I_n - \\frac{1}{n}\\,\\mathbf{1}\\mathbf{1}^\\top$$

    where ``I_n`` is the identity and ``\\mathbf{1}`` is the
    all-ones column vector of length ``n``.

    ``H`` is **symmetric** (``H = H^T``) and **idempotent**
    (``H^2 = H``), so it acts as an orthogonal projection onto the
    subspace orthogonal to the constant vector.  Multiplying a data
    matrix ``X`` on the left by ``H`` subtracts the column-wise mean
    (``HX = X - mean(X, axis=0)``).

    Args:
        n: Matrix dimension (number of samples).  Must be a positive
            integer.

    Returns:
        Centering matrix of shape ``(n, n)`` with float dtype.  The
        matrix has rank ``n - 1`` and eigenvalues ``{0, 1}``
        (with multiplicity ``1`` and ``n - 1``, respectively).

    Example:
        >>> from fairkl.ops.centering import centering_matrix
        >>> H = centering_matrix(4)
        >>> H.shape
        (4, 4)

    Note:
        This function materializes a dense ``(n, n)`` matrix.  For large
        ``n``, prefer :func:`center_kernel` which applies the centering
        operation directly to a kernel matrix in ``O(n^2)`` without
        forming ``H`` explicitly.

    See Also:
        :func:`center_kernel`: Apply centering to a kernel matrix
            without materializing ``H``.
    """
    return ops.eye(n) - ops.ones((n, n)) / n


def center_kernel(K):
    """Center a kernel matrix in feature space.

    Computes the double-centered kernel matrix

    $$\\tilde{K} = H K H$$

    where ``H = I - (1/n)\\,\\mathbf{1}\\mathbf{1}^\\top`` is the
    centering matrix, **without** ever materializing ``H``.  The
    element-wise formula is

    $$\\tilde{K}_{ij} = K_{ij}
        - \\frac{1}{n}\\sum_k K_{ik}
        - \\frac{1}{n}\\sum_k K_{kj}
        + \\frac{1}{n^2}\\sum_{k,l} K_{kl}$$

    Centering removes the mean embedding in the RKHS, which is a
    prerequisite for kernel PCA and the HSIC independence test used in
    FairKL's fairness objectives.

    Args:
        K: Kernel matrix of shape ``(n, n)``.  Should be symmetric
            (e.g., the output of :func:`~fairkl.kernels.exact.rbf_kernel`).

    Returns:
        Centered kernel matrix of shape ``(n, n)`` with the same dtype
        as the input.  The result is symmetric with row and column sums
        equal to zero (up to floating-point precision).

    Example:
        >>> import keras
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> from fairkl.ops.centering import center_kernel
        >>> X = keras.random.normal((50, 5), seed=0)
        >>> K = rbf_kernel(X, sigma=1.0)
        >>> Kc = center_kernel(K)
        >>> Kc.shape
        (50, 50)

    Note:
        Computational complexity is ``O(n^2)`` -- two passes over the
        matrix to compute row means, column means, and the grand mean.
        This is much cheaper than the naive ``H @ K @ H`` which would
        cost ``O(n^3)`` from two dense matrix multiplications plus
        ``O(n^2)`` to build ``H``.

    See Also:
        :func:`centering_matrix`: Build the explicit ``H`` matrix
            (useful for small ``n`` or pedagogical purposes).
    """
    row_mean = ops.mean(K, axis=1, keepdims=True)  # (n, 1)
    col_mean = ops.mean(K, axis=0, keepdims=True)  # (1, n)
    total_mean = ops.mean(K)  # scalar
    return K - row_mean - col_mean + total_mean
