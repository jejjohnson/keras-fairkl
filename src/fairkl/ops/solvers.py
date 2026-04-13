"""Linear system solvers: Cholesky and conjugate gradient.

This module provides two strategies for solving symmetric
positive-definite (SPD) linear systems ``Ax = b``:

* **Cholesky** -- direct, numerically stable, cubic cost; best when
  ``n`` fits comfortably in memory.
* **Conjugate gradient (CG)** -- iterative, matrix-free compatible,
  quadratic cost per step; best for large sparse or structured systems.

Both solvers are pure functions and use only ``keras.ops``, so they
run on every Keras backend (TensorFlow, JAX, PyTorch, NumPy).
"""

from __future__ import annotations

import keras.ops as ops


def solve_cholesky(A, b):
    """Solve ``Ax = b`` via Cholesky factorization.

    Factorizes the matrix as ``A = L L^T`` where ``L`` is lower
    triangular, then solves the two triangular systems

    $$L\\,y = b \\quad\\text{(forward substitution)}$$
    $$L^\\top x = y \\quad\\text{(back substitution)}$$

    Cholesky factorization is the gold-standard direct solver for
    symmetric positive-definite (SPD) systems.  It is numerically
    stable (no pivoting required for SPD matrices) and exactly twice
    as efficient as the general LU factorization.

    Args:
        A: Symmetric positive-definite matrix of shape ``(n, n)``.
            The function reads only the lower triangle; the upper
            triangle is assumed to mirror it.  If ``A`` is not SPD the
            Cholesky factorization will fail.
        b: Right-hand side tensor of shape ``(n,)`` for a single system
            or ``(n, k)`` for ``k`` systems with the same coefficient
            matrix.

    Returns:
        Solution tensor ``x`` with the same shape as ``b``.

    Example:
        >>> import keras
        >>> import keras.ops as ops
        >>> from fairkl.ops.solvers import solve_cholesky
        >>> n = 50
        >>> A = keras.random.normal((n, n), seed=0)
        >>> A = A @ ops.transpose(A) + 0.1 * ops.eye(n)  # make SPD
        >>> b = keras.random.normal((n,), seed=1)
        >>> x = solve_cholesky(A, b)
        >>> x.shape
        (50,)

    Note:
        The Cholesky factorization costs ``O(n^3 / 3)`` flops.  Each
        subsequent triangular solve costs ``O(n^2)`` per right-hand-side
        column, so solving for ``k`` columns costs ``O(n^3 / 3 + n^2 k)``
        total.  Memory usage is ``O(n^2)`` for the factor ``L``.

        For large systems where ``A`` has special structure (sparse,
        banded, low-rank-plus-diagonal), consider :func:`solve_cg`
        which can exploit matrix-vector products without forming ``A``
        explicitly.

    See Also:
        :func:`solve_cg`: Iterative conjugate-gradient solver for large
            or matrix-free SPD systems.
    """
    L = ops.cholesky(A)  # A = L L^T
    # Forward solve: L y = b
    y = ops.solve(L, b)
    # Backward solve: L^T x = y
    return ops.solve(ops.transpose(L), y)


def solve_cg(A, b, x0=None, maxiter: int = 100, tol: float = 1e-5):
    """Conjugate gradient (CG) solver for ``Ax = b``.

    Iteratively solves a symmetric positive-definite linear system using
    the method of conjugate directions.  At each iteration ``k``, CG
    minimizes the ``A``-norm of the error over the Krylov subspace
    ``K_k(A, r_0) = span{r_0, A r_0, ..., A^{k-1} r_0}``.

    In exact arithmetic, CG converges in at most ``n`` iterations
    (where ``n`` is the system size).  In practice, convergence is much
    faster when ``A`` is well-conditioned or has clustered eigenvalues.

    This implementation uses a **fixed-iteration Python loop** (not a
    traced ``while_loop``), so it is compatible with all Keras backends
    including JIT compilation.  The trade-off is that ``maxiter`` is
    baked into the computation graph at trace time.

    Args:
        A: Symmetric positive-definite matrix of shape ``(n, n)``, **or**
            a callable ``v -> A @ v`` that computes the matrix-vector
            product without forming ``A`` explicitly (matrix-free mode).
            Matrix-free mode is useful when ``A`` is too large to store
            or has exploitable structure (e.g., Kronecker, sparse).
        b: Right-hand side tensor of shape ``(n,)`` or ``(n, k)``.
        x0: Initial guess tensor with the same shape as ``b``.  A good
            initial guess can dramatically reduce the number of
            iterations.  Defaults to the zero vector.
        maxiter: Maximum number of CG iterations.  The solver terminates
            early if the squared residual norm drops below ``tol^2``.
        tol: Convergence tolerance.  The iteration stops when
            ``||r||_2 < tol``, where ``r = b - A x`` is the residual.

    Returns:
        Solution tensor ``x`` with the same shape as ``b``.  If the
        solver did not converge within ``maxiter`` iterations the
        current best iterate is returned.

    Example:
        >>> import keras
        >>> import keras.ops as ops
        >>> from fairkl.ops.solvers import solve_cg
        >>> n = 100
        >>> A = keras.random.normal((n, n), seed=0)
        >>> A = A @ ops.transpose(A) + 0.1 * ops.eye(n)  # make SPD
        >>> b = keras.random.normal((n,), seed=1)
        >>> x = solve_cg(A, b, maxiter=200, tol=1e-6)
        >>> x.shape
        (100,)

    Note:
        Each CG iteration costs one matrix-vector product (``O(n^2)``
        for a dense matrix, or cheaper for structured / sparse ``A``),
        plus ``O(n)`` for vector updates and inner products.  Total cost
        for ``k`` iterations is therefore ``O(k * n^2)`` in the dense
        case.

        A small epsilon (``1e-30``) is added to denominators to prevent
        division by zero in degenerate cases; this does not affect
        convergence in well-posed problems.

        For well-conditioned systems where a direct solve fits in memory,
        :func:`solve_cholesky` is typically faster and more accurate.

    References:
        Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate
        gradients for solving linear systems. *Journal of Research of
        the National Bureau of Standards*, 49(6), pp. 409--436.

        Shewchuk, J. R. (1994). An introduction to the conjugate
        gradient method without the agonizing pain. Technical report,
        Carnegie Mellon University.

    See Also:
        :func:`solve_cholesky`: Direct Cholesky solver for small to
            medium SPD systems.
    """
    x = ops.zeros_like(b) if x0 is None else ops.copy(x0)

    matvec = (lambda v: ops.matmul(A, v)) if not callable(A) else A

    r = b - matvec(x)
    p = ops.copy(r)
    rs_old = ops.sum(r * r)
    tol_sq = ops.convert_to_tensor(tol * tol, dtype=rs_old.dtype)

    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = ops.sum(p * Ap)
        alpha = rs_old / (pAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = ops.sum(r * r)
        if float(rs_new) < float(tol_sq):
            break
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new

    return x
