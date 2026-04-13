"""Fair kernel PCA.

Performs principal component analysis in the reproducing kernel Hilbert
space (RKHS) induced by the chosen kernel, with an optional CKA
fairness penalty that discourages statistical dependence between the
projections and sensitive attributes.

The optimization objective is::

    max_V  tr(V^T K_c V) - mu * CKA(K_c V, q)
    s.t.   V^T V = I

where ``K_c`` is the centered kernel matrix, ``V`` is the ``(n, k)``
matrix of dual coefficients defining the projection directions in the
RKHS, and ``mu`` controls the fairness penalty.

Out-of-sample ``transform`` follows the scikit-learn ``KernelPCA``
convention: a cross-kernel matrix is computed and centered using stored
training statistics, then multiplied by ``V``.

``inverse_transform`` uses kernel ridge regression for the pre-image
problem (Bakir et al., 2004).
"""

from __future__ import annotations

import jax
import keras
import keras.ops as ops

from fairkl.kernels.exact import rbf_kernel
from fairkl.metrics.cka import cka_rbf
from fairkl.ops.centering import center_kernel


class FairKernelPCA(keras.Model):
    r"""Fair kernel PCA with a CKA fairness penalty.

    Performs principal component analysis in the reproducing kernel
    Hilbert space (RKHS) induced by the chosen kernel function.  The
    model learns dual coefficients :math:`V \in \mathbb{R}^{n \times k}`
    that define the projection directions in feature space.  The
    training objective is:

    .. math::
        \max_{V}\;
            \frac{1}{n}\,\mathrm{tr}\!\bigl(V^\top K_c V\bigr)
          - \mu\, \mathrm{CKA}(K_c V,\; q)
        \quad\text{s.t.}\quad V^\top V = I_k

    where :math:`K_c = H K H` is the centered kernel matrix
    (:math:`H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top`),
    :math:`k` is ``n_components``, and :math:`\mu` controls the
    fairness penalty.  The orthogonality constraint is relaxed via a
    quadratic penalty, as in :class:`FairPCA`.

    **Out-of-sample extension** follows the scikit-learn ``KernelPCA``
    pattern.  During ``fit``, the column means and global mean of the
    training kernel matrix are stored.  At ``transform`` time, a
    cross-kernel :math:`K(X_{\text{test}}, X_{\text{train}})` is
    computed and centered using these stored statistics (without
    recomputing or storing the full training kernel).

    **Pre-image reconstruction** (``inverse_transform``) uses kernel
    ridge regression to learn a mapping from the low-dimensional
    projections back to input space, following Bakir et al. (2004).

    Args:
        n_components (int): Number of kernel principal components
            ``k`` to retain.  Must be positive and at most ``n`` (the
            number of training samples).
        sigma (float): RBF bandwidth for the feature kernel
            :math:`k(x, x') = \exp(-\|x-x'\|^2 / (2\sigma^2))`.
            Ignored when ``kernel="linear"``.  Default ``1.0``.
        mu (float): CKA fairness penalty weight :math:`\mu`.  ``mu=0``
            recovers standard kernel PCA.  Larger values enforce
            stronger statistical independence between the projections
            and sensitive attributes.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute
            kernel inside CKA.  Default ``1.0``.
        kernel (str): Kernel type.  One of ``"rbf"`` (Gaussian /
            squared-exponential) or ``"linear"``
            (:math:`k(x,x') = x^\top x'`).  Default ``"rbf"``.
        alpha (float): Ridge regularization parameter for the
            ``inverse_transform`` pre-image regression.  Larger values
            improve numerical stability at the cost of reconstruction
            fidelity.  Default ``1.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__``.

    Examples:
        Fit, transform, and reconstruct:

        >>> import numpy as np
        >>> from fairkl.models import FairKernelPCA
        >>> X = np.random.randn(150, 8).astype("float32")
        >>> q = np.random.randn(150, 1).astype("float32")
        >>> model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.5)
        >>> model.fit(X, q=q, epochs=200, lr=1e-2)
        >>> Z = model.transform(X)          # shape (150, 3)
        >>> X_hat = model.inverse_transform(Z)  # shape (150, 8)

        One-step ``fit_transform``:

        >>> Z = model.fit_transform(X, q=q, epochs=200, lr=1e-2)

    Note:
        * The centered kernel matrix :math:`K_c` is computed once
          during ``fit`` and reused for all optimization epochs.
        * Centering statistics (``_K_col_mean``, ``_K_total_mean``)
          are stored so that ``transform`` can center the cross-kernel
          efficiently without recomputing or storing :math:`K_{\text{train}}`.
        * Computational complexity per epoch is :math:`O(n^2 k)` for
          the projection plus :math:`O(n^2)` for the CKA term.
          Building the kernel matrix costs :math:`O(n^2 d)`.
        * ``inverse_transform`` solves an :math:`n \times n` linear
          system, costing :math:`O(n^3)`.

    References:
        * Scholkopf, B. et al. (1998). "Nonlinear Component Analysis
          as a Kernel Eigenvalue Problem." *Neural Computation*, 10(5),
          1299--1319.  (Kernel PCA.)
        * Bakir, G. et al. (2004). "Learning to Find Pre-Images."
          *NeurIPS*.  (Kernel ridge regression pre-image.)
        * Cortes, C. et al. (2012). "Algorithms for Learning Kernels
          Based on Centered Alignment." *JMLR*, 13, 795--828.  (CKA.)
        * Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning
          in Python." *JMLR*, 12, 2825--2830.  (KernelPCA centering
          pattern.)

    See Also:
        :class:`FairPCA`: Primal-form (linear) fair PCA.
        :class:`FairKernelRidge`: Dual-form fair regression.
        :func:`~fairkl.ops.centering.center_kernel`: Kernel centering
            used internally.
    """

    def __init__(
        self,
        n_components: int,
        sigma: float = 1.0,
        mu: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        alpha: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.sigma = sigma
        self.mu = mu
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.alpha = alpha
        # Populated during fit
        self._X_train = None
        self._V = None  # (n, n_components)
        self._K_col_mean = None  # (1, n) centering stat
        self._K_total_mean = None  # scalar centering stat

    def _compute_kernel(self, X, Y=None):
        if self.kernel == "linear":
            from fairkl.kernels.exact import linear_kernel

            return linear_kernel(X, Y)
        return rbf_kernel(X, Y, sigma=self.sigma)

    def fit(self, X, q=None, epochs: int = 200, lr: float = 1e-2, **kwargs):
        r"""Learn the fair kernel PCA projection.

        Computes the kernel matrix :math:`K`, stores centering
        statistics for out-of-sample use, and optimizes the dual
        coefficient matrix :math:`V` via Adam gradient descent on:

        .. math::
            \mathcal{L} =
              - \frac{1}{n}\|K_c V\|_F^2
              + \mu\, \mathrm{CKA}(K_c V, q)
              + w_{\mathrm{ortho}}\, \|V^\top V - I_k\|_F^2

        **Centering statistics stored:**

        * ``_K_col_mean``: shape ``(1, n)`` -- column means of the
          training kernel matrix.
        * ``_K_total_mean``: scalar -- grand mean of the training
          kernel matrix.

        These are used by :meth:`transform` to center cross-kernel
        matrices without recomputing :math:`K_{\text{train}}`.

        Args:
            X (array-like): Training inputs of shape ``(n, d)`` where
                ``n`` is the number of samples and ``d`` is the feature
                dimensionality.  Converted to float32.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.  When ``None`` or when ``mu=0``, the
                fairness penalty is omitted.
            epochs (int): Number of full-batch Adam epochs.  Default
                ``200``.
            lr (float): Adam learning rate.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            self: The fitted model instance (for method chaining).

        Note:
            The kernel matrix :math:`K` is computed once and centered
            in :math:`O(n^2)`.  Each epoch then costs
            :math:`O(n^2 k)` for the projection gradient.  ``V`` is
            initialized only on the first call; subsequent calls to
            ``fit`` reuse the existing ``V`` as a warm start.
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        if q is not None:
            q = ops.convert_to_tensor(q, dtype="float32")
            if len(ops.shape(q)) == 1:
                q = ops.expand_dims(q, axis=-1)

        self._X_train = X
        n = ops.shape(X)[0]

        K = self._compute_kernel(X)

        # Store centering statistics (sklearn KernelCenterer pattern)
        self._K_col_mean = ops.mean(K, axis=0, keepdims=True)  # (1, n)
        self._K_total_mean = ops.mean(K)  # scalar
        K_c = center_kernel(K)

        # Initialize V as (n, n_components) — only on first fit
        if self._V is None:
            self._V = self.add_weight(
                name="V",
                shape=(n, self.n_components),
                initializer="glorot_uniform",
                trainable=True,
            )

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        optimizer.build([self._V])

        ortho_weight = 10.0
        n_comp = self.n_components
        mu = self.mu
        sigma_q = self.sigma_q
        n_float = ops.cast(n, "float32")

        def loss_fn(V_val):
            Z = ops.matmul(K_c, V_val)
            variance = ops.sum(Z * Z) / n_float
            loss = -variance
            if q is not None and mu > 0:
                loss = loss + mu * cka_rbf(Z, q, sigma_f=1.0, sigma_q=sigma_q)
            VtV = ops.matmul(ops.transpose(V_val), V_val)
            eye = ops.eye(n_comp)
            ortho_penalty = ops.sum((VtV - eye) ** 2)
            loss = loss + ortho_weight * ortho_penalty
            return loss

        for _ in range(epochs):
            _loss_val, grads = jax.value_and_grad(loss_fn)(self._V.value)
            optimizer.apply([grads], [self._V])

        return self

    def _center_cross_kernel(self, K_cross):
        r"""Center a cross-kernel matrix using stored training statistics.

        Implements the scikit-learn ``KernelCenterer.transform`` formula:

        .. math::
            \tilde{K} = K_{\text{cross}}
              - \bar{K}_{\text{col}}
              - \mathrm{rowmean}(K_{\text{cross}})
              + \bar{K}_{\text{total}}

        where :math:`\bar{K}_{\text{col}}` and
        :math:`\bar{K}_{\text{total}}` are the column means and grand
        mean of the *training* kernel stored during :meth:`fit`.

        Args:
            K_cross: Cross-kernel matrix of shape ``(m, n)``.

        Returns:
            Centered cross-kernel matrix of shape ``(m, n)``.
        """
        row_mean = ops.mean(K_cross, axis=1, keepdims=True)  # (m, 1)
        return K_cross - self._K_col_mean - row_mean + self._K_total_mean

    def transform(self, X):
        r"""Project new data into the fair kernel PCA space.

        Computes the cross-kernel matrix
        :math:`K_{\text{cross}} = k(X_{\text{test}}, X_{\text{train}})`
        and centers it using the stored training statistics (column
        means and global mean from ``fit``).  The centered cross-kernel
        is then multiplied by the dual coefficient matrix :math:`V`:

        .. math::
            Z = \tilde{K}_{\text{cross}}\, V

        where :math:`\tilde{K}` is the centered cross-kernel.  This
        avoids recomputing or storing the full training kernel matrix.

        Args:
            X (array-like): Test inputs of shape ``(m, d)`` where ``d``
                must match the training feature dimensionality.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32
            containing the kernel PCA projections.

        Note:
            The centering formula follows the scikit-learn
            ``KernelCenterer.transform`` convention:

            .. math::
                \tilde{K} = K_{\text{cross}}
                  - \bar{K}_{\text{col}}
                  - \mathrm{rowmean}(K_{\text{cross}})
                  + \bar{K}_{\text{total}}
        """
        X = ops.convert_to_tensor(X)
        K_cross = self._compute_kernel(X, self._X_train)  # (m, n)
        K_cross_c = self._center_cross_kernel(K_cross)
        return ops.matmul(K_cross_c, self._V)

    def inverse_transform(self, Z):
        r"""Approximate pre-image via kernel ridge regression.

        Reconstructs inputs from their kernel PCA projections by
        learning a mapping from projection space back to input space.
        The pre-image problem is solved via kernel ridge regression
        following Bakir et al. (2004):

        1. Compute training projections :math:`Z_{\text{train}} = K_c V`.
        2. Build the kernel in projection space:
           :math:`K_Z = Z_{\text{train}} Z_{\text{train}}^\top`.
        3. Solve the ridge system for dual coefficients:
           :math:`C = (K_Z + \alpha I)^{-1} X_{\text{train}}`.
        4. Map new projections: :math:`\hat{X} = K(Z, Z_{\text{train}})\, C`
           where :math:`K(Z, Z_{\text{train}}) = Z Z_{\text{train}}^\top`.

        Args:
            Z (array-like): Projections of shape ``(m, n_components)``
                as returned by :meth:`transform`.

        Returns:
            Tensor of shape ``(m, d)`` with dtype float32 containing
            the approximate reconstructed inputs.

        Note:
            * This method recomputes the training kernel and
              projections each time it is called, costing
              :math:`O(n^2 d + n^3)`.  Cache the result if calling
              repeatedly with the same ``Z``.
            * The ``alpha`` constructor parameter controls the ridge
              regularization of the pre-image regression.  Too-small
              values may lead to numerical instability; too-large values
              produce overly smooth reconstructions.

        References:
            * Bakir, G. et al. (2004). "Learning to Find Pre-Images."
              *NeurIPS*.
        """
        # Compute training projections
        K_train = self._compute_kernel(self._X_train)
        K_c = center_kernel(K_train)
        Z_train = ops.matmul(K_c, self._V)  # (n, k)

        # Kernel ridge regression: dual_coef = (K_Z + alpha I)^{-1} X_train
        K_Z = ops.matmul(Z_train, ops.transpose(Z_train))  # (n, n)
        n = ops.shape(K_Z)[0]
        system = K_Z + self.alpha * ops.eye(n)
        dual_coef = ops.solve(system, self._X_train)  # (n, d)

        # Project new Z through the kernel in component space
        K_new = ops.matmul(Z, ops.transpose(Z_train))  # (m, n)
        return ops.matmul(K_new, dual_coef)  # (m, d)

    def fit_transform(self, X, q=None, **kwargs):
        """Fit the model and project the training data in one call.

        Equivalent to ``model.fit(X, q=q, **kwargs)`` followed by
        projecting with the training kernel, but avoids an extra
        cross-kernel computation by reusing the centered training
        kernel directly.

        Args:
            X (array-like): Training inputs of shape ``(n, d)``.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.
            **kwargs: Forwarded to :meth:`fit` (e.g. ``epochs``, ``lr``).

        Returns:
            Tensor of shape ``(n, n_components)`` with dtype float32
            containing the kernel PCA projections of the training data.
        """
        self.fit(X, q=q, **kwargs)
        K = self._compute_kernel(X)
        K_c = center_kernel(K)
        return ops.matmul(K_c, self._V)

    def call(self, X):
        return self.transform(X)

    def get_config(self):
        config = super().get_config()
        config.update(
            n_components=self.n_components,
            sigma=self.sigma,
            mu=self.mu,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
            alpha=self.alpha,
        )
        return config
