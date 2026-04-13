"""Fair kernel PCA.

Performs principal component analysis in the reproducing kernel Hilbert
space (RKHS) induced by the chosen kernel, with an optional CKA
fairness penalty that discourages statistical dependence between the
projections and sensitive attributes.

The optimization objective is::

    max_V  tr(V^T K_c V) - mu * CKA(K_c V, q)
    s.t.   V^T V = I

where ``K_c = H K H`` is the ``(n, n)`` centered kernel (Gram) matrix
(``H = I - (1/n) 11^T`` is the centering matrix), ``V`` is the
``(n, k)`` matrix of dual coefficients defining the ``k`` projection
directions in the RKHS, ``mu >= 0`` controls the CKA fairness penalty,
and ``q`` encodes sensitive attributes.

**Out-of-sample extension** follows the scikit-learn ``KernelPCA``
convention (Pedregosa et al., 2011): during ``fit``, the column means
and grand mean of the training kernel matrix are stored; at
``transform`` time, a cross-kernel matrix is computed and centered
using these stored statistics, then multiplied by ``V``.  This avoids
recomputing or storing the full training kernel matrix at test time.

**Pre-image reconstruction** (``inverse_transform``) uses kernel ridge
regression to learn a mapping from the low-dimensional projections back
to input space, following the approach of Bakir et al. (2004).

See :class:`FairPCA` for the primal-form (linear) variant.
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
            \underbrace{
                \frac{1}{n}\,\mathrm{tr}\!\bigl(V^\top K_c V\bigr)
            }_{\text{variance in RKHS}}
          - \underbrace{
                \mu\, \mathrm{CKA}(K_c V,\; q)
            }_{\text{fairness penalty}}
        \quad\text{s.t.}\quad V^\top V = I_k

    where :math:`K_c = H K H` is the centered kernel matrix
    (:math:`H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top` is the
    centering matrix), :math:`k` is ``n_components``, and
    :math:`\mu \ge 0` controls the fairness penalty.  The
    orthogonality constraint is relaxed via a quadratic penalty (weight
    ``10.0``), as in :class:`FairPCA`.

    **Relationship to standard kernel PCA.** In standard kernel PCA
    (Scholkopf et al., 1998), the top-:math:`k` eigenvectors of
    :math:`K_c` define the projection.  Here, the eigendecomposition
    is replaced by gradient-based optimization, which allows
    incorporating the non-linear CKA fairness penalty.  When ``mu=0``,
    the solution converges to the standard kernel PCA directions (up
    to optimization tolerance and the orthogonality relaxation).

    **Out-of-sample extension** follows the scikit-learn ``KernelPCA``
    centering pattern (Pedregosa et al., 2011).  During :meth:`fit`,
    the column means :math:`\bar{K}_{\mathrm{col}}` (shape
    ``(1, n)``) and global mean :math:`\bar{K}_{\mathrm{total}}`
    (scalar) of the training kernel matrix are stored.  At
    :meth:`transform` time, a cross-kernel
    :math:`K(X_{\text{test}}, X_{\text{train}})` is computed and
    centered using these stored statistics:

    .. math::
        \tilde{K} = K_{\text{cross}}
          - \bar{K}_{\mathrm{col}}
          - \mathrm{rowmean}(K_{\text{cross}})
          + \bar{K}_{\mathrm{total}}

    This avoids recomputing or storing the full ``(n, n)`` training
    kernel at test time.

    **Pre-image reconstruction** (:meth:`inverse_transform`) uses
    kernel ridge regression to learn a mapping from the
    low-dimensional projections back to input space, following Bakir
    et al. (2004).  Specifically, it solves
    :math:`C = (K_Z + \alpha I)^{-1} X_{\text{train}}` where
    :math:`K_Z = Z_{\text{train}} Z_{\text{train}}^\top` is the
    linear kernel in projection space, and then maps new projections
    via :math:`\hat{X} = Z Z_{\text{train}}^\top C`.

    Args:
        n_components (int): Number of kernel principal components
            ``k`` to retain.  Must satisfy
            ``1 <= n_components <= n`` where ``n`` is the number of
            training samples.
        sigma (float): RBF bandwidth for the feature kernel
            :math:`k(x, x') = \exp(-\|x-x'\|^2 / (2\sigma^2))`.
            Controls the length scale of the kernel.  Ignored when
            ``kernel="linear"``.  Default ``1.0``.
        mu (float): CKA fairness penalty weight :math:`\mu \ge 0`.
            ``mu=0`` recovers standard kernel PCA.  Larger values
            enforce stronger statistical independence between the
            projections and sensitive attributes.  Typical values
            range from ``0.01`` to ``10.0``.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute
            kernel inside CKA.  Default ``1.0``.
        kernel (str): Kernel type.  One of:

            - ``"rbf"`` -- Gaussian / squared-exponential kernel
              :math:`k(x,x') = \exp(-\|x-x'\|^2 / 2\sigma^2)`.
            - ``"linear"`` -- linear kernel
              :math:`k(x,x') = x^\top x'` (no bandwidth parameter).

            Default ``"rbf"``.
        alpha (float): Ridge regularization parameter for the
            :meth:`inverse_transform` pre-image regression.  Larger
            values improve numerical stability at the cost of
            reconstruction fidelity.  Default ``1.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__`` (e.g. ``name``, ``dtype``).

    Attributes:
        n_components (int): Number of retained components.
        sigma (float): Stored RBF bandwidth.
        mu (float): Stored CKA penalty weight.
        sigma_q (float): Stored sensitive-attribute RBF bandwidth.
        kernel (str): Stored kernel type.
        alpha (float): Stored pre-image ridge parameter.
        _X_train (Tensor or None): Training inputs stored after
            :meth:`fit`, shape ``(n, d)``.  Used by :meth:`transform`
            and :meth:`inverse_transform`.
        _V (Variable or None): Dual coefficient matrix of shape
            ``(n, k)``, optimized during :meth:`fit`.
        _K_col_mean (Tensor or None): Column means of the training
            kernel, shape ``(1, n)``.  Used for cross-kernel centering.
        _K_total_mean (Tensor or None): Grand mean (scalar) of the
            training kernel.  Used for cross-kernel centering.

    Examples:
        Full fit, transform, and reconstruct workflow:

        >>> import numpy as np
        >>> from fairkl.models import FairKernelPCA
        >>> X = np.random.randn(150, 8).astype("float32")
        >>> q = np.random.randn(150, 1).astype("float32")
        >>> model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.5)
        >>> model.fit(X, q=q, epochs=200, lr=1e-2)
        >>> Z = model.transform(X)              # shape (150, 3)
        >>> X_hat = model.inverse_transform(Z)  # shape (150, 8)

        One-step ``fit_transform``:

        >>> Z = model.fit_transform(X, q=q, epochs=200, lr=1e-2)

        Out-of-sample projection (no recomputation of training kernel):

        >>> X_new = np.random.randn(50, 8).astype("float32")
        >>> Z_new = model.transform(X_new)  # shape (50, 3)

    Note:
        * The centered kernel matrix :math:`K_c` is computed once
          during :meth:`fit` and reused for all optimization epochs.
          It is **not** stored after fitting -- only the centering
          statistics are kept.
        * Centering statistics (``_K_col_mean``, ``_K_total_mean``)
          are stored so that :meth:`transform` can center the
          cross-kernel efficiently without recomputing or storing
          :math:`K_{\text{train}}`.
        * **Computational complexity:**

          - ``fit``: :math:`O(n^2 d)` to build the kernel, then
            :math:`O(n^2 k)` per epoch for the projection gradient,
            plus :math:`O(n^2)` per epoch for the CKA term.
          - ``transform``: :math:`O(m \cdot n \cdot d)` to build the
            cross-kernel plus :math:`O(m \cdot n \cdot k)` for the
            projection.
          - ``inverse_transform``: :math:`O(n^2 d + n^3)` to solve the
            ridge system, plus :math:`O(m \cdot n)` for the
            pre-image mapping.

        * :meth:`inverse_transform` recomputes the training kernel and
          projections each time it is called.  Cache the result if
          calling repeatedly with the same ``Z``.

    References:
        * Scholkopf, B., Smola, A. J., and Muller, K.-R. (1998).
          "Nonlinear Component Analysis as a Kernel Eigenvalue Problem."
          *Neural Computation*, 10(5), 1299--1319.  (Kernel PCA.)
        * Bakir, G., Weston, J., and Scholkopf, B. (2004). "Learning to
          Find Pre-Images." *NeurIPS*.  (Kernel ridge regression
          pre-image reconstruction.)
        * Cortes, C., Mohri, M., and Rostamizadeh, A. (2012).
          "Algorithms for Learning Kernels Based on Centered Alignment."
          *JMLR*, 13, 795--828.  (CKA definition and properties.)
        * Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning
          in Python." *JMLR*, 12, 2825--2830.  (``KernelPCA`` centering
          pattern and out-of-sample extension.)
        * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for
          Stochastic Optimization." *ICLR*.  (Adam optimizer.)

    See Also:
        :class:`FairPCA`: Primal-form (linear) fair PCA; operates in
            :math:`\mathbb{R}^d` rather than the RKHS.
        :class:`FairKernelRidge`: Dual-form fair regression using the
            same kernel machinery.
        :func:`~fairkl.ops.centering.center_kernel`: Kernel centering
            function used internally.
        :func:`~fairkl.kernels.exact.rbf_kernel`: The RBF kernel
            function.
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
            \mathcal{L}(V) =
              \underbrace{- \frac{1}{n}\|K_c V\|_F^2}_{\text{neg. variance}}
              + \underbrace{\mu\, \mathrm{CKA}(K_c V, q)}_{\text{fairness}}
              + \underbrace{w_{\mathrm{ortho}}\, \|V^\top V - I_k\|_F^2}_{
                    \text{orthogonality}}

        **Algorithm steps:**

        1. Compute the kernel matrix :math:`K \in \mathbb{R}^{n
           \times n}`.
        2. Store centering statistics:
           :math:`\bar{K}_{\mathrm{col}}` (shape ``(1, n)``) and
           :math:`\bar{K}_{\mathrm{total}}` (scalar).
        3. Center the kernel: :math:`K_c = H K H`.
        4. Initialize :math:`V` (Glorot uniform, only on first call).
        5. For ``epochs`` iterations: compute gradients
           :math:`\nabla_V \mathcal{L}` via ``jax.value_and_grad`` and
           apply Adam update.

        **Centering statistics stored** (for out-of-sample
        :meth:`transform`):

        * ``_K_col_mean``: shape ``(1, n)`` -- column means of the
          training kernel matrix :math:`K`.
        * ``_K_total_mean``: scalar -- grand mean of the training
          kernel matrix :math:`K`.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs where
                ``n`` is the number of samples and ``d`` is the feature
                dimensionality.  Converted to float32.  Stored as
                ``self._X_train`` for cross-kernel computation.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.  When ``None`` or when ``mu=0``, the
                fairness penalty is omitted.  Automatically expanded to
                ``(n, 1)`` if 1-D.
            epochs (int): Number of full-batch Adam epochs.  Typical
                values: 200--500.  Default ``200``.
            lr (float): Adam learning rate.  Typical values: ``1e-3``
                to ``1e-2``.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility with
                scikit-learn-style interfaces.

        Returns:
            FairKernelPCA: The fitted model instance (``self``), for
            method chaining (e.g.
            ``Z = model.fit(X, q=q).transform(X_test)``).

        Examples:
            >>> model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.5)
            >>> model.fit(X_train, q=q, epochs=200, lr=1e-2)
            >>> Z = model.transform(X_test)

        Note:
            * The kernel matrix :math:`K` is computed once
              (:math:`O(n^2 d)`) and centered (:math:`O(n^2)`).  Each
              epoch then costs :math:`O(n^2 k)` for the projection
              gradient.
            * ``V`` is initialized only on the first call to ``fit``;
              subsequent calls reuse the existing ``V`` as a warm start,
              which can be useful for iterating over fairness penalty
              weights.
            * The orthogonality penalty weight is fixed at ``10.0``
              internally.
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

        Implements the scikit-learn ``KernelCenterer.transform`` formula
        (see ``sklearn.preprocessing.KernelCenterer``):

        .. math::
            \tilde{K} = K_{\text{cross}}
              - \bar{K}_{\text{col}}
              - \mathrm{rowmean}(K_{\text{cross}})
              + \bar{K}_{\text{total}}

        where :math:`\bar{K}_{\text{col}} \in \mathbb{R}^{1 \times n}`
        and :math:`\bar{K}_{\text{total}} \in \mathbb{R}` are the
        column means and grand mean of the *training* kernel matrix,
        stored during :meth:`fit`.  The row mean is computed from the
        cross-kernel itself.

        This operation is equivalent to applying the double-centering
        :math:`\tilde{K} = H_{\text{test}} K_{\text{cross}}
        H_{\text{train}}` without explicitly constructing the
        centering matrices.

        Args:
            K_cross (Tensor of shape ``(m, n)``): Cross-kernel matrix
                between ``m`` test points and ``n`` training points.

        Returns:
            Tensor of shape ``(m, n)`` with dtype float32 -- the
            centered cross-kernel matrix.
        """
        row_mean = ops.mean(K_cross, axis=1, keepdims=True)  # (m, 1)
        return K_cross - self._K_col_mean - row_mean + self._K_total_mean

    def transform(self, X):
        r"""Project new data into the fair kernel PCA space.

        Computes the cross-kernel matrix
        :math:`K_{\text{cross}} = k(X_{\text{test}}, X_{\text{train}})
        \in \mathbb{R}^{m \times n}` and centers it using the stored
        training statistics (column means and global mean from
        :meth:`fit`).  The centered cross-kernel is then multiplied by
        the dual coefficient matrix :math:`V`:

        .. math::
            Z = \tilde{K}_{\text{cross}}\, V

        where :math:`\tilde{K} \in \mathbb{R}^{m \times n}` is the
        centered cross-kernel.  This avoids recomputing or storing the
        full ``(n, n)`` training kernel matrix.

        Args:
            X (array-like of shape ``(m, d)``): Test inputs where ``d``
                must match the training feature dimensionality.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32
            containing the kernel PCA projections.  Each row is the
            ``k``-dimensional representation of one test sample.

        Raises:
            AttributeError: If called before :meth:`fit` (centering
                statistics not available).

        Examples:
            >>> model.fit(X_train, q=q, epochs=200)
            >>> Z_train = model.transform(X_train)   # (n, k)
            >>> Z_new = model.transform(X_new)       # (m, k)

        Note:
            * The centering formula follows the scikit-learn
              ``KernelCenterer.transform`` convention:

              .. math::
                  \tilde{K} = K_{\text{cross}}
                    - \bar{K}_{\text{col}}
                    - \mathrm{rowmean}(K_{\text{cross}})
                    + \bar{K}_{\text{total}}

            * **Complexity:** :math:`O(m \cdot n \cdot d)` to build the
              cross-kernel, plus :math:`O(m \cdot n \cdot k)` for the
              matrix product :math:`\tilde{K} V`.
        """
        X = ops.convert_to_tensor(X)
        K_cross = self._compute_kernel(X, self._X_train)  # (m, n)
        K_cross_c = self._center_cross_kernel(K_cross)
        return ops.matmul(K_cross_c, self._V)

    def inverse_transform(self, Z):
        r"""Approximate pre-image via kernel ridge regression.

        Reconstructs inputs from their kernel PCA projections by
        learning a mapping from the low-dimensional projection space
        back to the original input space.  The pre-image problem is
        inherently ill-posed for non-linear kernels (a point in RKHS
        may not correspond to any point in input space), so kernel
        ridge regression is used as an approximate solution following
        Bakir et al. (2004).

        **Algorithm:**

        1. Recompute training kernel :math:`K` and center it:
           :math:`K_c = HKH`.
        2. Compute training projections:
           :math:`Z_{\text{train}} = K_c V \in \mathbb{R}^{n \times k}`.
        3. Build the linear kernel in projection space:
           :math:`K_Z = Z_{\text{train}} Z_{\text{train}}^\top
           \in \mathbb{R}^{n \times n}`.
        4. Solve the ridge system for dual coefficients:
           :math:`C = (K_Z + \alpha I)^{-1} X_{\text{train}}
           \in \mathbb{R}^{n \times d}`.
        5. Map new projections to input space:
           :math:`\hat{X} = Z Z_{\text{train}}^\top C
           \in \mathbb{R}^{m \times d}`.

        Args:
            Z (array-like of shape ``(m, n_components)``): Projections
                as returned by :meth:`transform`.

        Returns:
            Tensor of shape ``(m, d)`` with dtype float32 containing
            the approximate reconstructed inputs.

        Raises:
            AttributeError: If called before :meth:`fit`.

        Examples:
            >>> model.fit(X_train, q=q, epochs=200)
            >>> Z = model.transform(X_train)
            >>> X_hat = model.inverse_transform(Z)  # shape (n, d)

        Note:
            * This method recomputes the training kernel and
              projections each time it is called, costing
              :math:`O(n^2 d + n^3)`.  Cache the result if calling
              repeatedly with the same ``Z``.
            * The ``alpha`` constructor parameter controls the ridge
              regularization of the pre-image regression.  Too-small
              values may lead to numerical instability (ill-conditioned
              system); too-large values produce overly smooth
              reconstructions.
            * The pre-image quality depends on the kernel and the number
              of components.  For RBF kernels with many components, the
              reconstruction can be quite accurate.

        References:
            * Bakir, G., Weston, J., and Scholkopf, B. (2004).
              "Learning to Find Pre-Images." *NeurIPS*.
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
        ``model.transform(X)``, but more efficient because it reuses
        the centered training kernel :math:`K_c` directly (computed
        during ``fit``) instead of recomputing the cross-kernel and
        centering it via the stored statistics.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.
            **kwargs: Forwarded to :meth:`fit` (e.g. ``epochs``,
                ``lr``).

        Returns:
            Tensor of shape ``(n, n_components)`` with dtype float32
            containing the kernel PCA projections of the training data.

        Examples:
            >>> model = FairKernelPCA(n_components=3, sigma=1.0, mu=0.5)
            >>> Z = model.fit_transform(X, q=q, epochs=200, lr=1e-2)
            >>> Z.shape  # (n, 3)
        """
        self.fit(X, q=q, **kwargs)
        K = self._compute_kernel(X)
        K_c = center_kernel(K)
        return ops.matmul(K_c, self._V)

    def call(self, X):
        """Forward pass (delegates to :meth:`transform`).

        This method satisfies the ``keras.Model`` interface, allowing
        the model to be used in Keras pipelines and with ``model(X)``
        syntax.  It is functionally identical to :meth:`transform`.

        Args:
            X (array-like of shape ``(m, d)``): Input data.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32.
        """
        return self.transform(X)

    def get_config(self):
        """Return the model configuration as a serializable dict.

        Extends the base ``keras.Model.get_config`` with all
        constructor arguments (``n_components``, ``sigma``, ``mu``,
        ``sigma_q``, ``kernel``, ``alpha``), enabling reconstruction
        via ``keras.Model.from_config``.

        Returns:
            dict: Configuration dictionary.  Keys include all base-class
            config entries plus the six constructor parameters.
        """
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
