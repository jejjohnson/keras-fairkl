"""Fair PCA.

Solves the fairness-regularized principal component analysis problem::

    max_V  tr(V^T X_c^T X_c V) - mu * CKA(X_c V, q)
    s.t.   V^T V = I

where ``X_c`` is the column-centered data matrix, ``V`` is the ``(d, k)``
projection matrix, ``mu`` controls the CKA fairness penalty, and ``q`` encodes
sensitive attributes.  The orthogonality constraint ``V^T V = I`` is relaxed
via a quadratic penalty ``ortho_weight * ||V^T V - I||_F^2`` and enforced
during gradient-descent optimization with the Adam optimizer.
"""

from __future__ import annotations

import jax
import keras
import keras.ops as ops

from fairkl.metrics.cka import cka_rbf


class FairPCA(keras.Model):
    r"""Fair PCA with a CKA fairness penalty and relaxed orthogonality.

    Finds a linear projection :math:`V \in \mathbb{R}^{d \times k}`
    that maximizes explained variance while penalizing statistical
    dependence between the projected data and sensitive attributes.
    The training objective is:

    .. math::
        \max_{V}\;
            \frac{1}{n}\,\mathrm{tr}\!\bigl(V^\top X_c^\top X_c V\bigr)
          - \mu\, \mathrm{CKA}(X_c V,\; q)
        \quad\text{s.t.}\quad V^\top V = I_k

    where :math:`X_c = X - \bar{X}` is the column-centered data matrix,
    :math:`k` is ``n_components``, and :math:`\mu` controls the
    fairness penalty.  The hard orthogonality constraint
    :math:`V^\top V = I` is relaxed via a quadratic penalty:

    .. math::
        \mathcal{L}_{\mathrm{ortho}} =
            \mathrm{ortho\_weight} \cdot \|V^\top V - I_k\|_F^2

    The combined loss (negated variance + fairness + orthogonality
    penalty) is minimized with the Adam optimizer.

    Args:
        n_components (int): Number of principal components ``k`` to
            retain.  Must be positive and at most ``d`` (the input
            dimensionality).
        mu (float): CKA fairness penalty weight :math:`\mu`.  ``mu=0``
            recovers standard PCA (up to the relaxed orthogonality).
            Larger values enforce stronger statistical independence
            between the projections and sensitive attributes.  Default
            ``1.0``.
        sigma_f (float): RBF bandwidth for the projection kernel inside
            CKA.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute kernel
            inside CKA.  Default ``1.0``.
        ortho_weight (float): Weight of the orthogonality penalty
            :math:`\|V^\top V - I\|_F^2`.  Should be large enough to
            enforce near-orthogonality but not so large as to dominate
            the variance objective.  Default ``10.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__``.

    Examples:
        Fair PCA with ``fit_transform``:

        >>> import numpy as np
        >>> from fairkl.models import FairPCA
        >>> X = np.random.randn(200, 10).astype("float32")
        >>> q = np.random.randn(200, 1).astype("float32")
        >>> model = FairPCA(n_components=3, mu=0.5)
        >>> Z = model.fit_transform(X, q=q, epochs=300, lr=1e-2)
        >>> Z.shape  # (200, 3)
        >>> ratios = model.explained_variance_ratio(X)  # per-component
        >>> err = model.reconstruction_error(X)  # relative error

    Note:
        * Data is column-centered internally before computing variance
          and projections, consistent with standard PCA.
        * The projection matrix ``V`` is initialized with an orthogonal
          initializer, which provides a good starting point.
        * Unlike eigendecomposition-based PCA, this gradient-based
          formulation does not guarantee globally optimal variance
          capture, but it allows incorporating the non-linear CKA
          penalty.
        * Computational complexity per epoch is :math:`O(n \cdot d \cdot k)`
          for the projection plus :math:`O(n^2)` for the CKA term.

    References:
        * Jolliffe, I. T. (2002). *Principal Component Analysis*. 2nd
          ed. Springer.  (Standard PCA.)
        * Cortes, C. et al. (2012). "Algorithms for Learning Kernels
          Based on Centered Alignment." *JMLR*, 13, 795--828.  (CKA.)
        * Wen, Z. and Yin, W. (2013). "A Feasible Method for
          Optimization with Orthogonality Constraints."
          *Math. Program.*, 142, 397--434.  (Orthogonality penalties.)

    See Also:
        :class:`FairKernelPCA`: Fair PCA in the RKHS (kernel version).
        :class:`FairLinear`: Fair linear regression / classification.
    """

    def __init__(
        self,
        n_components: int,
        mu: float = 1.0,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        ortho_weight: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mu = mu
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self.ortho_weight = ortho_weight
        self._V = None

    def build(self, input_shape):
        d = input_shape[-1]
        self._V = self.add_weight(
            name="V",
            shape=(d, self.n_components),
            initializer="orthogonal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X):
        return ops.matmul(X, self._V)

    def transform(self, X):
        r"""Project data into the learned fair subspace.

        Computes the low-dimensional representation :math:`Z = X V`
        where :math:`V \in \mathbb{R}^{d \times k}` is the projection
        matrix learned during ``fit``.

        Args:
            X (array-like): Input data of shape ``(m, d)``.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32
            containing the projected data.

        Note:
            For proper PCA semantics, ``X`` should be centered using
            the same column means as the training data.  This method
            does not re-center automatically.
        """
        X = ops.convert_to_tensor(X)
        return self.call(X)

    def inverse_transform(self, Z):
        r"""Reconstruct data from the projected representation.

        Computes the approximate reconstruction
        :math:`\hat{X} = Z V^\top` by projecting back from the
        ``n_components``-dimensional subspace to the original
        ``d``-dimensional space.

        Args:
            Z (array-like): Projected data of shape
                ``(m, n_components)``.

        Returns:
            Tensor of shape ``(m, d)`` with dtype float32 containing
            the reconstructed data.  When ``n_components < d`` this is
            a lossy reconstruction.
        """
        Z = ops.convert_to_tensor(Z)
        return ops.matmul(Z, ops.transpose(self._V))

    def fit_transform(self, X, q=None, **kwargs):
        """Fit the model and project the training data in one call.

        Equivalent to ``model.fit(X, q=q, **kwargs); model.transform(X)``
        but avoids redundant computation.

        Args:
            X (array-like): Training inputs of shape ``(n, d)``.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.
            **kwargs: Forwarded to :meth:`fit` (e.g. ``epochs``, ``lr``).

        Returns:
            Tensor of shape ``(n, n_components)`` with dtype float32
            containing the projected training data.
        """
        self.fit(X, q=q, **kwargs)
        return self.transform(X)

    def explained_variance_ratio(self, X):
        r"""Proportion of variance explained by each component.

        Computes the ratio of the variance captured by each of the
        ``n_components`` projection directions to the total variance
        of the centered data:

        .. math::
            \mathrm{ratio}_j =
                \frac{\|X_c\, v_j\|^2}{\|X_c\|_F^2}

        where :math:`v_j` is the ``j``-th column of ``V`` and
        :math:`X_c` is the column-centered input.

        Args:
            X (array-like): Input data of shape ``(n, d)``.  Column
                centering is applied internally.

        Returns:
            Tensor of shape ``(n_components,)`` with dtype float32.
            Each entry is in :math:`[0, 1]` and the entries sum to
            at most 1 (equality when ``n_components = d``).
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        X_c = X - ops.mean(X, axis=0, keepdims=True)
        Z = ops.matmul(X_c, self._V)
        Z_c = Z - ops.mean(Z, axis=0, keepdims=True)
        var_components = ops.sum(Z_c * Z_c, axis=0)
        total_var = ops.sum(X_c * X_c)
        return var_components / (total_var + 1e-10)

    def reconstruction_error(self, X):
        r"""Relative reconstruction error from the PCA projection.

        Measures how much information is lost by projecting to
        ``n_components`` dimensions and reconstructing:

        .. math::
            \mathrm{error} =
                \frac{\|X_c - X_c V V^\top\|_F^2}{\|X_c\|_F^2}

        A value of 0 indicates perfect reconstruction; a value near 1
        indicates nearly all variance is lost.

        Args:
            X (array-like): Input data of shape ``(n, d)``.  Column
                centering is applied internally.

        Returns:
            Scalar tensor (float32) with the relative reconstruction
            error in :math:`[0, 1]`.
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        X_c = X - ops.mean(X, axis=0, keepdims=True)
        Z = ops.matmul(X_c, self._V)
        X_hat = ops.matmul(Z, ops.transpose(self._V))
        return ops.sum((X_c - X_hat) ** 2) / (ops.sum(X_c * X_c) + 1e-10)

    def fit(self, X, q=None, epochs: int = 200, lr: float = 1e-2, **kwargs):
        r"""Learn the fair projection matrix :math:`V`.

        Minimizes the combined loss (negated variance + CKA fairness
        penalty + orthogonality penalty) using full-batch Adam gradient
        descent.  The data is column-centered internally before
        optimization.

        The per-epoch loss is:

        .. math::
            \mathcal{L} =
              - \frac{1}{n}\|X_c V\|_F^2
              + \mu\, \mathrm{CKA}(X_c V, q)
              + w_{\mathrm{ortho}}\, \|V^\top V - I_k\|_F^2

        Args:
            X (array-like): Training inputs of shape ``(n, d)`` where
                ``n`` is the number of samples and ``d`` is the number
                of features.  Converted to float32.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.  When ``None`` or when ``mu=0``, the
                fairness penalty term is omitted.
            epochs (int): Number of full-batch Adam epochs.  Default
                ``200``.
            lr (float): Adam learning rate.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            self: The fitted model instance (for method chaining).

        Note:
            The model is built lazily on the first call.  ``V`` is
            initialized with the Keras ``"orthogonal"`` initializer,
            which provides a near-feasible starting point for the
            :math:`V^\top V = I` constraint.
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        if q is not None:
            q = ops.convert_to_tensor(q, dtype="float32")
            if len(ops.shape(q)) == 1:
                q = ops.expand_dims(q, axis=-1)

        if not self.built:
            self(X[:1])

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        optimizer.build([self._V])

        n_comp = self.n_components
        mu = self.mu
        sigma_f = self.sigma_f
        sigma_q = self.sigma_q
        ortho_w = self.ortho_weight
        n_float = ops.cast(ops.shape(X)[0], "float32")
        # Center the data for proper PCA
        X_c = X - ops.mean(X, axis=0, keepdims=True)

        def loss_fn(V_val):
            Z = ops.matmul(X_c, V_val)
            # Maximize variance => minimize negative variance
            variance = ops.sum(Z * Z) / n_float
            loss = -variance
            # Fairness penalty
            if q is not None and mu > 0:
                loss = loss + mu * cka_rbf(Z, q, sigma_f=sigma_f, sigma_q=sigma_q)
            # Orthogonality penalty: ||V^T V - I||_F^2
            VtV = ops.matmul(ops.transpose(V_val), V_val)
            eye = ops.eye(n_comp)
            ortho_penalty = ops.sum((VtV - eye) ** 2)
            loss = loss + ortho_w * ortho_penalty
            return loss

        for _ in range(epochs):
            _loss_val, grads = jax.value_and_grad(loss_fn)(self._V.value)
            optimizer.apply([grads], [self._V])

        return self

    def get_config(self):
        config = super().get_config()
        config.update(
            n_components=self.n_components,
            mu=self.mu,
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            ortho_weight=self.ortho_weight,
        )
        return config
