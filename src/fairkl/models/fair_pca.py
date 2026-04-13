"""Fair PCA.

Solves the fairness-regularized principal component analysis problem::

    max_V  tr(V^T X_c^T X_c V) - mu * CKA(X_c V, q)
    s.t.   V^T V = I

where ``X_c`` is the ``(n, d)`` column-centered data matrix, ``V`` is the
``(d, k)`` projection matrix (``k = n_components``), ``mu >= 0`` controls
the CKA fairness penalty, and ``q`` encodes sensitive attributes.

The hard orthogonality constraint ``V^T V = I`` is relaxed via a quadratic
penalty ``ortho_weight * ||V^T V - I||_F^2`` and enforced during
gradient-descent optimization with the Adam optimizer.  This relaxation
converts the Stiefel-manifold constrained problem into an unconstrained
one, at the cost of approximate (rather than exact) orthogonality.

The projection ``Z = X_c V`` maps data from ``R^d`` to ``R^k``, and the
reconstruction ``X_hat = Z V^T`` maps back.  When ``mu=0`` and the
orthogonality penalty is strong enough, the solution converges to the
top-``k`` eigenvectors of ``X_c^T X_c`` (standard PCA).

See :class:`FairKernelPCA` for the kernelized (RKHS) variant.
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
            \underbrace{
                \frac{1}{n}\,\mathrm{tr}\!\bigl(V^\top X_c^\top X_c V\bigr)
            }_{\text{explained variance}}
          - \underbrace{
                \mu\, \mathrm{CKA}(X_c V,\; q)
            }_{\text{fairness penalty}}
        \quad\text{s.t.}\quad V^\top V = I_k

    where :math:`X_c = X - \bar{X}` is the column-centered data matrix
    (centering is applied internally), :math:`k` is ``n_components``,
    and :math:`\mu \ge 0` controls the fairness penalty.

    **Orthogonality via penalty relaxation.** The hard constraint
    :math:`V^\top V = I_k` (which restricts :math:`V` to the Stiefel
    manifold :math:`\mathrm{St}(k, d)`) is relaxed via a quadratic
    penalty:

    .. math::
        \mathcal{L}_{\mathrm{ortho}} =
            w_{\mathrm{ortho}} \cdot \|V^\top V - I_k\|_F^2

    The combined loss (negated variance + fairness + orthogonality
    penalty) is minimized with full-batch Adam gradient descent:

    .. math::
        \mathcal{L}(V) =
          - \frac{1}{n}\|X_c V\|_F^2
          + \mu\, \mathrm{CKA}(X_c V, q)
          + w_{\mathrm{ortho}} \|V^\top V - I_k\|_F^2

    **Relationship to standard PCA.** When ``mu=0`` and the
    orthogonality penalty is large enough to enforce
    :math:`V^\top V \approx I`, the solution converges to the
    top-:math:`k` eigenvectors of :math:`\tfrac{1}{n} X_c^\top X_c`
    (i.e. standard PCA up to optimization tolerance).

    Args:
        n_components (int): Number of principal components ``k`` to
            retain.  Must satisfy ``1 <= n_components <= d`` where ``d``
            is the input dimensionality.
        mu (float): CKA fairness penalty weight :math:`\mu \ge 0`.
            ``mu=0`` recovers standard PCA (up to relaxed
            orthogonality).  Larger values enforce stronger statistical
            independence between the projections and sensitive
            attributes.  Typical values range from ``0.01`` to ``10.0``.
            Default ``1.0``.
        sigma_f (float): RBF bandwidth for the projection kernel inside
            CKA.  Controls the length scale at which projection
            similarity is measured.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute
            kernel inside CKA.  Controls the length scale at which
            sensitive-attribute similarity is measured.  Default ``1.0``.
        ortho_weight (float): Weight :math:`w_{\mathrm{ortho}}` of the
            orthogonality penalty :math:`\|V^\top V - I\|_F^2`.  Should
            be large enough to enforce near-orthogonality but not so
            large as to dominate the variance objective.  Typical values:
            ``1.0`` to ``100.0``.  Default ``10.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__`` (e.g. ``name``, ``dtype``).

    Attributes:
        n_components (int): Number of retained components.
        mu (float): Stored CKA penalty weight.
        sigma_f (float): Stored projection-kernel RBF bandwidth.
        sigma_q (float): Stored sensitive-attribute RBF bandwidth.
        ortho_weight (float): Stored orthogonality penalty weight.
        _V (Variable or None): Projection matrix of shape ``(d, k)``,
            allocated during :meth:`build`.  Initialized with the
            Keras ``"orthogonal"`` initializer.

    Examples:
        Fair PCA with ``fit_transform``:

        >>> import numpy as np
        >>> from fairkl.models import FairPCA
        >>> X = np.random.randn(200, 10).astype("float32")
        >>> q = np.random.randn(200, 1).astype("float32")
        >>> model = FairPCA(n_components=3, mu=0.5)
        >>> Z = model.fit_transform(X, q=q, epochs=300, lr=1e-2)
        >>> Z.shape  # (200, 3)

        Diagnostic methods:

        >>> ratios = model.explained_variance_ratio(X)  # (3,)
        >>> err = model.reconstruction_error(X)  # scalar in [0, 1]

        Standard PCA (``mu=0``):

        >>> model = FairPCA(n_components=3, mu=0.0)
        >>> Z = model.fit_transform(X, epochs=200)

        Reconstruction round-trip:

        >>> Z = model.transform(X)               # (200, 3)
        >>> X_hat = model.inverse_transform(Z)   # (200, 10)

    Note:
        * Data is column-centered internally (``X_c = X - mean(X)``)
          before computing variance and projections, consistent with
          standard PCA.  The centering is done once at the start of
          :meth:`fit`.
        * The projection matrix ``V`` is initialized with the Keras
          ``"orthogonal"`` initializer, which provides a near-feasible
          starting point on the Stiefel manifold.
        * Unlike eigendecomposition-based PCA, this gradient-based
          formulation does not guarantee globally optimal variance
          capture, but it allows incorporating the non-linear CKA
          penalty.  For ``mu=0``, prefer ``sklearn.decomposition.PCA``
          if exact solutions are needed.
        * **Computational complexity per epoch:**
          :math:`O(n \cdot d \cdot k)` for the projection
          :math:`Z = X_c V`, plus :math:`O(d \cdot k^2)` for the
          orthogonality penalty :math:`V^\top V`, plus :math:`O(n^2)`
          for the CKA term (when ``mu > 0``).

    References:
        * Jolliffe, I. T. (2002). *Principal Component Analysis*. 2nd
          ed. Springer.  (Standard PCA theory.)
        * Cortes, C., Mohri, M., and Rostamizadeh, A. (2012).
          "Algorithms for Learning Kernels Based on Centered Alignment."
          *JMLR*, 13, 795--828.  (CKA definition and properties.)
        * Wen, Z. and Yin, W. (2013). "A Feasible Method for
          Optimization with Orthogonality Constraints."
          *Math. Program.*, 142, 397--434.  (Orthogonality penalties
          and Stiefel manifold optimization.)
        * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for
          Stochastic Optimization." *ICLR*.  (Adam optimizer.)

    See Also:
        :class:`FairKernelPCA`: Fair PCA in the RKHS (kernel version);
            handles non-linear structure via kernel functions.
        :class:`FairLinear`: Fair linear regression / classification.
        :func:`~fairkl.metrics.cka.cka_rbf`: The CKA function used in
            the fairness penalty.
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
        """Allocate the projection matrix ``V``.

        Called automatically on the first :meth:`fit` or :meth:`call`.
        Creates one trainable Keras ``Variable``:

        * ``_V``: shape ``(d, n_components)``, orthogonal initialization.

        The orthogonal initializer provides a starting point that
        already satisfies :math:`V^\top V \approx I`, reducing the
        number of epochs needed for the orthogonality penalty to
        converge.

        Args:
            input_shape (tuple): Shape tuple ``(..., d)`` where ``d``
                is the input feature dimensionality.  Only the last
                dimension is used.
        """
        d = input_shape[-1]
        self._V = self.add_weight(
            name="V",
            shape=(d, self.n_components),
            initializer="orthogonal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X):
        r"""Compute the linear projection :math:`Z = X V`.

        This is the core forward pass.  :meth:`transform` delegates
        to this method.

        Args:
            X (array-like of shape ``(m, d)``): Input data.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32.
        """
        return ops.matmul(X, self._V)

    def transform(self, X):
        r"""Project data into the learned fair subspace.

        Computes the low-dimensional representation :math:`Z = X V`
        where :math:`V \in \mathbb{R}^{d \times k}` is the projection
        matrix learned during :meth:`fit`.

        Args:
            X (array-like of shape ``(m, d)``): Input data where ``d``
                must match the training feature dimensionality.

        Returns:
            Tensor of shape ``(m, n_components)`` with dtype float32
            containing the projected data.  Each row is the
            ``k``-dimensional representation of one sample.

        Examples:
            >>> model.fit(X_train, q=q, epochs=200)
            >>> Z_train = model.transform(X_train)   # (n, k)
            >>> Z_test = model.transform(X_test)     # (m, k)

        Note:
            For proper PCA semantics, ``X`` should be centered using
            the same column means as the training data.  This method
            does **not** re-center automatically -- the user is
            responsible for centering test data with the training mean
            when using :meth:`transform` on new data.
        """
        X = ops.convert_to_tensor(X)
        return self.call(X)

    def inverse_transform(self, Z):
        r"""Reconstruct data from the projected representation.

        Computes the approximate reconstruction
        :math:`\hat{X} = Z V^\top` by projecting back from the
        ``n_components``-dimensional subspace to the original
        ``d``-dimensional space.

        When :math:`V` is exactly orthogonal (:math:`V^\top V = I`),
        the composition
        :math:`X \mapsto XV \mapsto XV V^\top` is the orthogonal
        projection onto the column space of :math:`V`.  The
        reconstruction error
        :math:`\|X - X V V^\top\|_F^2` equals the variance in the
        discarded components.

        Args:
            Z (array-like of shape ``(m, n_components)``): Projected
                data as returned by :meth:`transform`.

        Returns:
            Tensor of shape ``(m, d)`` with dtype float32 containing
            the reconstructed data.  When ``n_components < d`` this is
            a lossy reconstruction; see
            :meth:`reconstruction_error` for quantification.

        Examples:
            >>> Z = model.transform(X)               # (m, k)
            >>> X_hat = model.inverse_transform(Z)   # (m, d)
        """
        Z = ops.convert_to_tensor(Z)
        return ops.matmul(Z, ops.transpose(self._V))

    def fit_transform(self, X, q=None, **kwargs):
        """Fit the model and project the training data in one call.

        Equivalent to ``model.fit(X, q=q, **kwargs)`` followed by
        ``model.transform(X)``, but expressed as a single convenience
        call consistent with the scikit-learn API.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.
            **kwargs: Forwarded to :meth:`fit` (e.g. ``epochs``,
                ``lr``).

        Returns:
            Tensor of shape ``(n, n_components)`` with dtype float32
            containing the projected training data.

        Examples:
            >>> model = FairPCA(n_components=3, mu=0.5)
            >>> Z = model.fit_transform(X, q=q, epochs=300, lr=1e-2)
            >>> Z.shape  # (n, 3)
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

        where :math:`v_j` is the ``j``-th column of :math:`V` and
        :math:`X_c = X - \bar{X}` is the column-centered input.  This
        is analogous to ``sklearn.decomposition.PCA.explained_variance_ratio_``.

        Args:
            X (array-like of shape ``(n, d)``): Input data.  Column
                centering is applied internally.

        Returns:
            Tensor of shape ``(n_components,)`` with dtype float32.
            Each entry is in :math:`[0, 1]` and the entries sum to
            at most 1 (equality when ``n_components = d`` and
            :math:`V` is exactly orthogonal).

        Examples:
            >>> ratios = model.explained_variance_ratio(X)  # (k,)
            >>> total = float(np.sum(ratios))  # total variance retained

        Note:
            * A small epsilon (``1e-10``) is added to the denominator
              to avoid division by zero for constant data.
            * The ratios are computed from the centered projected data,
              so the centering of :math:`Z` ensures consistency even
              when :math:`V` does not pass exactly through the origin.
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

        A value of 0 indicates perfect reconstruction (lossless); a
        value near 1 indicates nearly all variance is lost.  The
        reconstruction error is complementary to the sum of the
        explained variance ratios:
        ``error + sum(explained_variance_ratio) ~ 1`` when :math:`V`
        is orthogonal.

        Args:
            X (array-like of shape ``(n, d)``): Input data.  Column
                centering is applied internally.

        Returns:
            Scalar tensor (float32) with the relative reconstruction
            error in :math:`[0, 1]`.

        Examples:
            >>> err = model.reconstruction_error(X)
            >>> print(f"Information lost: {float(err):.1%}")

        Note:
            A small epsilon (``1e-10``) is added to the denominator
            to avoid division by zero for constant data.
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
            \mathcal{L}(V) =
              \underbrace{- \frac{1}{n}\|X_c V\|_F^2}_{\text{neg. variance}}
              + \underbrace{\mu\, \mathrm{CKA}(X_c V, q)}_{\text{fairness}}
              + \underbrace{w_{\mathrm{ortho}}\, \|V^\top V - I_k\|_F^2}_{
                    \text{orthogonality}}

        Gradients :math:`\nabla_V \mathcal{L}` are computed via
        ``jax.value_and_grad`` and applied with Adam.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs where
                ``n`` is the number of samples and ``d`` is the number
                of features.  Converted to float32.  Column-centered
                internally: :math:`X_c = X - \bar{X}`.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.  When ``None`` or when ``mu=0``, the
                fairness penalty term is omitted.  Automatically
                expanded to ``(n, 1)`` if 1-D.
            epochs (int): Number of full-batch Adam epochs.  Typical
                values: 200--500.  Default ``200``.
            lr (float): Adam learning rate.  Typical values: ``1e-3``
                to ``1e-2``.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility with
                scikit-learn-style interfaces.

        Returns:
            FairPCA: The fitted model instance (``self``), for method
            chaining (e.g.
            ``Z = model.fit(X, q=q).transform(X)``).

        Examples:
            >>> model = FairPCA(n_components=3, mu=0.5)
            >>> model.fit(X_train, q=q, epochs=300, lr=1e-2)
            >>> Z = model.transform(X_train)

        Note:
            * The model is built lazily on the first call.  ``V`` is
              initialized with the Keras ``"orthogonal"`` initializer,
              which provides a near-feasible starting point for the
              :math:`V^\top V = I` constraint.
            * **Complexity per epoch:** :math:`O(n \cdot d \cdot k)` for
              the projection :math:`X_c V`, plus :math:`O(d \cdot k^2)`
              for the orthogonality penalty, plus :math:`O(n^2)` for the
              CKA term (when ``mu > 0``).
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
        """Return the model configuration as a serializable dict.

        Extends the base ``keras.Model.get_config`` with all
        constructor arguments (``n_components``, ``mu``, ``sigma_f``,
        ``sigma_q``, ``ortho_weight``), enabling reconstruction via
        ``keras.Model.from_config``.

        Returns:
            dict: Configuration dictionary.  Keys include all base-class
            config entries plus the five constructor parameters.
        """
        config = super().get_config()
        config.update(
            n_components=self.n_components,
            mu=self.mu,
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            ortho_weight=self.ortho_weight,
        )
        return config
