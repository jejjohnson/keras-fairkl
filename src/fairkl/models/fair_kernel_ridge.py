"""Fair kernel ridge regression.

Solves the fairness-regularized kernel ridge regression problem in dual
form via the representer theorem::

    min_alpha  ||K alpha - y||^2 + lambda * alpha^T K alpha
               + mu * CKA(K alpha, q)

where ``K`` is the ``(n, n)`` kernel (Gram) matrix, ``alpha`` is the
``(n, 1)`` vector of dual coefficients, ``lambda >= 0`` controls ridge
regularization, ``mu >= 0`` controls the CKA fairness penalty, and ``q``
encodes sensitive attributes.

The dual formulation arises from the representer theorem: the optimal
predictor in the RKHS can be written as
``f(x) = sum_i alpha_i k(x, x_i)``, so the primal ridge regression
problem reduces to optimizing over the ``n``-dimensional coefficient
vector ``alpha`` rather than a potentially infinite-dimensional weight
vector.

**Two training paths:**

* **Exact (mu=0):** The optimality condition yields the closed-form
  system ``(K + lambda I) alpha = y``, solved via Cholesky factorization
  (O(n^3)) or conjugate gradient (O(n^2 k), k iterations).
* **Fair (mu>0):** The CKA term is non-linear in ``alpha``, so gradient
  descent (Adam) is used instead.  The optimizer is *warm-started* from
  the exact (unfair) solution to accelerate convergence.

See :class:`FairLinear` for the primal-form (weight-space) analogue.
"""

from __future__ import annotations

import jax
import keras
import keras.ops as ops

from fairkl.kernels.exact import rbf_kernel
from fairkl.metrics.cka import cka_rbf


class FairKernelRidge(keras.Model):
    r"""Fair kernel ridge regression with a CKA fairness penalty.

    Solves the dual-form fairness-regularized kernel ridge regression
    problem via the representer theorem.  The learned predictor has the
    form :math:`f(x) = \sum_{i=1}^{n} \alpha_i\, k(x, x_i)` where
    :math:`\alpha \in \mathbb{R}^n` is the vector of dual coefficients
    and :math:`k` is the chosen kernel function.

    The training objective is:

    .. math::
        \min_{\alpha}\;
            \underbrace{\|K\alpha - y\|^2}_{\text{MSE}}
          + \underbrace{\lambda\, \alpha^\top K \alpha}_{\text{ridge}}
          + \underbrace{\mu\, \mathrm{CKA}(K\alpha,\; q)}_{\text{fairness}}

    where :math:`K \in \mathbb{R}^{n \times n}` is the kernel (Gram)
    matrix, :math:`\lambda \ge 0` is the ridge regularization strength,
    :math:`\mu \ge 0` is the CKA fairness penalty weight, and
    :math:`q \in \mathbb{R}^{n \times d_q}` encodes the sensitive
    attributes.

    **Dual formulation and the representer theorem.** Standard kernel
    ridge regression operates in a reproducing kernel Hilbert space
    (RKHS) :math:`\mathcal{H}`.  The representer theorem guarantees
    that the minimizer of the regularized empirical risk lies in the
    span of the training kernel evaluations, i.e.
    :math:`f^* = \sum_i \alpha_i\, k(\cdot, x_i)`.  Substituting
    into the primal objective and using
    :math:`f(x_j) = \sum_i \alpha_i k(x_j, x_i) = (K\alpha)_j` yields
    the finite-dimensional dual problem above.  The ridge term
    :math:`\alpha^\top K \alpha = \|f\|_{\mathcal{H}}^2` penalizes
    the RKHS norm of the predictor.

    **Two training paths:**

    * **Exact (mu=0):** When no fairness penalty is active, the
      first-order optimality condition
      :math:`2K(K\alpha - y) + 2\lambda K\alpha = 0` simplifies to the
      linear system :math:`(K + \lambda I)\alpha = y`, solved via
      Cholesky factorization or conjugate-gradient iteration.
    * **Fair (mu>0):** The CKA term is non-linear in :math:`\alpha`
      (it involves an RBF kernel on the predictions), so no closed-form
      solution exists.  Gradient descent (Adam) is used instead, and the
      optimizer is *warm-started* from the exact (unfair) solution
      :math:`\alpha_0 = (K + \lambda I)^{-1} y` to accelerate
      convergence.

    Args:
        sigma (float): RBF bandwidth for the feature kernel
            :math:`k(x, x') = \exp\!\bigl(-\|x - x'\|^2 / (2\sigma^2)\bigr)`.
            Controls the length scale of the kernel: smaller values give
            a more local (spiky) kernel, larger values a smoother one.
            Ignored when ``kernel="linear"``.  Default ``1.0``.
        lam (float): Ridge regularization strength :math:`\lambda \ge 0`.
            Larger values shrink the dual coefficients toward zero,
            reducing overfitting at the cost of higher bias.  Setting
            ``lam=0`` removes regularization (may cause numerical issues
            if :math:`K` is singular).  Default ``1e-3``.
        mu (float): CKA fairness penalty weight :math:`\mu \ge 0`.
            ``mu=0`` recovers standard (unfair) kernel ridge regression.
            Larger values enforce stronger statistical independence
            between predictions and sensitive attributes.  Typical
            values range from ``0.01`` to ``10.0`` depending on the
            data scale and desired fairness--accuracy trade-off.
            Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute
            kernel used inside CKA.  Controls the granularity of
            dependence measurement: smaller values detect finer-grained
            dependence.  Default ``1.0``.
        kernel (str): Kernel type for the feature space.  One of:

            - ``"rbf"`` -- Gaussian / squared-exponential kernel
              :math:`k(x,x') = \exp(-\|x-x'\|^2 / 2\sigma^2)`.
            - ``"linear"`` -- linear kernel
              :math:`k(x,x') = x^\top x'` (no bandwidth parameter).

            Default ``"rbf"``.
        solver (str): Linear solver for the exact (``mu=0``) path.  One
            of:

            - ``"cholesky"`` -- direct Cholesky factorization,
              :math:`O(n^3)` time and :math:`O(n^2)` memory.  Preferred
              for ``n < 10{,}000``.
            - ``"cg"`` -- iterative conjugate gradient,
              :math:`O(n^2 k)` time with ``k`` iterations.
              Memory-friendly for large ``n``.

            Ignored when ``mu > 0`` (the fair path always uses
            Cholesky for the warm-start initialization).  Default
            ``"cholesky"``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__`` (e.g. ``name``, ``dtype``).

    Attributes:
        sigma (float): Stored RBF bandwidth.
        lam (float): Stored ridge regularization strength.
        mu (float): Stored CKA penalty weight.
        sigma_q (float): Stored sensitive-attribute RBF bandwidth.
        kernel (str): Stored kernel type (``"rbf"`` or ``"linear"``).
        solver (str): Stored linear solver type.
        _X_train (Tensor or None): Training inputs stored after
            :meth:`fit`, shape ``(n, d)``.  Used by :meth:`predict`
            and :meth:`get_kernel_matrix` to compute cross-kernels.
        _alpha (Tensor or Variable or None): Dual coefficients after
            :meth:`fit`, shape ``(n, 1)``.  A plain tensor when
            ``mu=0``; a Keras ``Variable`` when ``mu>0``.

    Examples:
        Standard (unfair) kernel ridge regression (``mu=0``), using
        the closed-form Cholesky solver:

        >>> import numpy as np
        >>> from fairkl.models import FairKernelRidge
        >>> X = np.random.randn(100, 5).astype("float32")
        >>> y = X @ np.ones(5) + 0.1 * np.random.randn(100)
        >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
        >>> model.fit(X, y)
        >>> preds = model.predict(X[:10])  # shape (10, 1)
        >>> alpha = model.get_alpha()      # shape (100, 1)

        Fair kernel ridge regression (``mu>0``), using Adam gradient
        descent warm-started from the exact solution:

        >>> q = np.random.randn(100, 1).astype("float32")  # sensitive attr
        >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.5)
        >>> model.fit(X, y, q=q, epochs=200, lr=1e-3)
        >>> preds = model.predict(X[:10])  # shape (10, 1)

        Inspecting the kernel matrix for diagnostics:

        >>> K = model.get_kernel_matrix(X[:10])  # shape (10, 100)

    Note:
        * The CKA (Centered Kernel Alignment) penalty measures normalized
          statistical dependence between the model predictions and
          sensitive attributes via HSIC (Hilbert-Schmidt Independence
          Criterion).  CKA is bounded in :math:`[0, 1]` and invariant
          to isotropic scaling, making :math:`\mu` interpretable
          regardless of data scale.
        * **Computational complexity:**

          - Exact path: :math:`O(n^3)` for Cholesky, :math:`O(n^2 k)`
            for CG (``k`` iterations).
          - Fair path: :math:`O(n^3)` for the warm-start solve, then
            :math:`O(n^2)` per epoch for the CKA gradient (dominated by
            the kernel-vector product and the RBF kernel on predictions).
          - Prediction: :math:`O(m \cdot n \cdot d)` to build the
            cross-kernel plus :math:`O(m \cdot n)` for the
            matrix-vector product.

        * The Adam warm-start from the exact solution typically converges
          in far fewer epochs than random initialization, often 50--200
          epochs suffice.
        * Memory footprint is dominated by the ``(n, n)`` kernel matrix.
          For datasets with ``n > 10,000``, consider the ``"cg"`` solver
          or a Nystrom approximation.

    References:
        * Scholkopf, B. and Smola, A. J. (2002). *Learning with Kernels:
          Support Vector Machines, Regularization, Optimization, and
          Beyond*. MIT Press.  (Kernel ridge regression, representer
          theorem, Chapter 1--2.)
        * Cortes, C., Mohri, M., and Rostamizadeh, A. (2012). "Algorithms
          for Learning Kernels Based on Centered Alignment." *JMLR*, 13,
          795--828.  (CKA definition and properties.)
        * Bakir, G., Weston, J., and Scholkopf, B. (2004). "Learning to
          Find Pre-Images." *NeurIPS*.  (Kernel methods framework.)
        * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for Stochastic
          Optimization." *ICLR*.  (Adam optimizer used in the fair path.)

    See Also:
        :class:`FairLinear`: Primal-form (weight-space) fair linear model;
            scales as :math:`O(nd)` per epoch rather than :math:`O(n^2)`.
        :class:`FairKernelPCA`: Fair dimensionality reduction in the RKHS.
        :func:`~fairkl.metrics.cka.cka_rbf`: The CKA function used in the
            fairness penalty.
        :func:`~fairkl.kernels.exact.rbf_kernel`: The RBF kernel function.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        lam: float = 1e-3,
        mu: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        solver: str = "cholesky",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.lam = lam
        self.mu = mu
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.solver = solver
        self._X_train = None
        self._alpha = None

    def _compute_kernel(self, X, Y=None):
        if self.kernel == "linear":
            from fairkl.kernels.exact import linear_kernel

            return linear_kernel(X, Y)
        return rbf_kernel(X, Y, sigma=self.sigma)

    def fit(self, X, y, q=None, epochs: int = 100, lr: float = 1e-2, **kwargs):
        r"""Fit the model to training data.

        Computes the kernel matrix :math:`K \in \mathbb{R}^{n \times n}`
        and solves for the dual coefficients
        :math:`\alpha \in \mathbb{R}^{n \times 1}`.  The training
        inputs ``X`` are stored internally for use by :meth:`predict`
        and :meth:`get_kernel_matrix`.

        **Exact path (mu=0 or q is None):**

        Solves the linear system :math:`(K + \lambda I)\,\alpha = y`
        via Cholesky factorization (default) or conjugate gradient.
        The ``epochs`` and ``lr`` arguments are ignored.  This is
        equivalent to standard kernel ridge regression:

        .. math::
            \alpha^* = (K + \lambda I)^{-1} y

        **Fair path (mu>0 and q is not None):**

        1. **Warm-start:** compute the exact (unfair) solution
           :math:`\alpha_0 = (K + \lambda I)^{-1} y` via Cholesky.
        2. **Initialize** a Keras ``Variable`` from :math:`\alpha_0`.
        3. **Optimize** with Adam gradient descent on the full objective

           .. math::
               \mathcal{L}(\alpha) =
                   \frac{1}{n}\|K\alpha - y\|^2
                 + \lambda\, \alpha^\top K \alpha
                 + \mu\, \mathrm{CKA}(K\alpha, q)

           for ``epochs`` iterations.  Gradients are computed via
           ``jax.value_and_grad``.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs where
                ``n`` is the number of samples and ``d`` is the feature
                dimensionality.  Internally converted to a float32
                tensor.  Stored as ``self._X_train`` for out-of-sample
                prediction.
            y (array-like of shape ``(n,)`` or ``(n, 1)``): Target
                values.  Automatically expanded to ``(n, 1)`` if 1-D.
                Converted to float32.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.  When ``None`` or when ``mu=0``, the
                fairness penalty is disabled and the exact path is used.
                When provided and ``mu > 0``, automatically expanded to
                ``(n, 1)`` if 1-D and converted to float32.
            epochs (int): Number of Adam gradient-descent epochs.  Only
                used when ``mu > 0`` and ``q`` is provided.  Typical
                values: 100--500.  Default ``100``.
            lr (float): Adam learning rate.  Only used when ``mu > 0``
                and ``q`` is provided.  Typical values: ``1e-4`` to
                ``1e-2``.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility with
                scikit-learn-style interfaces.

        Returns:
            FairKernelRidge: The fitted model instance (``self``), for
            method chaining (e.g.
            ``model.fit(X, y).predict(X_test)``).

        Examples:
            Exact (unfair) solve with Cholesky:

            >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
            >>> model.fit(X_train, y_train)  # closed-form, ignores epochs

            Fair solve with Adam warm-started from exact:

            >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.5)
            >>> model.fit(X_train, y_train, q=q_train, epochs=200, lr=1e-3)

        Note:
            * The warm-start strategy (initializing from the unfair
              closed-form solution) substantially reduces the number of
              epochs needed for convergence compared to random
              initialization.  In practice, 50--200 epochs often suffice.
            * **Complexity (exact path):** :math:`O(n^2 d)` to build the
              kernel, :math:`O(n^3)` for Cholesky, or :math:`O(n^2 k)`
              for CG (``k`` = number of CG iterations).
            * **Complexity (fair path):** :math:`O(n^3)` for the
              warm-start Cholesky solve, then :math:`O(n^2)` per Adam
              epoch for the loss and gradient evaluation.
            * The kernel matrix is computed once and reused across all
              epochs.  For ``n > 10,000`` this dominates memory.

        References:
            * Scholkopf, B. and Smola, A. J. (2002). *Learning with
              Kernels*. MIT Press.  (Closed-form KRR solution.)
            * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for
              Stochastic Optimization." *ICLR*.  (Adam optimizer.)
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        y = ops.convert_to_tensor(y, dtype="float32")
        if len(ops.shape(y)) == 1:
            y = ops.expand_dims(y, axis=-1)

        self._X_train = X
        n = ops.shape(X)[0]
        K = self._compute_kernel(X)  # (n, n)

        has_fairness = q is not None and self.mu > 0
        if not has_fairness:
            # Exact closed-form KRR: (K + lambda I) alpha = y
            system = K + self.lam * ops.eye(n)
            if self.solver == "cholesky":
                from fairkl.ops.solvers import solve_cholesky

                self._alpha = solve_cholesky(system, y)
            else:
                from fairkl.ops.solvers import solve_cg

                self._alpha = solve_cg(system, y)
        else:
            # Gradient-based fair KRR with CKA penalty
            q = ops.convert_to_tensor(q, dtype="float32")
            if len(ops.shape(q)) == 1:
                q = ops.expand_dims(q, axis=-1)

            # Initialize alpha from the unfair closed-form solution
            system = K + self.lam * ops.eye(n)
            from fairkl.ops.solvers import solve_cholesky

            alpha_init = solve_cholesky(system, y)

            self._alpha = self.add_weight(
                name="alpha",
                shape=(n, 1),
                initializer=keras.initializers.Constant(alpha_init),
                trainable=True,
            )

            optimizer = keras.optimizers.Adam(learning_rate=lr)
            optimizer.build([self._alpha])

            lam = self.lam
            mu = self.mu
            sigma_q = self.sigma_q

            def loss_fn(alpha_val):
                pred = ops.matmul(K, alpha_val)  # (n, 1)
                mse = ops.mean((pred - y) ** 2)
                ridge = lam * ops.sum(alpha_val * ops.matmul(K, alpha_val))
                fairness = mu * cka_rbf(pred, q, sigma_f=1.0, sigma_q=sigma_q)
                return mse + ridge + fairness

            for _ in range(epochs):
                _loss, grads = jax.value_and_grad(loss_fn)(self._alpha.value)
                optimizer.apply([grads], [self._alpha])

        return self

    def predict(self, X, **kwargs):
        r"""Predict on new (out-of-sample) data.

        Computes predictions via the cross-kernel between test and
        training points:

        .. math::
            \hat{y} = K(X_{\text{test}},\, X_{\text{train}})\, \alpha

        where :math:`K(X_{\text{test}}, X_{\text{train}}) \in
        \mathbb{R}^{m \times n}` is the cross-kernel matrix evaluated
        between the ``m`` test points and ``n`` training points, and
        :math:`\alpha \in \mathbb{R}^{n \times 1}` is the dual
        coefficient vector learned during :meth:`fit`.

        This is the out-of-sample extension of the representer-theorem
        predictor :math:`f(x) = \sum_i \alpha_i k(x, x_i)`, evaluated
        simultaneously for all test points via the matrix product.

        Args:
            X (array-like of shape ``(m, d)``): Test inputs where ``m``
                is the number of test samples and ``d`` must match the
                training feature dimensionality.  Converted to a tensor
                (inherits the backend dtype).
            **kwargs: Unused; accepted for API compatibility with
                scikit-learn-style interfaces.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32 containing
            the predicted target values.  Each row corresponds to one
            test sample.

        Raises:
            AttributeError: If called before :meth:`fit` (i.e.
                ``_X_train`` or ``_alpha`` is ``None``).

        Examples:
            >>> model.fit(X_train, y_train)
            >>> y_pred = model.predict(X_test)  # shape (m, 1)
            >>> y_pred_train = model.predict(X_train)  # in-sample

        Note:
            * Computational complexity is :math:`O(m \cdot n \cdot d)` to
              build the cross-kernel (RBF) or :math:`O(m \cdot n \cdot d)`
              (linear), plus :math:`O(m \cdot n)` for the matrix-vector
              product.
            * The training data ``_X_train`` is stored in memory after
              :meth:`fit` and is required for cross-kernel computation.
        """
        X = ops.convert_to_tensor(X)
        K_cross = self._compute_kernel(X, self._X_train)  # (m, n)
        return ops.matmul(K_cross, self._alpha)

    def call(self, X):
        """Forward pass (delegates to :meth:`predict`).

        This method satisfies the ``keras.Model`` interface, allowing
        the model to be used in Keras pipelines and with ``model(X)``
        syntax.  It is functionally identical to :meth:`predict`.

        Args:
            X (array-like of shape ``(m, d)``): Input data.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32.
        """
        return self.predict(X)

    def get_alpha(self):
        r"""Return the learned dual coefficient vector :math:`\alpha`.

        The dual coefficients define the predictor in the kernel
        expansion :math:`f(x) = \sum_i \alpha_i\, k(x, x_i)`.
        Inspecting :math:`\alpha` is useful for:

        * **Sparsity analysis:** identifying which training points
          contribute most to predictions (large :math:`|\alpha_i|`).
        * **Convergence diagnostics:** comparing :math:`\alpha` across
          epochs or fairness penalty weights.
        * **Influence analysis:** the magnitude of :math:`\alpha_i`
          reflects the influence of training sample :math:`i` on
          predictions.

        Returns:
            Tensor or Variable of shape ``(n, 1)`` with dtype float32,
            where ``n`` is the number of training samples.

            * When ``mu=0`` (exact path): returns a plain JAX tensor.
            * When ``mu>0`` (fair path): returns a Keras ``Variable``
              that was optimized by Adam.

        Raises:
            AttributeError: If called before :meth:`fit` (i.e.
                ``_alpha`` is ``None``).

        Examples:
            >>> model.fit(X_train, y_train)
            >>> alpha = model.get_alpha()   # shape (n, 1)
            >>> alpha.shape
            (100, 1)
        """
        return self._alpha

    def get_kernel_matrix(self, X):
        r"""Return the cross-kernel matrix between ``X`` and training data.

        Computes :math:`K(X, X_{\text{train}}) \in \mathbb{R}^{m
        \times n}` using the kernel function specified at construction
        time.  This is the same cross-kernel used internally by
        :meth:`predict`.

        Useful for diagnostics such as:

        * **Kernel alignment:** checking how well the kernel captures
          the target structure.
        * **Effective rank:** computing the eigenvalue spectrum of the
          training Gram matrix (pass ``X = X_train``).
        * **Visualization:** plotting the kernel matrix as a heatmap.

        Args:
            X (array-like of shape ``(m, d)``): Input data.  When
                ``X`` is the training data, the returned matrix is the
                ``(n, n)`` training Gram matrix.

        Returns:
            Tensor of shape ``(m, n)`` with dtype float32, where ``n``
            is the number of training samples stored during :meth:`fit`.
            Entry ``(i, j)`` is :math:`k(X_i, X_{\text{train},j})`.

        Raises:
            AttributeError: If called before :meth:`fit`.

        Examples:
            >>> model.fit(X_train, y_train)
            >>> K_cross = model.get_kernel_matrix(X_test)  # (m, n)
            >>> K_train = model.get_kernel_matrix(X_train)  # (n, n)
        """
        X = ops.convert_to_tensor(X)
        return self._compute_kernel(X, self._X_train)

    def get_config(self):
        """Return the model configuration as a serializable dict.

        Extends the base ``keras.Model.get_config`` with all
        constructor arguments (``sigma``, ``lam``, ``mu``, ``sigma_q``,
        ``kernel``, ``solver``), enabling reconstruction via
        ``keras.Model.from_config``.

        Returns:
            dict: Configuration dictionary.  Keys include all base-class
            config entries plus the six constructor parameters.
        """
        config = super().get_config()
        config.update(
            sigma=self.sigma,
            lam=self.lam,
            mu=self.mu,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
            solver=self.solver,
        )
        return config
