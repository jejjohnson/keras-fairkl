"""Fair kernel ridge regression.

Solves the fairness-regularized kernel ridge regression problem::

    min_alpha  ||K alpha - y||^2 + lambda * alpha^T K alpha
               + mu * CKA(K alpha, q)

where ``K`` is the kernel matrix, ``alpha`` is the vector of dual
coefficients, ``lambda`` controls ridge regularization, ``mu`` controls
the CKA fairness penalty, and ``q`` encodes sensitive attributes.

When ``mu=0``, a closed-form Cholesky/CG solve is used (exact KRR).
When ``mu>0``, gradient descent with the CKA penalty is used, warm-started
from the exact (unfair) solution.
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
            \|K\alpha - y\|^2
          + \lambda\, \alpha^\top K \alpha
          + \mu\, \mathrm{CKA}(K\alpha,\; q)

    where :math:`K \in \mathbb{R}^{n \times n}` is the kernel (Gram)
    matrix, :math:`\lambda \ge 0` is the ridge regularization strength,
    :math:`\mu \ge 0` is the CKA fairness penalty weight, and
    :math:`q \in \mathbb{R}^{n \times d_q}` encodes the sensitive
    attributes.

    **Two training paths:**

    * **Exact (mu=0):** When no fairness penalty is active, the first-order
      optimality condition yields the closed-form system
      :math:`(K + \lambda I)\alpha = y`, solved via Cholesky factorization
      or conjugate-gradient iteration.
    * **Fair (mu>0):** The CKA term is non-linear in :math:`\alpha`, so
      gradient descent (Adam) is used instead.  To accelerate convergence
      the optimizer is *warm-started* from the exact (unfair) solution.

    Args:
        sigma (float): RBF bandwidth for the feature kernel
            :math:`k(x, x') = \exp\!\bigl(-\|x - x'\|^2 / (2\sigma^2)\bigr)`.
            Ignored when ``kernel="linear"``.  Default ``1.0``.
        lam (float): Ridge regularization strength :math:`\lambda`.
            Larger values shrink the dual coefficients toward zero,
            reducing overfitting at the cost of higher bias.  Default
            ``1e-3``.
        mu (float): CKA fairness penalty weight :math:`\mu`.  ``mu=0``
            recovers standard (unfair) kernel ridge regression.  Larger
            values enforce stronger statistical independence between
            predictions and sensitive attributes.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute kernel
            used inside CKA.  Default ``1.0``.
        kernel (str): Kernel type for the feature space.  One of
            ``"rbf"`` (Gaussian / squared-exponential) or ``"linear"``
            (:math:`k(x,x') = x^\top x'`).  Default ``"rbf"``.
        solver (str): Linear solver for the exact (``mu=0``) path.  One
            of ``"cholesky"`` (direct, :math:`O(n^3)`) or ``"cg"``
            (iterative, memory-friendly for large ``n``).  Ignored when
            ``mu > 0``.  Default ``"cholesky"``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__``.

    Examples:
        Standard (unfair) kernel ridge regression (``mu=0``):

        >>> import numpy as np
        >>> from fairkl.models import FairKernelRidge
        >>> X = np.random.randn(100, 5).astype("float32")
        >>> y = X @ np.ones(5) + 0.1 * np.random.randn(100)
        >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.0)
        >>> model.fit(X, y)
        >>> preds = model.predict(X[:10])  # shape (10, 1)

        Fair kernel ridge regression (``mu>0``):

        >>> q = np.random.randn(100, 1).astype("float32")  # sensitive attr
        >>> model = FairKernelRidge(sigma=1.0, lam=1e-2, mu=0.5)
        >>> model.fit(X, y, q=q, epochs=200, lr=1e-3)
        >>> preds = model.predict(X[:10])  # shape (10, 1)

    Note:
        * The CKA (Centered Kernel Alignment) penalty measures normalized
          statistical dependence between the model predictions and
          sensitive attributes via HSIC.  It is bounded in :math:`[0, 1]`,
          making :math:`\mu` interpretable regardless of data scale.
        * Computational complexity of the exact path is :math:`O(n^3)` for
          Cholesky and :math:`O(n^2 k)` for CG (``k`` iterations).  The
          fair path adds :math:`O(n^2)` per epoch for the CKA gradient.
        * The Adam warm-start from the exact solution typically converges
          in far fewer epochs than random initialization.

    References:
        * Scholkopf, B. and Smola, A. J. (2002). *Learning with Kernels*.
          MIT Press.  (Kernel ridge regression, representer theorem.)
        * Cortes, C. et al. (2012). "Algorithms for Learning Kernels Based
          on Centered Alignment." *JMLR*, 13, 795--828.  (CKA.)
        * Bakir, G. et al. (2004). "Learning to Find Pre-Images."
          *NeurIPS*.  (Kernel methods and pre-image problems.)

    See Also:
        :class:`FairLinear`: Primal-form (weight-space) fair linear model.
        :class:`FairKernelPCA`: Fair dimensionality reduction in the RKHS.
        :func:`~fairkl.metrics.cka.cka_rbf`: The CKA function used in the
            fairness penalty.
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

        Computes the kernel matrix :math:`K` and solves for the dual
        coefficients :math:`\alpha`.

        **Exact path (mu=0 or q is None):**

        Solves the linear system :math:`(K + \lambda I)\,\alpha = y`
        via Cholesky factorization (default) or conjugate gradient.
        The ``epochs`` and ``lr`` arguments are ignored.

        **Fair path (mu>0 and q is not None):**

        1. Warm-start: compute the exact solution
           :math:`\alpha_0 = (K + \lambda I)^{-1} y`.
        2. Run Adam gradient descent on the full objective
           :math:`\mathrm{MSE} + \lambda\,\mathrm{ridge} + \mu\,\mathrm{CKA}`
           starting from :math:`\alpha_0`.

        Args:
            X (array-like): Training inputs of shape ``(n, d)`` where
                ``n`` is the number of samples and ``d`` is the feature
                dimensionality.  Converted to float32.
            y (array-like): Target values of shape ``(n,)`` or ``(n, 1)``.
                Automatically expanded to ``(n, 1)`` if 1-D.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.  When ``None`` or when ``mu=0``, the
                fairness penalty is disabled and the exact path is used.
            epochs (int): Number of Adam gradient-descent epochs.  Only
                used when ``mu > 0`` and ``q`` is provided.  Default
                ``100``.
            lr (float): Adam learning rate.  Only used when ``mu > 0``
                and ``q`` is provided.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            self: The fitted model instance (for method chaining).

        Note:
            The warm-start strategy (initializing from the unfair
            closed-form solution) substantially reduces the number of
            epochs needed for convergence compared to random
            initialization.  For the exact path, the Cholesky solver
            has complexity :math:`O(n^3)` and the CG solver
            :math:`O(n^2 k)` where ``k`` is the number of CG iterations.

        References:
            * Bakir, G. et al. (2004). "Learning to Find Pre-Images."
              *NeurIPS*.  (Kernel methods framework.)
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

        where :math:`K(X_{\text{test}}, X_{\text{train}})` is the
        ``(m, n)`` cross-kernel matrix and :math:`\alpha` is the
        ``(n, 1)`` dual coefficient vector learned during ``fit``.

        Args:
            X (array-like): Test inputs of shape ``(m, d)`` where ``m``
                is the number of test samples and ``d`` must match the
                training feature dimensionality.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32 containing
            the predicted target values.

        Note:
            Computational complexity is :math:`O(m \cdot n \cdot d)` to
            build the cross-kernel plus :math:`O(m \cdot n)` for the
            matrix-vector product.
        """
        X = ops.convert_to_tensor(X)
        K_cross = self._compute_kernel(X, self._X_train)  # (m, n)
        return ops.matmul(K_cross, self._alpha)

    def call(self, X):
        return self.predict(X)

    def get_alpha(self):
        r"""Return the learned dual coefficient vector :math:`\alpha`.

        The dual coefficients define the predictor in the kernel
        expansion :math:`f(x) = \sum_i \alpha_i\, k(x, x_i)`.
        Inspecting :math:`\alpha` is useful for diagnosing sparsity,
        checking convergence, or computing influence of individual
        training points.

        Returns:
            Tensor of shape ``(n, 1)`` with dtype float32, where ``n``
            is the number of training samples.  When ``mu=0`` this is
            a plain tensor; when ``mu>0`` it is a Keras ``Variable``.
        """
        return self._alpha

    def get_kernel_matrix(self, X):
        r"""Return the cross-kernel matrix between ``X`` and training data.

        Computes :math:`K(X, X_{\text{train}})` using the kernel
        function specified at construction time.  Useful for
        diagnostics such as inspecting kernel alignment, checking
        effective rank, or visualizing the kernel structure.

        Args:
            X (array-like): Input data of shape ``(m, d)``.

        Returns:
            Tensor of shape ``(m, n)`` with dtype float32, where ``n``
            is the number of training samples stored during ``fit``.
        """
        X = ops.convert_to_tensor(X)
        return self._compute_kernel(X, self._X_train)

    def get_config(self):
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
