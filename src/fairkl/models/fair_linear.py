"""Fair linear model.

Solves the fairness-regularized linear regression / classification problem
in primal (weight-space) form::

    min_{w, b}  (1/n)||Xw + b - y||^2 + lambda * ||w||^2
                + mu * CKA(Xw + b, q)

where ``w`` is the ``(d, 1)`` weight vector, ``b`` is the scalar bias,
``lambda >= 0`` controls L2 (ridge) regularization, ``mu >= 0`` controls
the CKA fairness penalty, and ``q`` encodes sensitive attributes.

Optimization is performed via full-batch gradient descent with the Adam
optimizer, computing gradients jointly over ``w`` and ``b`` using
``jax.value_and_grad``.  Unlike :class:`FairKernelRidge`, this model
operates entirely in weight space and scales as :math:`O(n d)` per epoch
rather than :math:`O(n^2)`, making it suitable for high-dimensional or
large-sample settings where kernel methods become prohibitive.

See :class:`FairKernelRidge` for the dual-form (kernel) analogue.
"""

from __future__ import annotations

import jax
import keras
import keras.ops as ops

from fairkl.metrics.cka import cka_rbf


class FairLinear(keras.Model):
    r"""Fair linear classifier / regressor with a CKA fairness penalty.

    Learns a linear model :math:`f(X) = X w + b` by jointly minimizing
    prediction error, L2 regularization, and a CKA-based fairness
    penalty:

    .. math::
        \min_{w,\, b}\;
            \underbrace{\frac{1}{n}\|Xw + b - y\|^2}_{\text{MSE}}
          + \underbrace{\lambda\, \|w\|^2}_{\text{ridge}}
          + \underbrace{\mu\, \mathrm{CKA}(Xw + b,\; q)}_{\text{fairness}}

    where :math:`w \in \mathbb{R}^{d \times 1}` is the weight vector,
    :math:`b \in \mathbb{R}` is the bias, :math:`\lambda \ge 0` controls
    L2 regularization, :math:`\mu \ge 0` controls the CKA fairness
    penalty, and :math:`q \in \mathbb{R}^{n \times d_q}` encodes
    sensitive attributes.

    Optimization uses the Adam optimizer with joint gradients over
    ``w`` and ``b`` via ``jax.value_and_grad``.  This is a full-batch
    method: the entire dataset is used for each gradient step.

    **Training algorithm:**

    1. **Build** (lazy, on first ``fit`` or ``call``): allocate ``w``
       with Glorot uniform initialization and ``b`` as zero.
    2. **Loop** for ``epochs`` iterations:

       a. Compute the forward pass :math:`\hat{y} = Xw + b`.
       b. Evaluate the loss :math:`\mathcal{L}(w, b)`.
       c. Compute gradients :math:`\nabla_w \mathcal{L},\;
          \nabla_b \mathcal{L}` via ``jax.value_and_grad``.
       d. Update ``w`` and ``b`` with Adam.

    Args:
        lam (float): L2 (ridge) regularization strength
            :math:`\lambda \ge 0`.  Penalizes :math:`\|w\|^2` to
            prevent overfitting.  Larger values produce smaller weight
            norms (more regularization).  Setting ``lam=0`` removes
            regularization entirely.  Default ``1e-3``.
        mu (float): CKA fairness penalty weight :math:`\mu \ge 0`.
            ``mu=0`` recovers standard ridge regression /
            classification.  Larger values enforce stronger statistical
            independence between predictions and sensitive attributes.
            Typical values range from ``0.01`` to ``10.0``.  Default
            ``1.0``.
        sigma_f (float): RBF bandwidth for the prediction kernel inside
            CKA.  Controls the length scale at which prediction
            similarity is measured.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute
            kernel inside CKA.  Controls the length scale at which
            sensitive-attribute similarity is measured.  Default ``1.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__`` (e.g. ``name``, ``dtype``).

    Attributes:
        lam (float): Stored ridge regularization strength.
        mu (float): Stored CKA penalty weight.
        sigma_f (float): Stored prediction-kernel RBF bandwidth.
        sigma_q (float): Stored sensitive-attribute RBF bandwidth.
        _w (Variable or None): Weight matrix of shape ``(d, 1)``,
            allocated during :meth:`build`.  Initialized with Glorot
            uniform.
        _b (Variable or None): Bias vector of shape ``(1,)``,
            allocated during :meth:`build`.  Initialized to zero.

    Examples:
        Fair linear regression with CKA penalty:

        >>> import numpy as np
        >>> from fairkl.models import FairLinear
        >>> X = np.random.randn(200, 5).astype("float32")
        >>> y = X @ np.ones(5) + 0.1 * np.random.randn(200)
        >>> q = np.random.randn(200, 1).astype("float32")
        >>> model = FairLinear(lam=1e-2, mu=0.5)
        >>> model.fit(X, y, q=q, epochs=100, lr=1e-3)
        >>> scores = model.decision_function(X[:10])  # shape (10, 1)

        Standard (unfair) ridge regression (``mu=0``):

        >>> model = FairLinear(lam=1e-2, mu=0.0)
        >>> model.fit(X, y, epochs=50)
        >>> preds = model.predict(X[:10])  # shape (10, 1)

    Note:
        * The model is built lazily on the first call to :meth:`fit`
          (or :meth:`call`).  The weight matrix ``w`` is initialized
          with Glorot uniform and the bias ``b`` is initialized to zero.
        * Unlike :class:`FairKernelRidge`, this model operates in the
          primal (weight space) and scales as :math:`O(n \cdot d)` per
          epoch rather than :math:`O(n^2)`, making it appropriate for
          large-scale or high-dimensional datasets.
        * When ``mu=0`` (or ``q=None``), the CKA term is skipped
          entirely and the model reduces to standard L2-regularized
          linear regression optimized with Adam.
        * This is a full-batch method.  For very large datasets,
          consider mini-batch training via the standard Keras
          ``compile`` / ``fit`` API.

    References:
        * Cortes, C., Mohri, M., and Rostamizadeh, A. (2012).
          "Algorithms for Learning Kernels Based on Centered Alignment."
          *JMLR*, 13, 795--828.  (CKA definition and properties.)
        * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for
          Stochastic Optimization." *ICLR*.  (Adam optimizer.)
        * Hoerl, A. E. and Kennard, R. W. (1970). "Ridge Regression:
          Biased Estimation for Nonorthogonal Problems."
          *Technometrics*, 12(1), 55--67.  (L2 / ridge regularization.)

    See Also:
        :class:`FairKernelRidge`: Dual-form (kernel) fair regression;
            handles non-linear relationships via kernel functions.
        :class:`FairPCA`: Fair dimensionality reduction (primal form).
        :func:`~fairkl.metrics.cka.cka_rbf`: The CKA function used in
            the fairness penalty.
    """

    def __init__(
        self,
        lam: float = 1e-3,
        mu: float = 1.0,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lam = lam
        self.mu = mu
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self._w = None
        self._b = None

    def build(self, input_shape):
        """Allocate the weight matrix and bias vector.

        Called automatically on the first :meth:`fit` or :meth:`call`.
        Creates two trainable Keras ``Variable`` objects:

        * ``_w``: shape ``(d, 1)``, Glorot uniform initialization.
        * ``_b``: shape ``(1,)``, zero initialization.

        Args:
            input_shape (tuple): Shape tuple ``(..., d)`` where ``d``
                is the input feature dimensionality.  Only the last
                dimension is used.
        """
        d = input_shape[-1]
        self._w = self.add_weight(
            name="w", shape=(d, 1), initializer="glorot_uniform", trainable=True
        )
        self._b = self.add_weight(
            name="b", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, X):
        r"""Compute the linear model output :math:`Xw + b`.

        This is the core forward pass.  Both :meth:`predict` (inherited
        from ``keras.Model``) and :meth:`decision_function` delegate to
        this method.

        Args:
            X (array-like of shape ``(m, d)``): Input data.  Converted
                to a tensor internally.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32 containing
            the raw (un-thresholded) model output.
        """
        X = ops.convert_to_tensor(X)
        return ops.matmul(X, self._w) + self._b

    def decision_function(self, X):
        r"""Compute raw decision scores (alias for :meth:`call`).

        Returns the linear model output :math:`f(X) = Xw + b` without
        any thresholding or activation function.  This follows the
        scikit-learn convention where ``decision_function`` returns raw
        scores and ``predict`` returns final predictions.

        For binary classification, pass the output through a sigmoid:
        ``probs = 1 / (1 + exp(-scores))``.  For multi-class, apply
        softmax externally.

        Args:
            X (array-like of shape ``(m, d)``): Input data where ``m``
                is the number of samples and ``d`` must match the
                training feature dimensionality.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32 containing
            the raw (un-thresholded) model output.

        Examples:
            >>> model.fit(X_train, y_train, q=q, epochs=100)
            >>> scores = model.decision_function(X_test)  # (m, 1)
            >>> # For classification: apply sigmoid externally
            >>> probs = 1.0 / (1.0 + np.exp(-scores))
        """
        return self.call(X)

    def fit(self, X, y, q=None, epochs: int = 50, lr: float = 1e-2, **kwargs):
        r"""Fit the model using Adam gradient descent.

        On the first call the model is built (``w`` and ``b`` are
        allocated via :meth:`build`).  Then for ``epochs`` iterations
        the full-batch loss

        .. math::
            \mathcal{L}(w, b) =
                \underbrace{\frac{1}{n}\|Xw + b - y\|^2}_{\text{MSE}}
              + \underbrace{\lambda \|w\|^2}_{\text{ridge}}
              + \underbrace{\mu\, \mathrm{CKA}(Xw + b,\; q)}_{\text{fairness}}

        is evaluated and ``w``, ``b`` are updated jointly via Adam.

        The optimization loop is:

        1. Evaluate :math:`\mathcal{L}` and compute
           :math:`(\nabla_w \mathcal{L},\, \nabla_b \mathcal{L})` via
           ``jax.value_and_grad(loss_fn, argnums=(0, 1))``.
        2. Apply Adam update to both ``w`` and ``b``.

        Args:
            X (array-like of shape ``(n, d)``): Training inputs where
                ``n`` is the number of samples and ``d`` is the number
                of features.  Converted to float32.
            y (array-like of shape ``(n,)`` or ``(n, 1)``): Target
                values.  Automatically expanded to ``(n, 1)`` if 1-D.
                Converted to float32.
            q (array-like of shape ``(n, d_q)`` or None): Sensitive
                attributes.  When ``None`` or when ``mu=0``, the
                fairness term is omitted and the model reduces to
                standard ridge regression.  Automatically expanded to
                ``(n, 1)`` if 1-D.
            epochs (int): Number of full-batch Adam epochs.  Typical
                values: 50--500.  Default ``50``.
            lr (float): Adam learning rate.  Typical values: ``1e-4``
                to ``1e-2``.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility with
                scikit-learn-style interfaces.

        Returns:
            FairLinear: The fitted model instance (``self``), for
            method chaining (e.g.
            ``preds = model.fit(X, y, q=q).predict(X_test)``).

        Examples:
            >>> model = FairLinear(lam=1e-2, mu=0.5)
            >>> model.fit(X_train, y_train, q=q_train, epochs=100)
            >>> preds = model.predict(X_test)  # shape (m, 1)

        Note:
            * Gradients are computed via ``jax.value_and_grad`` with
              ``argnums=(0, 1)`` over the current values of ``w`` and
              ``b``.
            * This is a full-batch method: the entire dataset is used
              for each gradient step.  For very large datasets, consider
              mini-batch training through the standard Keras ``compile``
              / ``fit`` API.
            * **Complexity per epoch:** :math:`O(n \cdot d)` for the
              forward pass and gradient computation (dominated by the
              matrix-vector product :math:`Xw`), plus :math:`O(n^2)` if
              the CKA fairness penalty is active (due to the pairwise
              RBF kernel on predictions).
        """
        X = ops.convert_to_tensor(X, dtype="float32")
        y = ops.convert_to_tensor(y, dtype="float32")
        if len(ops.shape(y)) == 1:
            y = ops.expand_dims(y, axis=-1)
        if q is not None:
            q = ops.convert_to_tensor(q, dtype="float32")
            if len(ops.shape(q)) == 1:
                q = ops.expand_dims(q, axis=-1)

        # Build if needed
        if not self.built:
            self(X[:1])

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        optimizer.build(self.trainable_variables)

        lam = self.lam
        mu = self.mu
        sigma_f = self.sigma_f
        sigma_q = self.sigma_q

        def loss_fn(w_val, b_val):
            pred = ops.matmul(X, w_val) + b_val
            task_loss = ops.mean((pred - y) ** 2)
            reg = lam * ops.sum(w_val * w_val)
            total = task_loss + reg
            if q is not None and mu > 0:
                total = total + mu * cka_rbf(pred, q, sigma_f=sigma_f, sigma_q=sigma_q)
            return total

        for _ in range(epochs):
            (_loss_val, (g_w, g_b)) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                self._w.value, self._b.value
            )
            optimizer.apply([g_w, g_b], self.trainable_variables)

        return self

    def get_config(self):
        """Return the model configuration as a serializable dict.

        Extends the base ``keras.Model.get_config`` with all
        constructor arguments (``lam``, ``mu``, ``sigma_f``,
        ``sigma_q``), enabling reconstruction via
        ``keras.Model.from_config``.

        Returns:
            dict: Configuration dictionary.  Keys include all base-class
            config entries plus the four constructor parameters.
        """
        config = super().get_config()
        config.update(
            lam=self.lam,
            mu=self.mu,
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
        )
        return config
