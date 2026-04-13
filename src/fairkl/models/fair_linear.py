"""Fair linear model.

Solves the fairness-regularized linear regression / classification problem::

    min_{w, b}  MSE(Xw + b, y) + lambda * ||w||^2 + mu * CKA(Xw + b, q)

where ``w`` is the weight vector, ``b`` is the bias, ``lambda`` controls L2
regularization, ``mu`` controls the CKA fairness penalty, and ``q`` encodes
sensitive attributes.  Optimization is performed via gradient descent with the
Adam optimizer.
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
            \frac{1}{n}\|Xw + b - y\|^2
          + \lambda\, \|w\|^2
          + \mu\, \mathrm{CKA}(Xw + b,\; q)

    where :math:`w \in \mathbb{R}^{d \times 1}` is the weight vector,
    :math:`b \in \mathbb{R}` is the bias, :math:`\lambda` controls L2
    regularization, :math:`\mu` controls the CKA fairness penalty, and
    :math:`q \in \mathbb{R}^{n \times d_q}` encodes sensitive
    attributes.

    Optimization uses the Adam optimizer with joint gradients over
    ``w`` and ``b`` via ``jax.value_and_grad``.

    Args:
        lam (float): L2 (ridge) regularization strength :math:`\lambda`.
            Penalizes :math:`\|w\|^2` to prevent overfitting.  Default
            ``1e-3``.
        mu (float): CKA fairness penalty weight :math:`\mu`.  ``mu=0``
            recovers standard ridge regression / classification.  Larger
            values enforce stronger statistical independence between
            predictions and sensitive attributes.  Default ``1.0``.
        sigma_f (float): RBF bandwidth for the prediction kernel inside
            CKA.  Default ``1.0``.
        sigma_q (float): RBF bandwidth for the sensitive-attribute kernel
            inside CKA.  Default ``1.0``.
        **kwargs: Additional keyword arguments passed to
            ``keras.Model.__init__``.

    Examples:
        >>> import numpy as np
        >>> from fairkl.models import FairLinear
        >>> X = np.random.randn(200, 5).astype("float32")
        >>> y = X @ np.ones(5) + 0.1 * np.random.randn(200)
        >>> q = np.random.randn(200, 1).astype("float32")
        >>> model = FairLinear(lam=1e-2, mu=0.5)
        >>> model.fit(X, y, q=q, epochs=100, lr=1e-3)
        >>> scores = model.decision_function(X[:10])  # shape (10, 1)

    Note:
        * The model is built lazily on the first call to ``fit`` (or
          ``call``).  The weight matrix ``w`` is initialized with Glorot
          uniform and the bias ``b`` is initialized to zero.
        * Unlike :class:`FairKernelRidge`, this model operates in the
          primal (weight space) and scales as :math:`O(n \cdot d)` per
          epoch rather than :math:`O(n^2)`.
        * When ``mu=0`` (or ``q=None``), the CKA term is skipped entirely
          and the model reduces to standard L2-regularized linear
          regression.

    References:
        * Cortes, C. et al. (2012). "Algorithms for Learning Kernels
          Based on Centered Alignment." *JMLR*, 13, 795--828.  (CKA.)
        * Kingma, D. P. and Ba, J. (2015). "Adam: A Method for
          Stochastic Optimization." *ICLR*.  (Adam optimizer.)

    See Also:
        :class:`FairKernelRidge`: Dual-form (kernel) fair regression.
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
        d = input_shape[-1]
        self._w = self.add_weight(
            name="w", shape=(d, 1), initializer="glorot_uniform", trainable=True
        )
        self._b = self.add_weight(
            name="b", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, X):
        X = ops.convert_to_tensor(X)
        return ops.matmul(X, self._w) + self._b

    def decision_function(self, X):
        r"""Compute raw decision scores (alias for ``call``).

        Returns the linear model output :math:`Xw + b` without any
        thresholding or activation.  For classification tasks, these
        scores can be passed to a sigmoid or softmax externally.

        Args:
            X (array-like): Input data of shape ``(m, d)``.

        Returns:
            Tensor of shape ``(m, 1)`` with dtype float32 containing
            the raw (un-thresholded) model output.
        """
        return self.call(X)

    def fit(self, X, y, q=None, epochs: int = 50, lr: float = 1e-2, **kwargs):
        r"""Fit the model using Adam gradient descent.

        On the first call the model is built (``w`` and ``b`` are
        allocated).  Then for ``epochs`` iterations the full-batch loss

        .. math::
            \mathcal{L} =
                \frac{1}{n}\|Xw + b - y\|^2
              + \lambda \|w\|^2
              + \mu\, \mathrm{CKA}(Xw + b,\; q)

        is evaluated and ``w``, ``b`` are updated jointly via Adam.

        Args:
            X (array-like): Training inputs of shape ``(n, d)`` where
                ``n`` is the number of samples and ``d`` is the number
                of features.  Converted to float32.
            y (array-like): Target values of shape ``(n,)`` or
                ``(n, 1)``.  Automatically expanded to ``(n, 1)`` if
                1-D.
            q (array-like or None): Sensitive attributes of shape
                ``(n, d_q)``.  When ``None`` or when ``mu=0``, the
                fairness term is omitted.
            epochs (int): Number of full-batch Adam epochs.  Default
                ``50``.
            lr (float): Adam learning rate.  Default ``1e-2``.
            **kwargs: Unused; accepted for API compatibility.

        Returns:
            self: The fitted model instance (for method chaining).

        Note:
            Gradients are computed via ``jax.value_and_grad`` with
            ``argnums=(0, 1)`` over the current values of ``w`` and
            ``b``.  This is a full-batch method -- for very large
            datasets consider mini-batch training through the standard
            Keras ``compile`` / ``fit`` API instead.
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
        config = super().get_config()
        config.update(
            lam=self.lam,
            mu=self.mu,
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
        )
        return config
