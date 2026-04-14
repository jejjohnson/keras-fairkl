"""Fair training wrapper for arbitrary Keras models.

Wraps any :class:`keras.Model` with a CKA-style fairness penalty so the
user can bring their own network architecture (MLP, CNN, transformer,
...) and train it with the standard Keras ``compile`` / ``fit`` API.

The fairness penalty is added via :meth:`keras.Model.add_loss` inside
:meth:`call`, which is the backend-agnostic extension point in
Keras 3 (works under JAX, TensorFlow, and PyTorch without modification).

See :class:`fairkl.models.fair_linear.FairLinear` and
:class:`fairkl.models.fair_kernel_ridge.FairKernelRidge` for the
built-in model classes that hard-wire a linear or kernel-ridge
predictor; :class:`FairModelWrapper` generalizes the fairness mechanism
to arbitrary predictors.
"""

from __future__ import annotations

import keras
import keras.ops as ops

from fairkl.metrics.cka import CKALoss


class FairModelWrapper(keras.Model):
    r"""Wrap any ``keras.Model`` with a CKA fairness penalty.

    Given a base predictor :math:`f_\theta : \mathcal{X} \to
    \mathbb{R}^{d_y}` and sensitive attributes :math:`q \in
    \mathbb{R}^{n \times d_q}`, this wrapper minimizes

    .. math::

        \mathcal{L}(\theta) =
            \underbrace{\mathcal{L}_{\text{task}}(f_\theta(X),\, y)}
                      _{\text{user's loss}}
          + \underbrace{\mu\, \mathrm{CKA}(f_\theta(X),\, q)}
                      _{\text{fairness}}

    where :math:`\mathcal{L}_{\text{task}}` is whichever loss the user
    passes to :meth:`compile` (MSE, binary cross-entropy, categorical
    cross-entropy, ...) and the fairness term defaults to
    :class:`fairkl.metrics.cka.CKALoss` (swappable via
    ``fairness_loss=``).

    The fairness term is added through :meth:`keras.Model.add_loss`
    inside :meth:`call`, so the wrapper works with whatever optimizer,
    metrics, callbacks, and dataset pipelines the user already uses.
    Because CKA is a batch-level statistic, the effective penalty is
    estimated on each minibatch -- use reasonably large batch sizes for
    stable fairness gradients.

    Args:
        base_model (keras.Model): The user's predictor.  Any
            ``keras.Model`` (Sequential, Functional, or subclassed) is
            accepted.  Its ``call`` must map an input tensor ``X`` of
            shape ``(batch, ...)`` to predictions of shape
            ``(batch, d_y)``.
        mu (float): Fairness penalty weight :math:`\mu \ge 0`.  Setting
            ``mu=0`` disables the fairness term entirely (the wrapper
            then behaves identically to ``base_model``).  Typical
            values: ``0.01`` to ``10.0``.  Default ``1.0``.
        fairness_loss (keras.losses.Loss or None): A dependence loss
            taking ``(y_true=q, y_pred=f)``.  Configure the kernel
            family and bandwidths by instantiating the loss yourself,
            e.g. ``CKALoss(sigma_f=0.5, sigma_q=2.0, kernel="rbf")`` or
            ``HSICLoss(sigma_f=1.0)`` or ``MMDLoss(...)``.  If
            ``None``, a default ``CKALoss()`` (RBF, unit bandwidths,
            biased) is used.  Default ``None``.
        **kwargs: Forwarded to ``keras.Model.__init__``.

    Examples:
        Wrap a small MLP and train with an MSE task loss and the
        default RBF CKA fairness penalty:

        >>> import keras
        >>> from fairkl.models import FairModelWrapper
        >>> mlp = keras.Sequential([
        ...     keras.layers.Dense(32, activation="relu"),
        ...     keras.layers.Dense(1),
        ... ])
        >>> model = FairModelWrapper(mlp, mu=0.5)
        >>> model.compile(optimizer="adam", loss="mse")
        >>> model.fit(X, y, q=q, epochs=20, batch_size=128)
        >>> preds = model.predict(X_test)

        Configure the fairness loss explicitly -- pick your own kernel
        family and bandwidths, or swap in HSIC / MMD:

        >>> from fairkl.metrics.cka import CKALoss
        >>> from fairkl.metrics.hsic import HSICLoss
        >>> from fairkl.metrics.mmd import MMDLoss
        >>> model = FairModelWrapper(
        ...     mlp, mu=0.5,
        ...     fairness_loss=CKALoss(sigma_f=0.5, sigma_q=2.0, debiased=True),
        ... )
        >>> model = FairModelWrapper(mlp, mu=0.1, fairness_loss=HSICLoss())
        >>> model = FairModelWrapper(mlp, mu=0.1, fairness_loss=MMDLoss())

    Note:
        * **Input protocol.** When the wrapper needs to see ``q`` (i.e.
          during training with a fairness penalty), the input is packed
          as a dict ``{"x": X, "q": q}``.  Anything else -- a plain
          tensor, a list/tuple, a nested dict -- is forwarded verbatim
          to ``base_model``, so multi-input predictors (``[x1, x2]``,
          ``{"a": a, "b": b}``) continue to work for inference.
          :meth:`fit` and :meth:`predict` handle the top-level packing
          / unpacking automatically so the user sees a flat
          ``fit(X, y, q=q)`` / ``predict(X)`` surface.  Validation
          inputs do not need ``q`` (the fairness penalty is only added
          when ``training=True``); pass ``validation_data=(X_val,
          y_val)`` as usual.
        * **Fairness is training-only.** The penalty is only added when
          ``call`` is invoked with ``training=True`` (inside
          :meth:`keras.Model.fit`).  It is *not* added during
          :meth:`predict` / :meth:`evaluate`, so the reported eval loss
          is pure task loss.
        * **Batch-size sensitivity.** CKA is a batch-level statistic;
          very small batches produce noisy fairness gradients.  A batch
          size of 128 or more is typical.
        * **Backend-agnostic.** Uses only ``keras.ops`` and
          :meth:`add_loss`; runs under JAX, TensorFlow, and PyTorch.

    References:
        * Cortes, C., Mohri, M., and Rostamizadeh, A. (2012).
          "Algorithms for Learning Kernels Based on Centered Alignment."
          *JMLR*, 13, 795--828.
        * Gretton, A., Bousquet, O., Smola, A., and Scholkopf, B.
          (2005).  "Measuring Statistical Dependence with
          Hilbert--Schmidt Norms."  *ALT*.

    See Also:
        :class:`fairkl.metrics.cka.CKALoss`: default fairness penalty.
        :class:`fairkl.models.fair_linear.FairLinear`: hard-wired
            linear fair predictor.
        :class:`fairkl.models.fair_kernel_ridge.FairKernelRidge`:
            hard-wired kernel-ridge fair predictor.
    """

    def __init__(
        self,
        base_model: keras.Model,
        mu: float = 1.0,
        fairness_loss: keras.losses.Loss | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mu < 0:
            raise ValueError(f"mu must be non-negative, got {mu}")
        self.base_model = base_model
        self.mu = float(mu)
        self.fairness_loss = fairness_loss if fairness_loss is not None else CKALoss()

    def call(self, inputs, training=False):
        """Forward pass; adds the fairness penalty only during training.

        The wrapper recognises a dict input ``{"x": X, "q": q}`` as its
        fairness protocol.  Any other input shape -- tensor, list,
        tuple, or a dict without both ``"x"`` and ``"q"`` keys -- is
        forwarded verbatim to ``base_model``, so multi-input predictors
        are unaffected.

        Args:
            inputs: Either ``{"x": X, "q": q}`` (packed by :meth:`fit`
                when sensitive attributes are supplied) or whatever
                ``base_model`` normally expects.
            training (bool): Forwarded to the base model.  The fairness
                penalty is only added when ``training`` is truthy.

        Returns:
            Predictions of shape ``(batch, d_y)``.
        """
        if isinstance(inputs, dict) and "x" in inputs and "q" in inputs:
            x, q = inputs["x"], inputs["q"]
            y_pred = self.base_model(x, training=training)
            if training and self.mu > 0:
                q = ops.cast(q, y_pred.dtype)
                self.add_loss(self.mu * self.fairness_loss(q, y_pred))
            return y_pred
        return self.base_model(inputs, training=training)

    def fit(self, X, y, q=None, **kwargs):
        """Fit the wrapped model with an optional fairness penalty.

        Args:
            X: Training inputs of shape ``(n, ...)``.
            y: Training targets compatible with the user's task loss.
            q: Sensitive attributes of shape ``(n,)`` or ``(n, d_q)``.
                When ``None``, the fairness term is disabled and fit
                degenerates to standard Keras training of
                ``base_model``.
            **kwargs: Forwarded to ``keras.Model.fit`` (``epochs``,
                ``batch_size``, ``validation_data``, ``callbacks``,
                ...).

        Returns:
            A ``keras.callbacks.History`` object.
        """
        if q is not None:
            q = ops.convert_to_tensor(q)
            if len(ops.shape(q)) == 1:
                q = ops.expand_dims(q, axis=-1)
            inputs = {"x": X, "q": q}
        else:
            inputs = X
        return super().fit(inputs, y, **kwargs)

    def predict(self, X, **kwargs):
        """Run the base model on ``X`` without adding any fairness term."""
        return super().predict(X, **kwargs)

    def get_config(self):
        """Return serializable config including the wrapped base model."""
        config = super().get_config()
        config.update(
            base_model=keras.saving.serialize_keras_object(self.base_model),
            mu=self.mu,
            fairness_loss=keras.saving.serialize_keras_object(self.fairness_loss),
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = dict(config)
        config["base_model"] = keras.saving.deserialize_keras_object(
            config["base_model"], custom_objects=custom_objects
        )
        config["fairness_loss"] = keras.saving.deserialize_keras_object(
            config["fairness_loss"], custom_objects=custom_objects
        )
        return cls(**config)
