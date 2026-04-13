"""KerasTuner search space helpers for fair kernel models.

Provides :class:`FairKernelRidgeHyperModel`, a
:class:`keras_tuner.HyperModel` subclass that defines the full search
space for :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge`.
Six hyperparameters are tuned: ``sigma``, ``lam``, ``mu``, ``sigma_q``,
``epochs``, and ``lr``.

Requires: ``pip install keras-tuner`` (or ``pip install fairkl[tuning]``).
"""

from __future__ import annotations

import numpy as np

from fairkl.metrics.cka import cka_rbf
from fairkl.models.fair_kernel_ridge import FairKernelRidge


try:
    import keras_tuner as kt
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "keras-tuner is required for fairkl.tuning. "
        "Install it with: pip install keras-tuner"
    ) from exc


class FairKernelRidgeHyperModel(kt.HyperModel):
    """KerasTuner HyperModel that defines the search space for ``FairKernelRidge``.

    Subclasses :class:`keras_tuner.HyperModel` and implements
    :meth:`build` and :meth:`fit` so that any KerasTuner search
    algorithm (``RandomSearch``, ``BayesianOptimization``,
    ``Hyperband``, etc.) can be used to tune the six
    hyperparameters of
    :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge`.

    The default search ranges are:

    ===============  ===============  ==========  ==========  ==========
    Hyperparameter   Type             Min         Max         Sampling
    ===============  ===============  ==========  ==========  ==========
    ``sigma``        ``Float``        0.1         5.0         log
    ``lam``          ``Float``        1e-4        1.0         log
    ``mu``           ``Float``        0.0         20.0        step=1.0
    ``sigma_q``      ``Float``        0.1         5.0         log
    ``epochs``       ``Int``          50          300         step=50
    ``lr``           ``Float``        1e-3        0.05        log
    ===============  ===============  ==========  ==========  ==========

    Training and validation data are passed at construction time (not
    through ``tuner.search()``) because ``FairKernelRidge`` is a
    non-standard Keras model whose ``fit`` signature accepts a *q*
    argument.

    Args:
        X_train (np.ndarray): Training features of shape ``(n, d)``,
            dtype ``float32`` or ``float64``.
        y_train (np.ndarray): Training targets of shape ``(n,)``.
        q_train (np.ndarray | None): Sensitive / protected attributes
            of shape ``(n, d_q)`` used for the CKA fairness penalty.
            Pass ``None`` for standard (unfair) kernel ridge regression.
        X_val (np.ndarray | None): Validation features of shape
            ``(n_val, d)``.  If ``None``, training data is used for
            evaluation (not recommended except for quick sanity checks).
        y_val (np.ndarray | None): Validation targets of shape
            ``(n_val,)``.
        q_val (np.ndarray | None): Validation sensitive attributes of
            shape ``(n_val, d_q)``.  Required for computing the
            ``val_cka`` metric; ignored if ``None``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras_tuner.HyperModel.__init__``.

    Examples:
        >>> import numpy as np
        >>> import keras_tuner as kt
        >>> from fairkl.tuning.keras_tuner import FairKernelRidgeHyperModel
        >>> X_tr = np.random.randn(200, 5).astype("float32")
        >>> y_tr = np.random.randn(200).astype("float32")
        >>> q_tr = np.random.randn(200, 1).astype("float32")
        >>> hm = FairKernelRidgeHyperModel(
        ...     X_train=X_tr, y_train=y_tr, q_train=q_tr,
        ... )
        >>> tuner = kt.RandomSearch(
        ...     hm,
        ...     objective=kt.Objective("val_mse", direction="min"),
        ...     max_trials=20,
        ...     directory="tuner_results",
        ...     project_name="fair_krr",
        ... )
        >>> tuner.search()                           # doctest: +SKIP
        >>> best_hp = tuner.get_best_hyperparameters()[0]

    Note:
        The :meth:`fit` method returns a plain ``dict`` (not a Keras
        ``History`` object).  KerasTuner interprets dictionary return
        values as single-epoch metric snapshots, which is appropriate
        here because ``FairKernelRidge.fit`` runs its own internal
        training loop.

    See Also:
        :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge`: The
            model being tuned.
        :class:`~fairkl.sklearn_compat.wrappers.FairKRREstimator`:
            scikit-learn wrapper (use with ``GridSearchCV`` instead of
            KerasTuner).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        q_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        q_val: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.q_train = q_train
        self.X_val = X_val
        self.y_val = y_val
        self.q_val = q_val

    def build(self, hp):
        """Build a ``FairKernelRidge`` model with tunable hyperparameters.

        Samples four continuous hyperparameters from the search space
        defined by *hp*:

        * ``sigma`` -- RBF bandwidth, log-uniform in ``[0.1, 5.0]``
        * ``lam`` -- ridge penalty, log-uniform in ``[1e-4, 1.0]``
        * ``mu`` -- fairness weight, uniform grid ``{0, 1, ..., 20}``
        * ``sigma_q`` -- sensitive-kernel bandwidth, log-uniform in
          ``[0.1, 5.0]``

        Args:
            hp (keras_tuner.HyperParameters): KerasTuner hyperparameter
                container used to register and sample values.

        Returns:
            FairKernelRidge: An untrained model instance configured with
            the sampled hyperparameters.
        """
        sigma = hp.Float("sigma", min_value=0.1, max_value=5.0, sampling="log")
        lam = hp.Float("lam", min_value=1e-4, max_value=1.0, sampling="log")
        mu = hp.Float("mu", min_value=0.0, max_value=20.0, step=1.0)
        sigma_q = hp.Float("sigma_q", min_value=0.1, max_value=5.0, sampling="log")
        return FairKernelRidge(sigma=sigma, lam=lam, mu=mu, sigma_q=sigma_q)

    def fit(self, hp, model, *args, **kwargs):
        """Train the model and return validation metrics.

        Samples two additional training hyperparameters from *hp*:

        * ``epochs`` -- integer in ``{50, 100, 150, ..., 300}``
        * ``lr`` -- learning rate, log-uniform in ``[1e-3, 0.05]``

        The model is trained on ``(X_train, y_train, q_train)`` using
        ``FairKernelRidge.fit``.  Evaluation is performed on the
        validation split if provided, otherwise on the training data.

        Two metrics are computed:

        * ``val_mse`` -- mean squared error between predictions and
          targets.
        * ``val_cka`` -- CKA dependence between predictions and
          sensitive attributes (via :func:`~fairkl.metrics.cka.cka_rbf`).
          Set to ``0.0`` when no sensitive attributes are available.

        Args:
            hp (keras_tuner.HyperParameters): KerasTuner hyperparameter
                container (shared with :meth:`build`).
            model (FairKernelRidge): The model instance returned by
                :meth:`build`.
            *args: Unused positional arguments (KerasTuner compatibility).
            **kwargs: Unused keyword arguments (KerasTuner compatibility).

        Returns:
            dict[str, float]: A dictionary with keys ``"val_mse"`` and
            ``"val_cka"``.  KerasTuner reads these to rank trials.
        """
        epochs = hp.Int("epochs", min_value=50, max_value=300, step=50)
        lr = hp.Float("lr", min_value=1e-3, max_value=0.05, sampling="log")

        model.fit(
            self.X_train,
            self.y_train,
            q=self.q_train,
            epochs=epochs,
            lr=lr,
        )

        # Evaluate on validation data (or train if no val provided)
        X_eval = self.X_val if self.X_val is not None else self.X_train
        y_eval = self.y_val if self.y_val is not None else self.y_train
        q_eval = self.q_val if self.q_val is not None else self.q_train

        y_pred = np.array(model.predict(X_eval)).ravel()
        mse = float(np.mean((y_pred - y_eval) ** 2))

        cka_val = 0.0
        if q_eval is not None:
            cka_val = float(
                cka_rbf(
                    y_pred.reshape(-1, 1).astype("float32"),
                    q_eval.astype("float32"),
                    sigma_q=model.sigma_q,
                )
            )

        return {"val_mse": mse, "val_cka": cka_val}
