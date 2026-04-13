"""KerasTuner search space helpers for fair kernel models.

Provides ``FairKernelRidgeHyperModel``, a ``keras_tuner.HyperModel``
subclass that tunes all hyperparameters (sigma, lam, mu, sigma_q,
epochs, lr) of ``FairKernelRidge``.

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
    """KerasTuner HyperModel for ``FairKernelRidge``.

    Tunes kernel bandwidth (sigma), ridge regularization (lam),
    fairness weight (mu), sensitive kernel bandwidth (sigma_q),
    and training parameters (epochs, lr).

    Args:
        X_train: Training features of shape ``(n, d)``.
        y_train: Targets of shape ``(n,)``.
        q_train: Sensitive attributes of shape ``(n, d_q)`` or ``None``.
        X_val: Validation features.
        y_val: Validation targets.
        q_val: Validation sensitive attributes.
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
        """Build a ``FairKernelRidge`` model with tunable hyperparameters."""
        sigma = hp.Float("sigma", min_value=0.1, max_value=5.0, sampling="log")
        lam = hp.Float("lam", min_value=1e-4, max_value=1.0, sampling="log")
        mu = hp.Float("mu", min_value=0.0, max_value=20.0, step=1.0)
        sigma_q = hp.Float("sigma_q", min_value=0.1, max_value=5.0, sampling="log")
        return FairKernelRidge(sigma=sigma, lam=lam, mu=mu, sigma_q=sigma_q)

    def fit(self, hp, model, *args, **kwargs):
        """Train the model and return validation metrics.

        Returns:
            Dictionary with ``val_mse`` and ``val_cka`` keys.
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
            cka_val = float(cka_rbf(y_pred.reshape(-1, 1).astype("float32"), q_eval))

        return {"val_mse": mse, "val_cka": cka_val}
