"""Scikit-learn compatible wrappers for fairkl models.

Requires: ``pip install scikit-learn`` (or ``pip install fairkl[sklearn]``).
"""

from __future__ import annotations

import numpy as np

from fairkl.models.fair_kernel_ridge import FairKernelRidge


try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for fairkl.sklearn_compat. "
        "Install it with: pip install scikit-learn"
    ) from exc


class FairKRREstimator(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for ``FairKernelRidge``.

    All constructor arguments are exposed as sklearn hyperparameters,
    so ``GridSearchCV`` and ``cross_val_score`` work out of the box.

    Args:
        sigma: RBF bandwidth for the feature kernel.
        lam: Ridge regularization strength.
        mu: CKA fairness penalty weight (0 = standard KRR).
        sigma_q: Bandwidth for the sensitive attribute kernel.
        kernel: ``"rbf"`` or ``"linear"``.
        epochs: Gradient-descent epochs (only when ``mu > 0``).
        lr: Learning rate (only when ``mu > 0``).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        lam: float = 1e-2,
        mu: float = 0.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        epochs: int = 200,
        lr: float = 0.005,
    ):
        self.sigma = sigma
        self.lam = lam
        self.mu = mu
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y, q=None):
        """Fit the model.

        Args:
            X: Training inputs of shape ``(n, d)``.
            y: Targets of shape ``(n,)`` or ``(n, 1)``.
            q: Sensitive attributes of shape ``(n, d_q)``.

        Returns:
            self
        """
        self.model_ = FairKernelRidge(
            sigma=self.sigma,
            lam=self.lam,
            mu=self.mu,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
        )
        self.model_.fit(X, y, q=q, epochs=self.epochs, lr=self.lr)
        return self

    def predict(self, X):
        """Predict on new data.

        Args:
            X: Test inputs of shape ``(m, d)``.

        Returns:
            Predictions as a 1-D numpy array of shape ``(m,)``.
        """
        return np.array(self.model_.predict(X)).ravel()
