"""Scikit-learn compatible wrappers for fairkl models.

This module provides thin adapter classes that make ``fairkl`` models
interoperable with the scikit-learn ecosystem (``GridSearchCV``,
``cross_val_score``, ``Pipeline``, etc.).

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

    Inherits from :class:`sklearn.base.BaseEstimator` and
    :class:`sklearn.base.RegressorMixin`, which together provide:

    * ``get_params()`` / ``set_params()`` -- automatic introspection of
      constructor arguments, enabling ``GridSearchCV`` and ``Pipeline``
      integration with no extra boilerplate.
    * ``score()`` -- defaults to R-squared (via ``RegressorMixin``).

    All constructor arguments are exposed as sklearn hyperparameters,
    so ``GridSearchCV``, ``cross_val_score``, and ``Pipeline`` work
    out of the box.

    Under the hood, :meth:`fit` instantiates a
    :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge` Keras model,
    trains it, and stores it as ``self.model_``.  :meth:`predict` delegates
    to the trained model and converts the output to a 1-D NumPy array.

    Args:
        sigma (float): RBF bandwidth for the feature kernel ``K_X``.
            Controls the length-scale of the Gaussian RBF: smaller values
            give a spikier kernel, larger values a smoother one.
            Defaults to ``1.0``.
        lam (float): Ridge regularization strength (``lambda``).  Larger
            values shrink the dual coefficients toward zero, reducing
            overfitting at the cost of bias.  Defaults to ``1e-2``.
        mu (float): CKA fairness penalty weight.  When ``mu = 0`` the
            model reduces to standard kernel ridge regression.  Increasing
            ``mu`` penalizes statistical dependence (measured by CKA)
            between predictions and the sensitive attributes *q*.
            Defaults to ``0.0``.
        sigma_q (float): RBF bandwidth for the sensitive attribute
            kernel ``K_Q``.  Only relevant when ``mu > 0``.
            Defaults to ``1.0``.
        kernel (str): Kernel type for the feature kernel.
            ``"rbf"`` (default) or ``"linear"``.
        epochs (int): Number of gradient-descent epochs.  Only used
            when ``mu > 0`` (the fairness-penalized objective is solved
            iteratively).  Defaults to ``200``.
        lr (float): Learning rate for gradient descent.  Only used when
            ``mu > 0``.  Defaults to ``0.005``.

    Examples:
        Basic usage with ``cross_val_score``:

        >>> import numpy as np
        >>> from sklearn.model_selection import cross_val_score
        >>> from fairkl.sklearn_compat.wrappers import FairKRREstimator
        >>> X = np.random.randn(100, 5).astype("float32")
        >>> y = np.random.randn(100).astype("float32")
        >>> est = FairKRREstimator(sigma=1.0, lam=0.01)
        >>> scores = cross_val_score(est, X, y, cv=3, scoring="r2")

        With fairness penalty and sensitive attributes:

        >>> q = np.random.randn(100, 1).astype("float32")
        >>> est_fair = FairKRREstimator(sigma=1.0, lam=0.01, mu=5.0)
        >>> est_fair.fit(X, y, q=q)
        FairKRREstimator(...)
        >>> preds = est_fair.predict(X)
        >>> preds.shape
        (100,)

    Note:
        When ``mu = 0`` the model is solved in closed form via a single
        Cholesky solve, so ``epochs`` and ``lr`` are unused.  When
        ``mu > 0`` the model uses gradient descent, and training time
        scales linearly with ``epochs``.

    See Also:
        :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge`: The
            underlying Keras model.
        :class:`~fairkl.tuning.keras_tuner.FairKernelRidgeHyperModel`:
            KerasTuner-based hyperparameter search.
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
        """Fit the fair kernel ridge regression model.

        Creates a fresh :class:`~fairkl.models.fair_kernel_ridge.FairKernelRidge`
        instance from the current hyperparameters and trains it on the
        supplied data.  The trained model is stored as ``self.model_``.

        When ``self.mu > 0``, the fairness-penalized objective is
        minimized via gradient descent for ``self.epochs`` steps at
        learning rate ``self.lr``.  The *q* argument must be provided
        in this case.

        Args:
            X (np.ndarray): Training input features of shape ``(n, d)``
                with dtype ``float32`` or ``float64``.
            y (np.ndarray): Target values of shape ``(n,)`` or
                ``(n, 1)``.  Internally reshaped if needed.
            q (np.ndarray | None): Sensitive / protected attributes of
                shape ``(n, d_q)``.  Required when ``mu > 0``.
                Ignored (may be ``None``) when ``mu == 0``.

        Returns:
            FairKRREstimator: ``self``, for method-chaining compatibility
            with the scikit-learn API.

        Note:
            Each call to :meth:`fit` creates a *new* underlying Keras
            model, so calling ``fit`` twice does not warm-start from the
            previous solution.
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
        """Generate predictions for new inputs.

        Delegates to the underlying ``FairKernelRidge.predict`` method
        and converts the result to a 1-D NumPy array, matching the
        scikit-learn convention that regressors return a flat array.

        Args:
            X (np.ndarray): Test input features of shape ``(m, d)``
                with the same dtype and feature dimensionality as the
                training data.

        Returns:
            np.ndarray: Predicted target values as a 1-D array of shape
            ``(m,)`` with dtype ``float32`` or ``float64`` (matching the
            model's internal precision).

        Raises:
            AttributeError: If called before :meth:`fit` (no
                ``self.model_`` attribute).
        """
        return np.array(self.model_.predict(X)).ravel()
