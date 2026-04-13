"""HSIC: Hilbert-Schmidt Independence Criterion.

Pure functions and Keras Loss/Metric classes for computing HSIC, a
kernel-based measure of statistical dependence between two random
variables.  HSIC is zero if and only if the variables are independent
(when using a characteristic kernel such as the RBF kernel).

The biased V-statistic estimator implemented here is non-negative,
unbounded above, and has :math:`O(n^2)` time complexity in the number
of samples.

References:
    Gretton, A., Bousquet, O., Smola, A., & Scholkopf, B. (2005).
    "Measuring Statistical Dependence with Hilbert-Schmidt Norms",
    ALT.

    Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B.,
    & Smola, A. (2012). "A Kernel Two-Sample Test", JMLR 13.
"""

from __future__ import annotations

import keras
import keras.ops as ops

from fairkl.kernels.exact import linear_kernel, rbf_kernel
from fairkl.ops.centering import center_kernel


# ---------------------------------------------------------------------------
# Pure functions (Layer 0)
# ---------------------------------------------------------------------------


def hsic_biased(K_f, K_q, H=None):
    """Biased V-statistic estimator of the Hilbert-Schmidt Independence Criterion.

    Computes the biased HSIC between two kernel matrices:

    .. math::

        \\widehat{\\text{HSIC}}_b = \\frac{1}{n^2} \\operatorname{tr}
        \\bigl(\\tilde{K}_f \\, \\tilde{K}_q\\bigr)

    where :math:`\\tilde{K} = H K H` is the centered kernel matrix and
    :math:`H = I - \\frac{1}{n} \\mathbf{1}\\mathbf{1}^\\top` is the
    centering matrix.  The trace-of-product is evaluated as the
    Frobenius inner product :math:`\\sum_{ij} (\\tilde{K}_f)_{ij}
    (\\tilde{K}_q)_{ij}`, avoiding an explicit matrix multiply and
    running in :math:`O(n^2)` time instead of :math:`O(n^3)`.

    This is a *biased* estimator (V-statistic).  Its value is always
    non-negative and unbounded above.

    Args:
        K_f (array-like): Kernel (Gram) matrix computed over
            predictions or features, shape ``(n, n)``.  Must be
            symmetric positive semi-definite.
        K_q (array-like): Kernel (Gram) matrix computed over sensitive
            attributes or a second set of variables, shape ``(n, n)``.
            Must be symmetric positive semi-definite.
        H (array-like or None): Centering matrix.  **Unused** -- kept
            only for API compatibility.  Centering is always performed
            efficiently via :func:`fairkl.ops.centering.center_kernel`.

    Returns:
        Scalar tensor (shape ``()``, same dtype as *K_f*) containing the
        biased HSIC estimate.  Non-negative, unbounded above.

    Examples:
        >>> import keras.ops as ops
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> from fairkl.metrics.hsic import hsic_biased
        >>> X = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> K = rbf_kernel(X, sigma=1.0)
        >>> hsic_biased(K, K)  # self-dependence, >= 0

    Note:
        * **Computational complexity**: :math:`O(n^2)` time where *n* is
          the number of samples, dominated by the element-wise product
          of the two centered matrices.
        * **Bias**: The V-statistic estimator is biased upward; for
          small *n* consider the debiased CKA variant
          (:func:`fairkl.metrics.cka.cka_debiased`).
        * **Numerical considerations**: For very large *n* the
          :math:`1/n^2` scaling can push the result toward zero; this
          is statistically correct and should not be compensated.

    References:
        Gretton, A., Bousquet, O., Smola, A., & Scholkopf, B. (2005).
        "Measuring Statistical Dependence with Hilbert-Schmidt Norms",
        ALT.

    See Also:
        :func:`hsic_rbf`: Convenience wrapper applying RBF kernels.
        :func:`hsic_linear`: Convenience wrapper applying linear kernels.
        :func:`fairkl.metrics.cka.cka_biased`: Normalized variant in [0, 1].
    """
    K_f_c = center_kernel(K_f)
    K_q_c = center_kernel(K_q)
    n = ops.cast(ops.shape(K_f)[0], dtype=K_f.dtype)
    # tr(A @ B) = sum(A * B)  — O(n^2) instead of O(n^3)
    return ops.sum(K_f_c * K_q_c) / (n * n)


def hsic_rbf(f, q, sigma_f: float = 1.0, sigma_q: float = 1.0):
    """HSIC with RBF (Gaussian) kernels applied to raw feature vectors.

    Convenience wrapper that constructs RBF Gram matrices from raw
    input tensors and delegates to :func:`hsic_biased`.  The RBF kernel
    is defined as:

    .. math::

        k(x, x') = \\exp\\!\\left(-\\frac{\\|x - x'\\|^2}{2\\sigma^2}\\right)

    Args:
        f (array-like): Predictions or features, shape ``(n, d_f)``.
            Each row is one sample.
        q (array-like): Sensitive attributes, shape ``(n, d_q)``.
            Must have the same number of samples *n* as *f*.
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *f*.  Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *q*.  Defaults to ``1.0``.

    Returns:
        Scalar tensor (shape ``()``) containing the biased HSIC
        estimate.

    Examples:
        >>> from fairkl.metrics.hsic import hsic_rbf
        >>> import keras.ops as ops
        >>> f = ops.convert_to_tensor([[0.0], [1.0], [2.0]])
        >>> q = ops.convert_to_tensor([[1.0], [0.0], [1.0]])
        >>> hsic_rbf(f, q, sigma_f=1.0, sigma_q=1.0)

    See Also:
        :func:`hsic_linear`: Linear-kernel variant.
        :func:`fairkl.metrics.cka.cka_rbf`: Normalized (CKA) RBF variant.
    """
    K_f = rbf_kernel(f, sigma=sigma_f)
    K_q = rbf_kernel(q, sigma=sigma_q)
    return hsic_biased(K_f, K_q)


def hsic_linear(f, q):
    """HSIC with linear kernels on raw feature vectors.

    Uses the linear kernel :math:`k(x, x') = x^\\top x'`.  With
    centered data this is equivalent to the squared Frobenius norm of
    the empirical cross-covariance matrix
    :math:`\\|\\hat{\\Sigma}_{fq}\\|_F^2` (up to a scaling factor).

    Args:
        f (array-like): Predictions or features, shape ``(n, d_f)``.
            Each row is one sample.
        q (array-like): Sensitive attributes, shape ``(n, d_q)``.
            Must have the same number of samples *n* as *f*.

    Returns:
        Scalar tensor (shape ``()``) containing the biased HSIC
        estimate under the linear kernel.

    Examples:
        >>> from fairkl.metrics.hsic import hsic_linear
        >>> import keras.ops as ops
        >>> f = ops.convert_to_tensor([[0.0], [1.0], [2.0]])
        >>> q = ops.convert_to_tensor([[1.0], [0.0], [1.0]])
        >>> hsic_linear(f, q)

    Note:
        The linear kernel is *not* characteristic, so
        ``hsic_linear(f, q) == 0`` does **not** guarantee independence
        -- only uncorrelatedness.  Use the RBF variant for a stricter
        independence test.

    See Also:
        :func:`hsic_rbf`: RBF-kernel variant (characteristic kernel).
        :func:`fairkl.metrics.cka.cka_linear`: Normalized (CKA) linear variant.
    """
    K_f = linear_kernel(f)
    K_q = linear_kernel(q)
    return hsic_biased(K_f, K_q)


# ---------------------------------------------------------------------------
# Keras Loss (Layer 1)
# ---------------------------------------------------------------------------


class HSICLoss(keras.losses.Loss):
    """HSIC dependence penalty as a Keras loss.

    Measures statistical dependence between model predictions and
    sensitive (protected) attributes using the biased HSIC estimator.
    Minimising this loss encourages the model to produce representations
    that are statistically independent of the sensitive attribute.

    In the Keras convention, ``y_true`` carries the sensitive attribute
    *q* and ``y_pred`` carries the model output *f*.  Both are
    automatically promoted to 2-D if passed as rank-1 tensors.

    The loss value is **non-negative and unbounded above** -- its
    magnitude depends on the kernel bandwidth and number of samples.
    If you need a bounded penalty whose weight ``mu`` is interpretable
    independently of data scale, prefer :class:`fairkl.metrics.cka.CKALoss`
    which is normalised to :math:`[0, 1]`.

    Args:
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_pred*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_true*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        kernel (str): Kernel type -- ``"rbf"`` (default) or
            ``"linear"``.
        name (str): Name of the loss instance.  Defaults to
            ``"hsic_loss"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.losses.Loss``.

    Examples:
        >>> from fairkl.metrics.hsic import HSICLoss
        >>> loss_fn = HSICLoss(sigma_f=1.0, sigma_q=1.0, kernel="rbf")
        >>> loss = loss_fn(y_true=sensitive_attrs, y_pred=predictions)

    See Also:
        :class:`fairkl.metrics.cka.CKALoss`: Normalised variant in [0, 1].
        :class:`HSICMetric`: Accumulating metric counterpart.
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        name: str = "hsic_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self.kernel = kernel

    def call(self, y_true, y_pred):
        # Ensure 2-D
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)

        if self.kernel == "linear":
            return hsic_linear(y_pred, y_true)
        return hsic_rbf(y_pred, y_true, self.sigma_f, self.sigma_q)

    def get_config(self):
        config = super().get_config()
        config.update(
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
        )
        return config


# ---------------------------------------------------------------------------
# Keras Metric (Layer 1)
# ---------------------------------------------------------------------------


class HSICMetric(keras.metrics.Metric):
    """Keras metric that tracks running-average HSIC over batches.

    Accumulates a running sum of per-batch HSIC values and divides by
    the number of batches seen at :meth:`result` time, yielding a
    simple **mean accumulation** estimate.

    This approach is exact when all batches have the same size.  For
    variable-size batches the mean is approximate because HSIC is
    computed per-batch and then averaged (rather than computed on the
    full dataset).

    Args:
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_pred*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_true*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        kernel (str): Kernel type -- ``"rbf"`` (default) or
            ``"linear"``.
        name (str): Name of the metric instance.  Defaults to
            ``"hsic"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.metrics.Metric``.

    Examples:
        >>> from fairkl.metrics.hsic import HSICMetric
        >>> metric = HSICMetric(kernel="rbf")
        >>> metric.update_state(y_true=sensitive_attrs, y_pred=preds)
        >>> metric.result()

    Note:
        The metric maintains two scalar variables: ``total`` (cumulative
        HSIC) and ``count`` (number of batches).  Call
        :meth:`reset_state` between epochs to clear these accumulators.

    See Also:
        :class:`HSICLoss`: Single-batch loss counterpart.
        :class:`fairkl.metrics.cka.CKAMetric`: Normalised CKA metric
            that accumulates HSIC components separately.
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        name: str = "hsic",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.total = self.add_variable(name="total", shape=(), initializer="zeros")
        self.count = self.add_variable(name="count", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)

        if self.kernel == "linear":
            val = hsic_linear(y_pred, y_true)
        else:
            val = hsic_rbf(y_pred, y_true, self.sigma_f, self.sigma_q)

        self.total.assign(self.total + val)
        self.count.assign(self.count + 1.0)

    def result(self):
        return self.total / ops.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
        )
        return config
