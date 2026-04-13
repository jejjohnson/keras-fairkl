"""CKA: Centered Kernel Alignment.

Normalized HSIC bounded in :math:`[0, 1]`.  CKA is invariant to
**orthogonal transformations** and to **isotropic scaling** of the
features; however, exact scale invariance holds only when linear kernels
are used.  With non-linear kernels (e.g., RBF) the value is
approximately but not exactly invariant to scaling.

This module provides biased and debiased estimators, convenience
wrappers for RBF and linear kernels, and Keras Loss/Metric classes for
use as fairness regularizers.

References:
    Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
    "Similarity of Neural Network Representations Revisited", ICML.

    Cortes, C., Mohri, M., & Rostamizadeh, A. (2012). "Algorithms for
    Learning Kernels Based on Centered Alignment", JMLR 13.

    Murphy, E., Hienert, D., Ruff, L., & Biecek, P. (2024).
    "Correcting Biased Centered Kernel Alignment Measures in Biological
    and Artificial Neural Networks" (Re-Align), ICLR.

    Szekely, G. J. & Rizzo, M. L. (2014). "Partial Distance Correlation
    with Methods for Dissimilarities", The Annals of Statistics 42(6).
"""

from __future__ import annotations

import keras
import keras.ops as ops

from fairkl.kernels.exact import linear_kernel, rbf_kernel
from fairkl.ops.centering import center_kernel


# ---------------------------------------------------------------------------
# Debiased centering (U-statistic)
# ---------------------------------------------------------------------------


def center_gram_unbiased(K):
    """U-statistic centering of a Gram matrix (debiased).

    Applies the unbiased centering operation that removes the
    finite-sample bias present in the standard (V-statistic) centering.
    The procedure:

    1. **Zeros the diagonal** of *K* (removes self-similarity terms).
    2. Uses an :math:`(n-2)` normalization for the column/row means
       instead of the standard :math:`n`, following the doubly-centered
       distance covariance formulation.

    This centering removes the upward bias that inflates CKA in
    high-dimensional, low-sample settings.

    .. math::

        \\tilde{K}_{ij} = K_{ij} - \\frac{1}{n-2}\\sum_k K_{ik}
        - \\frac{1}{n-2}\\sum_k K_{kj}
        + \\frac{1}{(n-1)(n-2)}\\sum_{k,l} K_{kl}

    with :math:`\\tilde{K}_{ii} = 0`.

    Args:
        K (array-like): Symmetric kernel (Gram) matrix of shape
            ``(n, n)``.  Must be square and symmetric.

    Returns:
        Centered kernel matrix of shape ``(n, n)`` with the same dtype
        as *K*.  The diagonal entries are exactly zero.

    Examples:
        >>> import keras.ops as ops
        >>> from fairkl.metrics.cka import center_gram_unbiased
        >>> K = ops.convert_to_tensor([[1.0, 0.5], [0.5, 1.0]])
        >>> K_c = center_gram_unbiased(K)

    Note:
        * Requires :math:`n \\geq 3` for the :math:`(n-2)` denominator
          to be positive.  Behaviour is undefined for :math:`n < 3`.
        * After centering the resulting matrix is **not** guaranteed to
          be positive semi-definite; the debiased HSIC may therefore be
          slightly negative.

    References:
        Szekely, G. J. & Rizzo, M. L. (2014). "Partial Distance
        Correlation with Methods for Dissimilarities", The Annals of
        Statistics 42(6).

    See Also:
        :func:`fairkl.ops.centering.center_kernel`: Standard
            (biased, V-statistic) centering.
    """
    n = ops.shape(K)[0]
    eye_n = ops.eye(n, dtype=K.dtype)
    # Zero diagonal
    K = K * (1.0 - eye_n)
    n_float = ops.cast(n, dtype=K.dtype)
    # Column means excluding diagonal
    means = ops.sum(K, axis=0) / (n_float - 2.0)
    means = means - ops.sum(means) / (2.0 * (n_float - 1.0))
    K = K - ops.expand_dims(means, 0) - ops.expand_dims(means, 1)
    # Zero diagonal again
    K = K * (1.0 - eye_n)
    return K


# ---------------------------------------------------------------------------
# Pure functions (Layer 0)
# ---------------------------------------------------------------------------


def cka_biased(K_f, K_q):
    """Biased CKA (Centered Kernel Alignment) on pre-computed kernel matrices.

    Computes the biased CKA as the cosine similarity of centered kernel
    matrices in the Hilbert-Schmidt inner-product space:

    .. math::

        \\text{CKA}(K, L) = \\frac{\\text{HSIC}(K, L)}
        {\\sqrt{\\text{HSIC}(K, K) \\cdot \\text{HSIC}(L, L)}}

    The :math:`1/n^2` factor in each HSIC term cancels in the ratio, so
    the implementation directly computes:

    .. math::

        \\frac{\\operatorname{tr}(\\tilde{K}_f \\, \\tilde{K}_q)}
        {\\sqrt{\\operatorname{tr}(\\tilde{K}_f^2) \\cdot
        \\operatorname{tr}(\\tilde{K}_q^2)}}

    The result is bounded in :math:`[0, 1]` and equals 1 when the two
    centred kernel matrices are identical up to a positive scalar.

    Args:
        K_f (array-like): Kernel (Gram) matrix over predictions or
            features, shape ``(n, n)``.  Must be symmetric PSD.
        K_q (array-like): Kernel (Gram) matrix over sensitive
            attributes, shape ``(n, n)``.  Must be symmetric PSD.

    Returns:
        Scalar tensor (shape ``()``, same dtype as *K_f*) containing
        the biased CKA value in :math:`[0, 1]`.

    Examples:
        >>> import keras.ops as ops
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> from fairkl.metrics.cka import cka_biased
        >>> X = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> K = rbf_kernel(X, sigma=1.0)
        >>> cka_biased(K, K)  # self-alignment => ~1.0

    Note:
        * **Epsilon floor**: A floor of ``1e-6`` is applied to the
          denominator to avoid division by zero when one kernel matrix
          is constant (zero after centering).
        * **Bias**: Uses V-statistic (biased) centering.  In
          high-dimensional, low-sample regimes the estimate is
          inflated; prefer :func:`cka_debiased` in such settings.

    References:
        Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
        "Similarity of Neural Network Representations Revisited", ICML.

    See Also:
        :func:`cka_debiased`: Debiased variant using U-statistic centering.
        :func:`fairkl.metrics.hsic.hsic_biased`: Unnormalized HSIC.
    """
    K_f_c = center_kernel(K_f)
    K_q_c = center_kernel(K_q)
    hsic_cross = ops.sum(K_f_c * K_q_c)
    hsic_ff = ops.sum(K_f_c * K_f_c)
    hsic_qq = ops.sum(K_q_c * K_q_c)
    denom = ops.sqrt(hsic_ff * hsic_qq)
    return hsic_cross / ops.maximum(denom, 1e-6)


def cka_debiased(K_f, K_q):
    """Debiased CKA with U-statistic centering.

    Uses :func:`center_gram_unbiased` to remove the finite-sample bias
    that inflates the biased CKA estimator when the number of samples
    *n* is small relative to the feature dimensionality.  The formula
    is otherwise identical to :func:`cka_biased` but operates on the
    U-statistic-centred Gram matrices.

    .. math::

        \\widehat{\\text{CKA}}_{\\text{debiased}} =
        \\frac{\\sum_{ij} \\tilde{K}^U_{f,ij} \\, \\tilde{K}^U_{q,ij}}
        {\\sqrt{\\sum_{ij} (\\tilde{K}^U_{f,ij})^2 \\cdot
        \\sum_{ij} (\\tilde{K}^U_{q,ij})^2}}

    where :math:`\\tilde{K}^U` denotes the U-statistic-centred matrix
    (see :func:`center_gram_unbiased`).

    Args:
        K_f (array-like): Kernel (Gram) matrix over predictions or
            features, shape ``(n, n)``.  Must be symmetric PSD.
        K_q (array-like): Kernel (Gram) matrix over sensitive
            attributes, shape ``(n, n)``.  Must be symmetric PSD.

    Returns:
        Scalar tensor (shape ``()``, same dtype as *K_f*).  The value
        is typically in :math:`[0, 1]`, but **may be slightly negative**
        because the debiased centering removes the non-negativity
        guarantee of the V-statistic estimator.

    Examples:
        >>> import keras.ops as ops
        >>> from fairkl.kernels.exact import rbf_kernel
        >>> from fairkl.metrics.cka import cka_debiased
        >>> X = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        >>> K = rbf_kernel(X, sigma=1.0)
        >>> cka_debiased(K, K)

    Note:
        * **Negative values**: Because the U-statistic centering does
          not guarantee positive semi-definiteness of the centred
          matrix, the HSIC cross-term and consequently the CKA value
          can be slightly negative.  This is statistically valid.
        * **Epsilon floor**: A ``1e-6`` floor on the denominator
          prevents division by zero; the self-HSIC terms are also
          clamped to non-negative before the square root.

    References:
        Murphy, E., Hienert, D., Ruff, L., & Biecek, P. (2024).
        "Correcting Biased Centered Kernel Alignment Measures in
        Biological and Artificial Neural Networks" (Re-Align), ICLR.

    See Also:
        :func:`cka_biased`: Standard (biased) CKA estimator.
        :func:`center_gram_unbiased`: The U-statistic centering
            operation used internally.
    """
    K_f_c = center_gram_unbiased(K_f)
    K_q_c = center_gram_unbiased(K_q)
    hsic_cross = ops.sum(K_f_c * K_q_c)
    hsic_ff = ops.sum(K_f_c * K_f_c)
    hsic_qq = ops.sum(K_q_c * K_q_c)
    denom = ops.sqrt(ops.maximum(hsic_ff, 0.0) * ops.maximum(hsic_qq, 0.0))
    return hsic_cross / ops.maximum(denom, 1e-6)


def cka_rbf(f, q, sigma_f: float = 1.0, sigma_q: float = 1.0, debiased: bool = False):
    """CKA with RBF (Gaussian) kernels applied to raw feature vectors.

    Convenience wrapper that constructs RBF Gram matrices from raw
    input tensors and delegates to :func:`cka_biased` or
    :func:`cka_debiased`.

    Args:
        f (array-like): Predictions or features, shape ``(n, d_f)``.
            Each row is one sample.
        q (array-like): Sensitive attributes, shape ``(n, d_q)``.
            Must have the same number of samples *n* as *f*.
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *f*.  Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *q*.  Defaults to ``1.0``.
        debiased (bool): If ``True``, use U-statistic (debiased)
            centering via :func:`cka_debiased`.  Defaults to ``False``.

    Returns:
        Scalar tensor (shape ``()``) containing the CKA value.

    Examples:
        >>> from fairkl.metrics.cka import cka_rbf
        >>> import keras.ops as ops
        >>> f = ops.convert_to_tensor([[0.0], [1.0], [2.0]])
        >>> q = ops.convert_to_tensor([[1.0], [0.0], [1.0]])
        >>> cka_rbf(f, q, sigma_f=1.0, sigma_q=1.0)

    Note:
        With RBF kernels, CKA is **approximately** but not exactly
        invariant to isotropic scaling of the inputs.  Exact scale
        invariance holds only for linear kernels (see
        :func:`cka_linear`).

    See Also:
        :func:`cka_linear`: Linear-kernel variant (exactly scale-invariant).
        :func:`fairkl.metrics.hsic.hsic_rbf`: Unnormalized HSIC with
            RBF kernels.
    """
    K_f = rbf_kernel(f, sigma=sigma_f)
    K_q = rbf_kernel(q, sigma=sigma_q)
    if debiased:
        return cka_debiased(K_f, K_q)
    return cka_biased(K_f, K_q)


def cka_linear(f, q, debiased: bool = False):
    """CKA with linear kernels on raw feature vectors.

    Uses the linear kernel :math:`k(x, x') = x^\\top x'`.  Linear CKA
    is **exactly invariant to isotropic scaling and orthogonal
    transformations** of the feature vectors, making it particularly
    suited for comparing representations.

    Args:
        f (array-like): Predictions or features, shape ``(n, d_f)``.
            Each row is one sample.
        q (array-like): Sensitive attributes, shape ``(n, d_q)``.
            Must have the same number of samples *n* as *f*.
        debiased (bool): If ``True``, use U-statistic (debiased)
            centering via :func:`cka_debiased`.  Defaults to ``False``.

    Returns:
        Scalar tensor (shape ``()``) containing the CKA value.

    Examples:
        >>> from fairkl.metrics.cka import cka_linear
        >>> import keras.ops as ops
        >>> f = ops.convert_to_tensor([[0.0], [1.0], [2.0]])
        >>> q = ops.convert_to_tensor([[1.0], [0.0], [1.0]])
        >>> cka_linear(f, q)

    Note:
        Scale invariance is **exact** for linear kernels (unlike RBF)
        because the linear Gram matrix :math:`K = X X^\\top` scales
        quadratically, and the normalization in CKA cancels this factor
        exactly.

    See Also:
        :func:`cka_rbf`: RBF-kernel variant (approximately scale-invariant).
        :func:`fairkl.metrics.hsic.hsic_linear`: Unnormalized HSIC with
            linear kernels.
    """
    K_f = linear_kernel(f)
    K_q = linear_kernel(q)
    if debiased:
        return cka_debiased(K_f, K_q)
    return cka_biased(K_f, K_q)


# ---------------------------------------------------------------------------
# Keras Loss (Layer 1)
# ---------------------------------------------------------------------------


class CKALoss(keras.losses.Loss):
    """CKA dependence penalty as a Keras loss.

    CKA is bounded in :math:`[0, 1]`, making the penalty weight
    ``mu`` interpretable regardless of data scale or batch size.  This
    is the key advantage over :class:`fairkl.metrics.hsic.HSICLoss`,
    whose magnitude varies with kernel bandwidth and sample count.

    In the Keras convention, ``y_true`` carries the sensitive attribute
    *q* and ``y_pred`` carries the model output *f*.  Both are
    automatically promoted to 2-D if passed as rank-1 tensors.

    .. math::

        \\mathcal{L}_{\\text{CKA}} = \\text{CKA}(K_f, K_q)
        \\in [0, 1]

    Args:
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_pred*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_true*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        kernel (str): Kernel type -- ``"rbf"`` (default) or
            ``"linear"``.
        debiased (bool): If ``True``, use U-statistic (debiased)
            centering.  Defaults to ``False``.
        name (str): Name of the loss instance.  Defaults to
            ``"cka_loss"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.losses.Loss``.

    Examples:
        >>> from fairkl.metrics.cka import CKALoss
        >>> loss_fn = CKALoss(kernel="linear", debiased=True)
        >>> loss = loss_fn(y_true=sensitive_attrs, y_pred=predictions)

    Note:
        Because the loss is bounded, setting ``mu = 0.5`` always
        allocates roughly half the gradient budget to fairness
        regardless of absolute data scale -- a property that HSIC-based
        losses do not share.

    See Also:
        :class:`fairkl.metrics.hsic.HSICLoss`: Unnormalized HSIC loss
            (unbounded, scale-dependent).
        :class:`CKAMetric`: Accumulating metric counterpart.
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        debiased: bool = False,
        name: str = "cka_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.debiased = debiased

    def call(self, y_true, y_pred):
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)

        if self.kernel == "linear":
            return cka_linear(y_pred, y_true, debiased=self.debiased)
        return cka_rbf(
            y_pred, y_true, self.sigma_f, self.sigma_q, debiased=self.debiased
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
            debiased=self.debiased,
        )
        return config


# ---------------------------------------------------------------------------
# Keras Metric (Layer 1)
# ---------------------------------------------------------------------------


class CKAMetric(keras.metrics.Metric):
    """Keras metric that tracks CKA over batches via component accumulation.

    Unlike a naive running average of per-batch CKA values, this metric
    accumulates the three HSIC components -- ``hsic_cross``,
    ``hsic_ff``, and ``hsic_qq`` -- separately across batches and
    computes the CKA ratio only at :meth:`result` time:

    .. math::

        \\text{CKA} = \\frac{\\sum_b \\text{HSIC}_{\\text{cross}}^{(b)}}
        {\\sqrt{\\sum_b \\text{HSIC}_{ff}^{(b)} \\cdot
        \\sum_b \\text{HSIC}_{qq}^{(b)}}}

    This approach is correct for **fixed-size batches** because the
    :math:`1/n^2` normalisation factor is constant and cancels.  For
    variable-size batches the estimate is approximate.

    Args:
        sigma_f (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_pred*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        sigma_q (float): Bandwidth (length-scale) for the RBF kernel
            applied to *y_true*.  Ignored when ``kernel="linear"``.
            Defaults to ``1.0``.
        kernel (str): Kernel type -- ``"rbf"`` (default) or
            ``"linear"``.
        debiased (bool): If ``True``, use U-statistic (debiased)
            centering.  Defaults to ``False``.
        name (str): Name of the metric instance.  Defaults to
            ``"cka"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.metrics.Metric``.

    Examples:
        >>> from fairkl.metrics.cka import CKAMetric
        >>> metric = CKAMetric(kernel="rbf", debiased=False)
        >>> metric.update_state(y_true=sensitive_attrs, y_pred=preds)
        >>> metric.result()  # CKA computed from accumulated components

    Note:
        * The metric maintains three scalar variables (``hsic_cross``,
          ``hsic_ff``, ``hsic_qq``).  Call :meth:`reset_state` between
          epochs to clear these accumulators.
        * Accumulating components and computing the ratio at the end is
          preferable to averaging per-batch CKA values because it
          avoids the non-linear-averaging bias introduced by dividing
          small HSIC estimates batch-by-batch.

    See Also:
        :class:`CKALoss`: Single-batch loss counterpart.
        :class:`fairkl.metrics.hsic.HSICMetric`: Unnormalized HSIC
            metric (simpler mean accumulation).
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        sigma_q: float = 1.0,
        kernel: str = "rbf",
        debiased: bool = False,
        name: str = "cka",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma_f = sigma_f
        self.sigma_q = sigma_q
        self.kernel = kernel
        self.debiased = debiased
        self.hsic_cross = self.add_variable(
            name="hsic_cross", shape=(), initializer="zeros"
        )
        self.hsic_ff = self.add_variable(name="hsic_ff", shape=(), initializer="zeros")
        self.hsic_qq = self.add_variable(name="hsic_qq", shape=(), initializer="zeros")

    def _center(self, K):
        if self.debiased:
            return center_gram_unbiased(K)
        return center_kernel(K)

    def _compute_kernels(self, y_true, y_pred):
        if self.kernel == "linear":
            return linear_kernel(y_pred), linear_kernel(y_true)
        return (
            rbf_kernel(y_pred, sigma=self.sigma_f),
            rbf_kernel(y_true, sigma=self.sigma_q),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)

        K_f, K_q = self._compute_kernels(y_true, y_pred)
        K_f_c = self._center(K_f)
        K_q_c = self._center(K_q)

        self.hsic_cross.assign(self.hsic_cross + ops.sum(K_f_c * K_q_c))
        self.hsic_ff.assign(self.hsic_ff + ops.sum(K_f_c * K_f_c))
        self.hsic_qq.assign(self.hsic_qq + ops.sum(K_q_c * K_q_c))

    def result(self):
        denom = ops.sqrt(
            ops.maximum(self.hsic_ff, 0.0) * ops.maximum(self.hsic_qq, 0.0)
        )
        return self.hsic_cross / ops.maximum(denom, 1e-6)

    def reset_state(self):
        self.hsic_cross.assign(0.0)
        self.hsic_ff.assign(0.0)
        self.hsic_qq.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            sigma_f=self.sigma_f,
            sigma_q=self.sigma_q,
            kernel=self.kernel,
            debiased=self.debiased,
        )
        return config
