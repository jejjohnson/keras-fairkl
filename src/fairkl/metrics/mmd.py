"""MMD: Maximum Mean Discrepancy.

Pure functions and Keras Loss/Metric classes for computing the squared
Maximum Mean Discrepancy (MMD), a kernel-based distance between
probability distributions.  MMD is zero if and only if the two
distributions are identical (when a universal kernel such as the RBF
kernel is used).

References:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B.,
    & Smola, A. (2012). "A Kernel Two-Sample Test", JMLR 13.
"""

from __future__ import annotations

import keras
import keras.ops as ops

from fairkl.kernels.exact import rbf_kernel


# ---------------------------------------------------------------------------
# Pure function (Layer 0)
# ---------------------------------------------------------------------------


def mmd_rbf(X, Y, sigma: float = 1.0):
    """Squared Maximum Mean Discrepancy with RBF kernel.

    Computes the biased estimator of the squared MMD between two
    empirical distributions:

    .. math::

        \\widehat{\\text{MMD}}^2(P, Q) =
        \\frac{1}{n^2} \\sum_{i,j} k(x_i, x_j)
        + \\frac{1}{m^2} \\sum_{i,j} k(y_i, y_j)
        - \\frac{2}{nm} \\sum_{i,j} k(x_i, y_j)

    which simplifies to ``mean(K_xx) + mean(K_yy) - 2 * mean(K_xy)``.

    MMD is a metric on the space of probability distributions (when
    using a universal kernel): :math:`\\text{MMD}^2(P, Q) = 0` if and
    only if :math:`P = Q`.  Unlike HSIC and CKA, MMD measures
    *distributional distance* rather than statistical dependence.

    Args:
        X (array-like): First sample set of shape ``(n, d)`` where *n*
            is the number of samples and *d* is the feature dimension.
        Y (array-like): Second sample set of shape ``(m, d)`` where *m*
            is the number of samples.  Must have the same feature
            dimension *d* as *X*.
        sigma (float): Bandwidth (length-scale) for the RBF kernel
            :math:`k(x,y) = \\exp(-\\|x-y\\|^2 / 2\\sigma^2)`.
            Defaults to ``1.0``.

    Returns:
        Scalar tensor (shape ``()``, same dtype as *X*) containing the
        squared MMD estimate.  Non-negative for the biased estimator.

    Examples:
        >>> import keras.ops as ops
        >>> from fairkl.metrics.mmd import mmd_rbf
        >>> X = ops.convert_to_tensor([[0.0, 0.0], [1.0, 1.0]])
        >>> Y = ops.convert_to_tensor([[5.0, 5.0], [6.0, 6.0]])
        >>> mmd_rbf(X, Y, sigma=1.0)  # large value (distributions far apart)

    Note:
        * **Computational complexity**: :math:`O((n+m)^2 \\cdot d)` --
          dominated by the three pairwise kernel matrix computations.
        * **Bias**: This is the *biased* estimator (includes
          :math:`k(x_i, x_i)` terms).  The biased estimator is
          non-negative, consistent, and has lower variance than the
          unbiased U-statistic estimator.
        * **Bandwidth selection**: The bandwidth ``sigma`` critically
          affects sensitivity.  A common heuristic is the median of
          pairwise distances in the combined sample.

    References:
        Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B.,
        & Smola, A. (2012). "A Kernel Two-Sample Test", JMLR 13.

    See Also:
        :func:`fairkl.metrics.hsic.hsic_rbf`: HSIC measures
            dependence rather than distributional distance.
        :func:`fairkl.metrics.cka.cka_rbf`: Normalised dependence
            measure in [0, 1].
    """
    K_xx = rbf_kernel(X, X, sigma=sigma)
    K_yy = rbf_kernel(Y, Y, sigma=sigma)
    K_xy = rbf_kernel(X, Y, sigma=sigma)
    return ops.mean(K_xx) + ops.mean(K_yy) - 2.0 * ops.mean(K_xy)


# ---------------------------------------------------------------------------
# Keras Loss (Layer 1)
# ---------------------------------------------------------------------------


class MMDLoss(keras.losses.Loss):
    """Squared MMD as a Keras loss for distribution matching.

    Minimising this loss encourages the distribution of ``y_pred`` to
    match that of ``y_true``.  This is useful, for example, when
    enforcing that predictions conditioned on different sensitive groups
    have the same marginal distribution (demographic parity).

    ``y_true`` and ``y_pred`` are treated as two empirical sample sets
    and the squared MMD is computed via :func:`mmd_rbf`.  Both inputs
    are automatically promoted to 2-D if passed as rank-1 tensors.

    Unlike :class:`fairkl.metrics.hsic.HSICLoss` and
    :class:`fairkl.metrics.cka.CKALoss`, which measure *dependence*
    between predictions and sensitive attributes, MMDLoss measures
    *distributional distance* between two sets of samples.

    Args:
        sigma (float): Bandwidth (length-scale) for the RBF kernel.
            Defaults to ``1.0``.
        kernel (str): Kernel type.  Currently only ``"rbf"`` is
            supported.  Defaults to ``"rbf"``.
        name (str): Name of the loss instance.  Defaults to
            ``"mmd_loss"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.losses.Loss``.

    Examples:
        >>> from fairkl.metrics.mmd import MMDLoss
        >>> loss_fn = MMDLoss(sigma=1.0)
        >>> loss = loss_fn(y_true=group_a_outputs, y_pred=group_b_outputs)

    See Also:
        :class:`fairkl.metrics.cka.CKALoss`: Normalised dependence
            penalty (bounded [0, 1]).
        :class:`fairkl.metrics.hsic.HSICLoss`: Unnormalized dependence
            penalty.
        :class:`MMDMetric`: Accumulating metric counterpart.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel: str = "rbf",
        name: str = "mmd_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma = sigma
        self.kernel = kernel

    def call(self, y_true, y_pred):
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)
        return mmd_rbf(y_pred, y_true, sigma=self.sigma)

    def get_config(self):
        config = super().get_config()
        config.update(sigma=self.sigma, kernel=self.kernel)
        return config


# ---------------------------------------------------------------------------
# Keras Metric (Layer 1)
# ---------------------------------------------------------------------------


class MMDMetric(keras.metrics.Metric):
    """Keras metric that tracks running-average squared MMD over batches.

    Accumulates a running sum of per-batch :math:`\\widehat{\\text{MMD}}^2`
    values and divides by the batch count at :meth:`result` time,
    yielding a simple mean accumulation estimate.  This is exact when
    all batches have the same size; for variable-size batches the mean
    is approximate.

    Args:
        sigma (float): Bandwidth (length-scale) for the RBF kernel.
            Defaults to ``1.0``.
        kernel (str): Kernel type.  Currently only ``"rbf"`` is
            supported.  Defaults to ``"rbf"``.
        name (str): Name of the metric instance.  Defaults to
            ``"mmd"``.
        **kwargs: Additional keyword arguments forwarded to
            ``keras.metrics.Metric``.

    Examples:
        >>> from fairkl.metrics.mmd import MMDMetric
        >>> metric = MMDMetric(sigma=1.0)
        >>> metric.update_state(y_true=group_a, y_pred=group_b)
        >>> metric.result()

    Note:
        The metric maintains two scalar variables: ``total`` (cumulative
        :math:`\\text{MMD}^2`) and ``count`` (number of batches).  Call
        :meth:`reset_state` between epochs to clear these accumulators.

    See Also:
        :class:`MMDLoss`: Single-batch loss counterpart.
        :class:`fairkl.metrics.cka.CKAMetric`: CKA metric with
            component-wise accumulation.
        :class:`fairkl.metrics.hsic.HSICMetric`: HSIC metric with
            mean accumulation.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel: str = "rbf",
        name: str = "mmd",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sigma = sigma
        self.kernel = kernel
        self.total = self.add_variable(name="total", shape=(), initializer="zeros")
        self.count = self.add_variable(name="count", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(ops.shape(y_true)) == 1:
            y_true = ops.expand_dims(y_true, axis=-1)
        if len(ops.shape(y_pred)) == 1:
            y_pred = ops.expand_dims(y_pred, axis=-1)
        val = mmd_rbf(y_pred, y_true, sigma=self.sigma)
        self.total.assign(self.total + val)
        self.count.assign(self.count + 1.0)

    def result(self):
        return self.total / ops.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(sigma=self.sigma, kernel=self.kernel)
        return config
