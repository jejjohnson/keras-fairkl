"""fairkl: Fairness-constrained kernel learning for Keras 3."""

from __future__ import annotations


__version__ = "0.1.5"

# Layer 0: Kernel primitives
from fairkl.kernels.approximate import (
    nystrom_approximate as nystrom_approximate,
    random_fourier_features as random_fourier_features,
    random_kitchen_sinks as random_kitchen_sinks,
)
from fairkl.kernels.exact import (
    linear_kernel as linear_kernel,
    polynomial_kernel as polynomial_kernel,
    rbf_kernel as rbf_kernel,
)

# Layer 1: Keras Layers
from fairkl.layers.kernel_layers import (
    LinearKernelLayer as LinearKernelLayer,
    NystromLayer as NystromLayer,
    RBFKernelLayer as RBFKernelLayer,
    RFFLayer as RFFLayer,
    RKSLayer as RKSLayer,
)

# Layer 0: Metrics primitives
# Layer 1: Keras Loss/Metric
from fairkl.metrics.cka import (
    CKALoss as CKALoss,
    CKAMetric as CKAMetric,
    center_gram_unbiased as center_gram_unbiased,
    cka_biased as cka_biased,
    cka_debiased as cka_debiased,
    cka_linear as cka_linear,
    cka_rbf as cka_rbf,
)
from fairkl.metrics.hsic import (
    HSICLoss as HSICLoss,
    HSICMetric as HSICMetric,
    hsic_biased as hsic_biased,
    hsic_linear as hsic_linear,
    hsic_rbf as hsic_rbf,
)
from fairkl.metrics.mmd import (
    MMDLoss as MMDLoss,
    MMDMetric as MMDMetric,
    mmd_rbf as mmd_rbf,
)

# Layer 2: Models
from fairkl.models.fair_kernel_pca import FairKernelPCA as FairKernelPCA
from fairkl.models.fair_kernel_ridge import FairKernelRidge as FairKernelRidge
from fairkl.models.fair_linear import FairLinear as FairLinear
from fairkl.models.fair_pca import FairPCA as FairPCA

# Layer 0: Ops
from fairkl.ops.centering import (
    center_kernel as center_kernel,
    centering_matrix as centering_matrix,
)
from fairkl.ops.solvers import (
    solve_cg as solve_cg,
    solve_cholesky as solve_cholesky,
)
