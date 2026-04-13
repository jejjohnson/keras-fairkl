"""Kernel functions: exact and approximate."""

from __future__ import annotations

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
