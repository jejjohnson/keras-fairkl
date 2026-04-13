# fairkl

[![Tests](https://github.com/jejjohnson/keras-fairkl/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/keras-fairkl/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/keras-fairkl/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/keras-fairkl/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fairness-constrained kernel learning for Keras 3. Composable HSIC-based penalties, exact and approximate kernels, and fair models that work with any Keras backend.

## Installation

```bash
pip install fairkl
```

## Quick Start

```python
import fairkl

# Compute an RBF kernel matrix
K = fairkl.rbf_kernel(X, sigma=0.5)

# Measure statistical dependence via HSIC
hsic_val = fairkl.hsic_rbf(predictions, sensitive_attrs)

# Fair kernel ridge regression
model = fairkl.FairKernelRidge(sigma=0.5, lam=1e-3, mu=1.0)
model.fit(X_train, y_train, q=sensitive_train, epochs=100)
y_pred = model.predict(X_test)
```

## Architecture

Three-layer stack built on pure `keras.ops`:

| Layer | Contents |
|-------|----------|
| **Layer 0 -- Primitives** | Pure functions: `rbf_kernel`, `hsic_biased`, `solve_cg`, `center_kernel`, ... |
| **Layer 1 -- Components** | Keras layers (`RBFKernelLayer`, `RFFLayer`), losses (`HSICLoss`), metrics |
| **Layer 2 -- Models** | `FairKernelRidge`, `FairLinear`, `FairPCA`, `FairKernelPCA` |

## Development

```bash
make install      # Install all dependency groups
make test         # Run tests
make lint         # Lint with ruff
make format       # Format with ruff
make typecheck    # Type check with ty
```

## License

MIT
