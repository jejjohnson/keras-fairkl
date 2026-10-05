# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial Part 3 --- The fairkl Toolkit
#
# Parts 1 and 2 built fair kernel ridge regression from scratch --- solving linear systems, computing CKA by hand, writing gradient loops.  That was useful for understanding.  But research moves faster when you can *compose* reusable building blocks instead of rewriting boilerplate.
#
# This tutorial introduces the `fairkl` library, which packages those primitives into three layers of abstraction.  We will walk through each layer, explain the mathematics and engineering decisions behind it, and finish with a Pareto sweep that reproduces the fairness--accuracy trade-off from Parts 1 and 2 in a few lines of code.

# %%
from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras.ops as ops
import matplotlib.pyplot as plt
import numpy as np

import fairkl
from fairkl.layers import RBFKernelLayer, RFFLayer
from fairkl.metrics.cka import CKALoss, CKAMetric, cka_rbf
from fairkl.models import FairKernelRidge
from _style import SCATTER_KW, style_ax

# %% [markdown]
# ---
# ## From Primitives to Models
#
# ### The three-layer architecture
#
# `fairkl` is organized as a three-layer stack.  Each layer builds on the one below, and every component is independently importable and testable.
#
# | Layer | Name | Examples | Interface |
# |:-----:|:-----|:---------|:----------|
# | **0** | Primitives | `rbf_kernel`, `cka_rbf`, `solve_cholesky` | Pure functions, no state |
# | **1** | Components | `CKALoss`, `CKAMetric`, `RBFKernelLayer`, `RFFLayer` | Keras `Loss`, `Metric`, `Layer` subclasses |
# | **2** | Models | `FairKernelRidge`, `FairLinear` | Keras `Model` subclasses with `fit` / `predict` |
#
# Layer 0 does the math.  Layer 1 wraps it in Keras's training protocol (state management, serialization, batched accumulation).  Layer 2 composes Layer 1 components into end-to-end workflows.
#
# ### Why composability matters
#
# A typical research cycle looks like this: you have a new fairness metric, a new kernel approximation, or a new optimization trick.  With a monolithic codebase you rewrite the whole pipeline.  With composable layers you swap one component and everything else stays the same.
#
# For example, `FairKernelRidge` uses `cka_rbf` (Layer 0) as its fairness penalty.  If we wanted to switch to an HSIC penalty, we would only need to change that one function call --- the solver, the kernel, and the warm-start logic remain untouched.
#
# Let us now walk through the stack bottom-up, starting with the model (Layer 2) because that is the entry point most users care about, then zooming into the building blocks (Layer 1) that power it.

# %% [markdown]
# ---
# ## Same Synthetic Data
#
# We use the same synthetic dataset as Parts 1 and 2: $y = \sin(x) + 3q + \varepsilon$, with $n = 200$ samples. The coefficient 3 on $q$ ensures that an unconstrained model will learn a strong dependence on the sensitive attribute.

# %%
rng = np.random.default_rng(0)
n = 200
x = rng.standard_normal((n, 1)).astype("float32")
q = rng.standard_normal((n, 1)).astype("float32")
X = np.hstack([x, q]).astype("float32")
y = (np.sin(x.ravel()) + 3.0 * q.ravel() + 0.3 * rng.standard_normal(n)).astype(
    "float32"
)

# %% [markdown]
# ---
# ## Layer 2: The FairKernelRidge Model
#
# `FairKernelRidge` is a Keras `Model` that solves the dual-form fairness-regularized kernel ridge regression problem:
#
# $$
# \min_{\alpha} \;\;   \underbrace{\|K\alpha - y\|^2}_{\text{MSE}}   \;+\; \underbrace{\lambda \, \alpha^\top K \alpha}_{\text{ridge}}   \;+\; \underbrace{\mu \, \mathrm{CKA}(K\alpha,\; q)}_{\text{fairness}}
# $$
#
# where $K \in \mathbb{R}^{n \times n}$ is the kernel (Gram) matrix, $\alpha \in \mathbb{R}^n$ is the dual coefficient vector (from the representer theorem), $\lambda$ controls ridge regularization, and $\mu$ controls the CKA fairness penalty.
#
# The model has two training paths depending on $\mu$.

# %% [markdown]
# ### Exact mode ($\mu = 0$)
#
# With no fairness penalty the objective is quadratic in $\alpha$. Setting the gradient to zero gives the linear system:
#
# $$
# (K + \lambda I)\,\alpha = y
# $$
#
# which we solve by Cholesky factorization in $O(n^3)$ time.  The `epochs` and `lr` arguments are ignored --- the solution is exact.

# %%
model_std = FairKernelRidge(sigma=1.0, lam=0.01, mu=0.0, sigma_q=1.0)
model_std.fit(X, y)

y_std = np.array(model_std.predict(X)).ravel()
mse_std = float(np.mean((y_std - y) ** 2))
cka_std = float(cka_rbf(y_std.reshape(-1, 1).astype("float32"), q))
print(f"Standard KRR:  MSE = {mse_std:.4f},  CKA = {cka_std:.4f}")

# %% [markdown]
# The MSE is low because the model has enough capacity to fit both the $\sin(x)$ and the $3q$ components.  But the CKA is high --- the predictions are strongly aligned with the sensitive attribute, exactly as we would expect from the data-generating process.

# %% [markdown]
# ### Fair mode ($\mu > 0$): warm-started gradient descent
#
# When $\mu > 0$, the CKA term is non-linear in $\alpha$ (it involves an RBF kernel on the *predictions* $K\alpha$), so no closed-form solution exists.  Instead, `fit()` does:
#
# 1. **Warm-start**: solve the exact ($\mu = 0$) system to get
#    $\alpha_0 = (K + \lambda I)^{-1} y$
# 2. **Optimize**: run Adam gradient descent on the full objective,
#    initialized at $\alpha_0$
#
# **Engineering trick: warm-starting.**  Each model in a Pareto sweep is independently warm-started from its own exact ($\mu = 0$) solution. This is done internally by `model.fit()`.  The warm start typically reduces the number of epochs needed from ~1000 (random init) to ~100--200, because the exact solution is already a good predictor --- we are only "nudging" it toward fairness.

# %%
model_fair = FairKernelRidge(sigma=1.0, lam=0.01, mu=10.0, sigma_q=1.0)
model_fair.fit(X, y, q=q, epochs=200, lr=0.005)

y_fair = np.array(model_fair.predict(X)).ravel()
mse_fair = float(np.mean((y_fair - y) ** 2))
cka_fair = float(cka_rbf(y_fair.reshape(-1, 1).astype("float32"), q))
print(f"Fair KRR (mu=10):  MSE = {mse_fair:.4f},  CKA = {cka_fair:.4f}")

# %% [markdown]
# The fairness penalty has done its job: CKA dropped substantially at the cost of a moderate MSE increase.  The model has learned to suppress the $3q$ component while preserving the $\sin(x)$ signal.
#
# Let us visualize the effect.  In the standard model, predictions track $q$ closely (sloped scatter).  In the fair model, the predictions are nearly flat with respect to $q$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, yh, title in zip(
    axes,
    [y_std, y_fair],
    [f"Standard  (CKA={cka_std:.3f})", f"Fair mu=10  (CKA={cka_fair:.3f})"],
):
    ax.scatter(q.ravel(), yh, c="C1", **SCATTER_KW)
    ax.set_xlabel("Sensitive attribute  q")
    ax.set_ylabel("Prediction  y_hat")
    ax.set_title(title, fontsize=11)
    style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inspecting internals: dual coefficients and the kernel matrix
#
# For diagnostics, `FairKernelRidge` exposes the learned dual coefficients $\alpha$ and the cross-kernel matrix $K(X_{\text{test}}, X_{\text{train}})$.
#
# The dual coefficients define the predictor via the representer theorem: $f(x) = \sum_{i=1}^{n} \alpha_i \, k(x, x_i)$.  Their magnitudes tell us how much each training point influences predictions.

# %%
alpha = np.array(model_std.get_alpha())
print(f"alpha shape: {alpha.shape},  ||alpha|| = {np.linalg.norm(alpha):.4f}")

K_cross = np.array(model_std.get_kernel_matrix(X[:5]))
print(f"K(X[:5], X_train) shape: {K_cross.shape}")

# %% [markdown]
# ---
# ## Layer 1: Keras Building Blocks
#
# Layer 2 models are built from Layer 1 components: Keras `Loss`, `Metric`, and `Layer` subclasses.  These are independently usable --- you can plug them into any Keras training loop, not just `FairKernelRidge`.

# %% [markdown]
# ### CKALoss --- a differentiable fairness penalty
#
# `CKALoss` is a standard Keras `Loss` subclass that computes CKA (Centered Kernel Alignment) between predictions and sensitive attributes.  In the Keras convention, `y_true` carries the sensitive attribute $q$ and `y_pred` carries the model output $f$.
#
# The formula is:
#
# $$
# \mathrm{CKA}(f, q) =   \frac{\mathrm{tr}(\tilde{K}_f \, \tilde{K}_q)}   {\sqrt{\mathrm{tr}(\tilde{K}_f^2) \cdot \mathrm{tr}(\tilde{K}_q^2)}}
# $$
#
# where $\tilde{K}_f$ and $\tilde{K}_q$ are the centered Gram matrices of the predictions and sensitive attributes respectively, using RBF kernels.  The normalization ensures $\mathrm{CKA} \in [0, 1]$ regardless of data scale, which makes the penalty weight $\mu$ interpretable across datasets.
#
# Let us verify that `CKALoss` matches the standalone `cka_rbf` function.

# %%
loss_fn = CKALoss(sigma_f=1.0, sigma_q=1.0)

loss_std = float(loss_fn(q, y_std.reshape(-1, 1).astype("float32")))
loss_fair = float(loss_fn(q, y_fair.reshape(-1, 1).astype("float32")))

cka_std_check = float(cka_rbf(y_std.reshape(-1, 1).astype("float32"), q))
cka_fair_check = float(cka_rbf(y_fair.reshape(-1, 1).astype("float32"), q))

print(f"CKALoss(std)  = {loss_std:.4f}   vs   cka_rbf = {cka_std_check:.4f}")
print(f"CKALoss(fair) = {loss_fair:.4f}   vs   cka_rbf = {cka_fair_check:.4f}")

# %% [markdown]
# The values match exactly.  `CKALoss` is just a Keras-compatible wrapper around the Layer 0 primitive --- it adds `get_config` serialization and automatic rank promotion, but the math is identical.

# %% [markdown]
# ### CKAMetric --- tracking fairness across batches
#
# In a training loop we need to track CKA over an entire epoch, not just per batch.  `CKAMetric` is a Keras `Metric` subclass that does this correctly via **component accumulation**.
#
# The metric maintains three running sums across batches:
#
# | Variable | Accumulated quantity |
# |:---------|:--------------------|
# | `hsic_cross` | $\sum_b \mathrm{tr}(\tilde{K}_f^{(b)} \, \tilde{K}_q^{(b)})$ |
# | `hsic_ff` | $\sum_b \mathrm{tr}((\tilde{K}_f^{(b)})^2)$ |
# | `hsic_qq` | $\sum_b \mathrm{tr}((\tilde{K}_q^{(b)})^2)$ |
#
# At `result()` time, it computes:
#
# $$
# \mathrm{CKA}_{\text{epoch}} =   \frac{\sum_b \mathrm{HSIC}_{\text{cross}}^{(b)}}   {\sqrt{\sum_b \mathrm{HSIC}_{ff}^{(b)} \cdot \sum_b \mathrm{HSIC}_{qq}^{(b)}}}
# $$
#
# ### Engineering trick: component accumulation avoids ratio bias
#
# **Engineering trick: component accumulation.**  Why not just average the per-batch CKA values?  Because CKA is a *ratio*, and by Jensen's inequality, the expectation of a ratio is not the ratio of expectations:
#
# $$
# \mathbb{E}\!\left[\frac{X}{Y}\right] \;\neq\; \frac{\mathbb{E}[X]}{\mathbb{E}[Y]}
# $$
#
# A running average of per-batch CKA values would compute the left side, introducing a systematic bias (typically upward for convex $f(x) = 1/x$). The running-sum approach computes the right side, which is the correct estimator for fixed-size batches because the $1/n^2$ normalization factor is constant and cancels in the ratio.

# %%
metric = CKAMetric(sigma_f=1.0, sigma_q=1.0)

# Simulate two batches
batch1_pred = y_std[:100].reshape(-1, 1).astype("float32")
batch1_q = q[:100]
batch2_pred = y_std[100:].reshape(-1, 1).astype("float32")
batch2_q = q[100:]

metric.update_state(batch1_q, batch1_pred)
print(f"After batch 1:  CKA = {float(metric.result()):.4f}")

metric.update_state(batch2_q, batch2_pred)
print(f"After batch 2:  CKA = {float(metric.result()):.4f}")

metric.reset_state()
print(f"After reset:    CKA = {float(metric.result()):.4f}")

# %% [markdown]
# After `reset_state()`, the metric returns 0 because the numerator (cross-HSIC) is zero and the denominator is floored to $10^{-6}$. This is the expected behavior at the start of each epoch.

# %% [markdown]
# ### Kernel layers: RBFKernelLayer and RFFLayer
#
# `fairkl` provides Keras `Layer` subclasses that wrap kernel computations.  These are composable building blocks for custom architectures.
#
# ### Engineering trick: log-sigma reparameterization
#
# **Engineering trick: log-sigma reparameterization.**  `RBFKernelLayer` stores `log(sigma)` as the trainable parameter, then computes `sigma = exp(log_sigma)` in the forward pass.  This serves two purposes:
#
# 1. **Positivity constraint without clipping**: since $\exp$ maps
#    $\mathbb{R} \to \mathbb{R}^+$, we get $\sigma > 0$ for free,    without gradient-killing clipping or projections.
#
# 2. **Natural gradient scaling**: the gradient of $\sigma$ with respect
#    to $\log\sigma$ is $d\sigma / d(\log\sigma) = \sigma$.  This means    the gradient *scales with the parameter value*, which is exactly the    right behavior for bandwidths --- a step of $\Delta(\log\sigma) = 0.1$    changes $\sigma$ by ~10% regardless of whether $\sigma = 0.1$ or    $\sigma = 10$.  Without the reparameterization, the same absolute    step size would overshoot for small $\sigma$ and undershoot for    large $\sigma$.

# %%
rbf_layer = RBFKernelLayer(sigma_init=1.0, trainable_sigma=True)
K_layer = np.array(rbf_layer(X))
K_prim = np.array(fairkl.rbf_kernel(X, sigma=1.0))

diff = float(np.max(np.abs(K_layer - K_prim)))
print(f"Max |K_layer - K_primitive| = {diff:.2e}  (should be ~0)")
sigma_val = float(ops.exp(rbf_layer.log_sigma))
print(
    f"Trainable sigma: log_sigma = {float(rbf_layer.log_sigma):.4f} -> sigma = {sigma_val:.4f}"
)

# %% [markdown]
# The layer produces exactly the same Gram matrix as the Layer 0 primitive.  The difference is that the layer carries state (the trainable `log_sigma` weight), so it can be optimized end-to-end in a Keras training loop.

# %% [markdown]
# Now let us look at `RFFLayer`, the Random Fourier Features approximation.  The core idea (Rahimi & Recht, 2007) is to replace the $n \times n$ kernel matrix with a low-rank approximation via a random feature map:
#
# $$
# Z(x) = \sqrt{\frac{2}{D}} \cos\!\left(\frac{\omega^\top x}{\sigma} + b\right)
# $$
#
# where $\omega \sim \mathcal{N}(0, I)$ and $b \sim \mathrm{Uniform}(0, 2\pi)$ are sampled once at layer construction time.  The key property is that inner products in feature space approximate the RBF kernel:
#
# $$
# Z(x)^\top Z(x') \;\approx\; k(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
# $$
#
# The approximation error decreases as $O(1/\sqrt{D})$ in the number of random features $D$.
#
# **Engineering trick: when to use RFF.**  For $n > \sim 1000$, the $O(n^2)$ kernel matrix becomes expensive to store and compute.  RFF maps to $D$ features with $O(n \cdot d \cdot D)$ cost, then any linear method can be applied.  Rule of thumb: $D \sim 10d$ to $100d$ gives good approximation quality.

# %%
rff_layer = RFFLayer(n_features=500, sigma_init=1.0, seed=42)
Z_rff = np.array(rff_layer(X))
K_rff = Z_rff @ Z_rff.T

err = np.linalg.norm(K_prim - K_rff) / np.linalg.norm(K_prim)
print(f"RFF (D=500):  relative error ||K - ZZ'||_F / ||K||_F = {err:.4f}")

# %% [markdown]
# The relative Frobenius-norm error $\|K - ZZ^\top\|_F / \|K\|_F$ gives us a single number summarizing approximation quality.  With $D = 500$ features and $d = 2$ input dimensions, we are well above the $D \sim 100d$ rule of thumb, so the error should be small.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(K_prim, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
axes[0].set_title("Exact RBF Kernel", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(K_rff, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
axes[1].set_title(f"RFF Approximation (D=500)\nerr = {err:.3f}", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# The two kernel matrices are visually indistinguishable.  The approximation error is small enough that downstream tasks (regression, CKA computation) would produce nearly identical results with the approximate kernel.

# %% [markdown]
# ---
# ## Serialization & Reproducibility
#
# ### get_config round-trip
#
# All `fairkl` components implement `get_config()` for Keras serialization.  This means you can save a model's hyperparameters, reconstruct it later, and get identical behavior.  This is essential for reproducibility in research workflows.

# %%
config = model_fair.get_config()
print("Model config:")
for k, v in config.items():
    if k != "name":
        print(f"  {k}: {v}")

# %% [markdown]
# The config contains every constructor argument needed to recreate the model.  Combined with saved dual coefficients, this gives full reproducibility.  The same `get_config` pattern works for `CKALoss`, `CKAMetric`, `RBFKernelLayer`, and `RFFLayer`.

# %% [markdown]
# ---
# ## Putting It Together: Pareto Sweep
#
# ### Warm-started mu sweep
#
# We now sweep over fairness penalty strengths $\mu \in \{0, 1, 5, 10, 20\}$ and record MSE and CKA for each.  Each model is independently constructed and fitted --- the warm-start from the exact solution happens automatically inside `fit()`.
#
# **Engineering trick: warm-starting.**  Each model in the Pareto sweep is independently warm-started from its own exact ($\mu = 0$) solution. This is done internally by `model.fit()`.  The warm start means we can use a modest epoch budget (200) and a small learning rate (0.005) and still get good convergence, because we are starting near the optimum of the unconstrained problem and only need to move along the fairness--accuracy trade-off surface.

# %%
mus = [0, 1, 5, 10, 20]
mse_list, cka_list = [], []

for mu in mus:
    model = FairKernelRidge(sigma=1.0, lam=0.01, mu=mu, sigma_q=1.0)
    model.fit(X, y, q=q, epochs=200, lr=0.005)
    yh = np.array(model.predict(X)).ravel()
    mse = float(np.mean((yh - y) ** 2))
    cka_val = float(cka_rbf(yh.reshape(-1, 1).astype("float32"), q))
    mse_list.append(mse)
    cka_list.append(cka_val)
    print(f"mu = {mu:5.1f}   MSE = {mse:.3f}   CKA = {cka_val:.3f}")

# %% [markdown]
# As $\mu$ increases, CKA decreases (fairer predictions) while MSE increases (less accurate predictions).  This is the fundamental fairness--accuracy trade-off: removing information about $q$ from the predictions necessarily removes some predictive signal, since $y$ genuinely depends on $q$ in our data-generating process.

# %% [markdown]
# ### Prediction diagnostics
#
# Let us visualize predictions for three representative points on the Pareto frontier: $\mu = 0$ (no fairness), $\mu = 5$ (moderate), and $\mu = 20$ (strong fairness).  We look at the scatter of predictions versus the sensitive attribute $q$ --- a fair model should show no trend.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, mu_sel in zip(axes, [0, 5, 20]):
    idx = mus.index(mu_sel)
    model = FairKernelRidge(sigma=1.0, lam=0.01, mu=mu_sel, sigma_q=1.0)
    model.fit(X, y, q=q, epochs=200, lr=0.005)
    yh = np.array(model.predict(X)).ravel()
    corr = np.corrcoef(yh, q.ravel())[0, 1]
    ax.scatter(q.ravel(), yh, c="C1", **SCATTER_KW)
    ax.set_xlabel("Sensitive attribute  q")
    ax.set_ylabel("Prediction  y_hat")
    ax.set_title(
        f"mu={mu_sel}   corr={corr:.2f}   CKA={cka_list[idx]:.3f}",
        fontsize=10,
    )
    style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# The progression is clear:
#
# - **$\mu = 0$**: strong linear trend --- predictions track $q$ closely
#   (correlation near 1), because the model freely uses the $3q$ signal.
# - **$\mu = 5$**: the slope is reduced --- the model partially suppresses
#   the $q$-dependent component to satisfy the fairness penalty.
# - **$\mu = 20$**: the scatter is nearly flat --- the model has
#   effectively removed the $q$ signal, at the cost of higher MSE.

# %% [markdown]
# ---
# ## Results: Fairness vs Accuracy
#
# The Pareto frontier traces the achievable (CKA, MSE) pairs.  Points to the lower-left are better on both axes, but the curve shows that improving one metric necessarily worsens the other.  The "elbow" of the curve is typically the best operating point in practice --- it gives the largest fairness improvement per unit of accuracy lost.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cka_list, mse_list, "o-", color="C0", lw=2, markersize=8)
for i, mu in enumerate(mus):
    ax.annotate(
        f"mu={mu}",
        (cka_list[i], mse_list[i]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
    )
ax.set_xlabel("CKA   (0 = fair,  1 = unfair)")
ax.set_ylabel("MSE")
ax.set_title("Fairness vs Accuracy --- Pareto Frontier", fontsize=12)
style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Summary.** The `fairkl` toolkit provides a three-layer architecture --- primitives, Keras components, and models --- that makes fair kernel learning composable and reproducible.  The key engineering decisions (component accumulation for CKA tracking, log-sigma reparameterization for stable bandwidth optimization, warm-starting from exact solutions) are baked into the building blocks, so you get correct and efficient behavior without writing boilerplate.
#
# The Pareto frontier confirms the same fairness--accuracy trade-off as Parts 1 and 2, validating that the toolkit correctly implements the theory we developed from scratch.
#
# ---
#
# **Next**: [Part 4](tutorial_fair_krr_part4.py) uses Keras Tuner for automated hyperparameter sweeps over the full search space.
