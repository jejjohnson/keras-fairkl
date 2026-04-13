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
# # Tutorial Part 1 — Primitives & the Fair Objective
#
# Machine-learning models trained on historical data can inherit and amplify societal biases. A hiring model trained on past decisions may learn that gender predicts job performance — not because it does, but because the training signal encodes past discrimination. Kernel methods are no exception: when the target variable depends on a sensitive attribute, the optimal predictor will happily exploit that dependence.
#
# This tutorial builds **Fair Kernel Ridge Regression** from first principles. We use only `fairkl` primitives and `keras.ops` so that every matrix operation is visible. By the end we will have a principled fairness penalty (CKA), a combined loss function, and a Pareto frontier that visualizes the accuracy--fairness trade-off.
#
# **What we cover**
#
# | Section | Key idea |
# |---------|----------|
# | The Problem | Why kernels + biased labels = biased predictions |
# | Background: KRR | Kernel matrix, centering, dual solution, bandwidth selection |
# | Diagnosing Unfairness | HSIC, CKA, and why normalization matters |
# | The Fair Objective | Three-term loss, loss decomposition, warm-starting |
# | Results | Pareto frontier over the penalty weight $\mu$ |

# %%
from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras.ops as ops
import matplotlib.pyplot as plt
import numpy as np

import fairkl
from _style import SCATTER_KW, style_ax

# %% [markdown]
# ## The Problem: Biased Predictions
#
# ### Synthetic data
#
# We need a dataset where the answer is known in advance. Our data-generating process is $y = \sin(x) + 3q + \varepsilon$, where $x$ is a legitimate feature, $q$ is a continuous sensitive attribute, and $\varepsilon \sim \mathcal{N}(0, 0.09)$ is noise. The coefficient on $q$ is large on purpose: the target is **strongly coupled** to the sensitive attribute, so any model that minimizes MSE will learn to exploit $q$.
#
# We draw $n = 200$ points with both $x$ and $q$ sampled from a standard normal.

# %%
rng = np.random.default_rng(0)
n = 200
x = rng.standard_normal((n, 1)).astype("float32")
q = rng.standard_normal((n, 1)).astype("float32")
X = np.hstack([x, q]).astype("float32")
y = (np.sin(x.ravel()) + 3.0 * q.ravel() + 0.3 * rng.standard_normal(n)).astype(
    "float32"
)

print(f"n = {n},  d = {X.shape[1]}")
print(f"Corr(y, q) = {np.corrcoef(y, q.ravel())[0, 1]:.3f}")

# %% [markdown]
# The left panel below shows that $y$ has a clear nonlinear dependence on $x$ (the sine wave), while the color gradient reveals the strong linear coupling with $q$. The right panel confirms a correlation above 0.9 between $y$ and the sensitive attribute.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sc = axes[0].scatter(x.ravel(), y, c=q.ravel(), cmap="coolwarm", **SCATTER_KW)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("y  vs  x   (colored by q)")
plt.colorbar(sc, ax=axes[0], label="q")
style_ax(axes[0])

axes[1].scatter(q.ravel(), y, c="C1", **SCATTER_KW)
axes[1].set_xlabel("Sensitive attribute  q")
axes[1].set_ylabel("y")
axes[1].set_title(f"y  vs  q   (corr = {np.corrcoef(y, q.ravel())[0, 1]:.2f})")
style_ax(axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why kernels? A nonlinear regression problem
#
# The sine component means a linear model cannot capture the full relationship between $x$ and $y$. Kernel methods let us work in an implicit high-dimensional feature space without ever computing the features explicitly. The radial basis function (RBF) kernel is the default choice because it is universal — it can approximate any continuous function given enough data.

# %% [markdown]
# ## Background: Kernel Ridge Regression
#
# Before we can make KRR fair, we need to understand the standard version. This section walks through each building block: the kernel matrix, centering, the dual solution, and bandwidth selection.
#
# ### The kernel matrix
#
# The RBF kernel measures similarity between data points in a feature space of infinite dimension:
#
# $$K_{ij} = \exp\!\Bigl(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\Bigr)$$
#
# The bandwidth $\sigma$ controls how quickly similarity decays with distance. When $\sigma$ is small the kernel is sharply peaked and each point only "sees" its nearest neighbours; when $\sigma$ is large the kernel is nearly flat and every point looks similar to every other. The matrix $K$ is symmetric positive semi-definite by construction.

# %%
sigma = 1.0
K = np.array(fairkl.rbf_kernel(X, sigma=sigma))

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(K, cmap="RdBu_r", aspect="auto")
ax.set_title(f"RBF kernel matrix  (sigma = {sigma})", fontsize=11)
ax.set_xlabel("sample j")
ax.set_ylabel("sample i")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# The heatmap shows a block-like structure because our 200 samples are unsorted. The diagonal is always 1 (a point is maximally similar to itself), and off-diagonal entries decay with squared Euclidean distance.
#
# ### Centering in feature space
#
# Many statistical quantities (covariance, HSIC) require zero-mean data. In kernel methods we center implicitly via the centering matrix $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$, giving $\tilde{K} = HKH$. Naively forming $H$ costs $O(n^2)$ storage and the double matrix product costs $O(n^3)$. The efficient formula avoids forming $H$ entirely:
#
# $$\tilde{K}_{ij} = K_{ij} - \bar{K}_{i\cdot} - \bar{K}_{\cdot j} + \bar{K}_{\cdot\cdot}$$
#
# where $\bar{K}_{i\cdot}$ is the mean of row $i$, $\bar{K}_{\cdot j}$ is the mean of column $j$, and $\bar{K}_{\cdot\cdot}$ is the grand mean. This runs in $O(n^2)$ time and requires only three passes over the matrix.

# %%
K_centered = np.array(fairkl.center_kernel(K))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(K, cmap="RdBu_r", aspect="auto")
axes[0].set_title("Original K", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(K_centered, cmap="RdBu_r", aspect="auto")
axes[1].set_title("Centered K", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# After centering, the row and column means are exactly zero. The color scale now spans both positive and negative values, reflecting the centered inner products in feature space.
#
# ### The dual solution
#
# Standard KRR minimizes a ridge-regularized squared loss in the dual (kernel) representation:
#
# $$\min_\alpha \;\|K\alpha - y\|^2 + \lambda\,\alpha^\top K\alpha$$
#
# Setting the gradient to zero gives the closed-form solution $\alpha = (K + \lambda I)^{-1} y$. We solve this via **Cholesky factorization** rather than matrix inversion: factor $K + \lambda I = LL^\top$ in $O(n^3/3)$ flops, then solve two triangular systems in $O(n^2)$ each. This is numerically stable and roughly twice as fast as a full inverse.
#
# ### Engineering trick: condition number
#
# Adding $\lambda I$ to the kernel matrix improves its condition number from $\kappa(K)$ to at most $\kappa(K + \lambda I) \le (\|K\| + \lambda) / \lambda$. Even a small $\lambda = 0.01$ keeps the Cholesky solve well-conditioned on our $n = 200$ problem.

# %%
lam = 0.01
K_t = ops.convert_to_tensor(K, dtype="float32")
y_t = ops.convert_to_tensor(y.reshape(-1, 1), dtype="float32")
n_t = ops.shape(K_t)[0]

system = K_t + lam * ops.eye(n_t)
alpha = fairkl.solve_cholesky(system, y_t)

y_pred_std = np.array(ops.matmul(K_t, alpha)).ravel()
mse_std = float(np.mean((y_pred_std - y) ** 2))
print(f"Standard KRR — MSE = {mse_std:.4f}")

# %% [markdown]
# With $\lambda = 0.01$ the standard KRR fit is nearly perfect. The scatter below shows predicted vs true values hugging the diagonal.

# %%
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, y_pred_std, c="C0", **SCATTER_KW)
lo, hi = y.min() - 0.5, y.max() + 0.5
ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)
ax.set_xlabel("True y")
ax.set_ylabel("Predicted ŷ")
ax.set_title(f"Standard KRR  (MSE = {mse_std:.3f})", fontsize=11)
style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Engineering trick: bandwidth selection
#
# How should we choose $\sigma$? A common heuristic is the **median heuristic**: set $\sigma$ to the median of all pairwise distances, $\sigma_{\text{med}} = \text{median}\{\|x_i - x_j\| : i < j\}$. This ensures the kernel is neither too peaked nor too flat for the given data scale.
#
# The grid below shows four bandwidths spanning two orders of magnitude. At $\sigma = 0.1$ the kernel matrix is nearly diagonal (each point only sees itself) and the model memorizes the training set. At $\sigma = 3.0$ the kernel is nearly constant and the model underfits. The best fit is near $\sigma = 1.0$, which is close to the median heuristic for our data.

# %%
sigmas = [0.1, 0.5, 1.0, 3.0]
fig, axes = plt.subplots(2, len(sigmas), figsize=(16, 7))

for j, s in enumerate(sigmas):
    K_s = np.array(fairkl.rbf_kernel(X, sigma=s))
    # Heatmap
    im = axes[0, j].imshow(K_s, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    axes[0, j].set_title(f"sigma = {s}", fontsize=11)
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    # Predictions
    K_s_t = ops.convert_to_tensor(K_s, dtype="float32")
    sys_s = K_s_t + lam * ops.eye(n_t)
    a_s = fairkl.solve_cholesky(sys_s, y_t)
    yh_s = np.array(ops.matmul(K_s_t, a_s)).ravel()
    mse_s = float(np.mean((yh_s - y) ** 2))
    axes[1, j].scatter(y, yh_s, c="C0", **SCATTER_KW)
    axes[1, j].plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)
    axes[1, j].set_title(f"MSE = {mse_s:.3f}", fontsize=10)
    axes[1, j].set_xlabel("True y")
    if j == 0:
        axes[1, j].set_ylabel("Predicted ŷ")
    style_ax(axes[1, j])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Diagnosing Unfairness
#
# Our standard KRR model achieves low MSE, but is the result fair? Since the target $y$ depends on $q$, we expect the optimal predictor to exploit that dependence. Let us quantify how much.
#
# ### Standard KRR exploits the sensitive attribute
#
# The predictions $\hat{y} = K\alpha$ are strongly correlated with $q$. This is not a bug — it is the statistically optimal thing to do when $q$ is predictive of $y$. The problem is that in many applications (hiring, lending, criminal justice) we do not **want** the model to rely on the sensitive attribute even if doing so improves accuracy.

# %%
corr_std = np.corrcoef(y_pred_std, q.ravel())[0, 1]
print(f"Corr(ŷ, q) = {corr_std:.3f}  — predictions are highly dependent on q")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(q.ravel(), y_pred_std, c="C1", **SCATTER_KW)
ax.set_xlabel("Sensitive attribute  q")
ax.set_ylabel("Prediction  ŷ")
ax.set_title(f"Standard KRR:  corr(ŷ, q) = {corr_std:.2f}", fontsize=11)
style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter shows a clear linear trend: as $q$ increases, so does $\hat{y}$. We need a way to measure this dependence that works beyond linear correlation and can serve as a differentiable penalty.
#
# ### Measuring dependence: HSIC
#
# The **Hilbert-Schmidt Independence Criterion** (HSIC) measures statistical dependence between two variables by comparing their centered kernel matrices. Intuitively, if knowing $f$ tells us something about $q$, then the structure in $\tilde{K}_f$ will align with the structure in $\tilde{K}_q$. HSIC quantifies this alignment as a Frobenius inner product:
#
# $$\text{HSIC}(K_f, K_q) = \frac{1}{n^2}\operatorname{tr}(\tilde{K}_f\,\tilde{K}_q)$$
#
# HSIC is zero if and only if $f$ and $q$ are independent (in the population limit with characteristic kernels). It is always non-negative and increases with the strength of the dependence.
#
# ### Measuring dependence: CKA
#
# **Centered Kernel Alignment** (CKA) normalizes HSIC to the unit interval:
#
# $$\text{CKA}(K_f, K_q) = \frac{\text{HSIC}(K_f, K_q)}{\sqrt{\text{HSIC}(K_f, K_f)\,\text{HSIC}(K_q, K_q) + \epsilon}}$$
#
# where $\epsilon = 10^{-6}$ is a floor that prevents division by zero when one of the self-HSIC terms is tiny (e.g. a constant predictor).

# %%
# Build kernel matrices on predictions and sensitive attribute
K_yhat = np.array(fairkl.rbf_kernel(y_pred_std.reshape(-1, 1), sigma=1.0))
K_q = np.array(fairkl.rbf_kernel(q, sigma=1.0))

hsic_val = float(fairkl.hsic_biased(K_yhat, K_q))
cka_val = float(fairkl.cka_biased(K_yhat, K_q))
print(f"HSIC = {hsic_val:.6f}")
print(f"CKA  = {cka_val:.4f}   (0 = independent, 1 = fully dependent)")

# %% [markdown]
# ### Why CKA? Boundedness and interpretability
#
# Why prefer CKA over raw HSIC as a penalty term? Two reasons:
#
# **Boundedness.** CKA is always in $[0, 1]$, so the penalty weight $\mu$ has a consistent meaning regardless of data scale or sample size. A practitioner can say "$\mu = 10$ means the fairness penalty is worth ten times the MSE" without worrying about HSIC's magnitude. Raw HSIC, by contrast, scales with $n$ and $\sigma$, making $\mu$ hard to interpret across datasets.
#
# **Scale invariance (linear kernels).** With linear kernels, scaling the predictions by a constant $c$ gives $\text{HSIC}(cf, q) = c^2 \cdot \text{HSIC}(f, q)$, so the penalty's magnitude changes even though the dependence structure is identical. CKA cancels the $c^2$ in numerator and denominator: $\text{CKA}(cf, q) = \text{CKA}(f, q)$. With RBF kernels and a fixed bandwidth, both HSIC and CKA change when data is scaled (the kernel matrix itself changes), so the advantage is specifically about normalization and interpretability.
#
# The experiment below demonstrates this with linear kernels, where the invariance is exact.

# %%
# Demonstrate with linear kernels where the difference is exact
scales = [0.1, 0.5, 1.0, 5.0, 10.0]
hsic_lin_vals, cka_lin_vals = [], []

for s in scales:
    y_scaled = (y_pred_std * s).reshape(-1, 1).astype("float32")
    hsic_lin_vals.append(float(fairkl.hsic_linear(y_scaled, q)))
    cka_lin_vals.append(float(fairkl.cka_linear(y_scaled, q)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(scales, hsic_lin_vals, "o-", color="C0", lw=2, markersize=7)
axes[0].set_xlabel("Prediction scale factor")
axes[0].set_ylabel("HSIC (linear)")
axes[0].set_title("Linear HSIC scales with c**2", fontsize=11)
style_ax(axes[0])

axes[1].plot(scales, cka_lin_vals, "s-", color="C1", lw=2, markersize=7)
axes[1].set_xlabel("Prediction scale factor")
axes[1].set_ylabel("CKA (linear)")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Linear CKA is exactly scale-invariant", fontsize=11)
style_ax(axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# The left panel shows linear HSIC growing quadratically with the scale factor $c$, confirming $\text{HSIC}(cf, q) = c^2 \cdot \text{HSIC}(f, q)$. The right panel shows linear CKA is flat at the same value for every scale — exactly the normalization property we want in a penalty term.

# %% [markdown]
# ## The Fair KRR Objective
#
# We are now ready to assemble the fair objective. The idea is simple: add a CKA penalty to the standard KRR loss so that the optimizer must balance accuracy against fairness.
#
# ### The three-term loss
#
# $$\min_\alpha \;\underbrace{\|K\alpha - y\|^2}_{\text{data fit (MSE)}} + \;\lambda\underbrace{\alpha^\top K\alpha}_{\text{ridge penalty}} + \;\mu\underbrace{\text{CKA}(K\alpha,\; q)}_{\text{fairness penalty}}$$
#
# - When $\mu = 0$ we recover standard KRR with a closed-form Cholesky solution.
# - When $\mu > 0$ the CKA term is nonlinear in $\alpha$ (it involves centering and normalization of the kernel on predictions), so we need iterative optimization. The `FairKernelRidge` model uses Adam.
#
# ### Loss decomposition
#
# To build intuition, let us evaluate each term for the **unfair** ($\mu = 0$) solution and see how the total loss changes as we increase $\mu$.

# %%
alpha_np = np.array(alpha)
K_np = np.array(K_t)

pred = K_np @ alpha_np
mse_term = float(np.mean((pred - y.reshape(-1, 1)) ** 2))
ridge_term = float(lam * np.sum(alpha_np * (K_np @ alpha_np)))
cka_term = float(fairkl.cka_rbf(pred.astype("float32"), q, sigma_q=1.0))

print("Loss decomposition for the unfair (μ=0) solution:")
print(f"  MSE   = {mse_term:.4f}")
print(f"  Ridge = {ridge_term:.4f}")
print(f"  CKA   = {cka_term:.4f}")
print()

mus_demo = [0, 1, 5, 10, 20]
print(f"{'μ':>4s}  {'MSE':>7s}  {'λ·ridge':>8s}  {'μ·CKA':>7s}  {'total':>8s}")
print("-" * 42)
for mu in mus_demo:
    total = mse_term + ridge_term + mu * cka_term
    print(
        f"{mu:4d}  {mse_term:7.4f}  {ridge_term:8.4f}"
        f"  {mu * cka_term:7.4f}  {total:8.4f}"
    )

# %% [markdown]
# The table shows that the CKA penalty dominates the total loss for large $\mu$. At $\mu = 20$ the fairness term is an order of magnitude larger than the MSE term, which means the optimizer will sacrifice accuracy to reduce dependence on $q$.
#
# ### What `model.fit()` does internally
#
# When $\mu > 0$, `FairKernelRidge.fit()` uses a two-phase strategy:
#
# 1. **Warm start.** Compute the exact KRR solution $\alpha_0 = (K + \lambda I)^{-1} y$ via Cholesky. This gives a good starting point that already fits the data well.
# 2. **Adam optimization.** Starting from $\alpha_0$, run gradient descent on the full three-term loss. The warm start means Adam only needs to "nudge" the solution toward fairness rather than learn the data fit from scratch.
#
# This engineering trick typically halves the number of epochs needed for convergence compared to random initialization.
#
# Let us verify that with $\mu = 0$ the model's solution matches our manual Cholesky solve.

# %%
model_std = fairkl.FairKernelRidge(sigma=sigma, lam=lam, mu=0.0)
model_std.fit(X, y)
alpha_model = np.array(model_std.get_alpha())

diff = float(np.max(np.abs(alpha_np - alpha_model)))
print(f"Max |alpha_manual - alpha_model| = {diff:.2e}  (should be ~0)")

# %% [markdown]
# The difference is at machine precision, confirming that `FairKernelRidge` with $\mu = 0$ reduces exactly to standard KRR.

# %% [markdown]
# ## Results: The Fairness-Accuracy Trade-off
#
# We now sweep $\mu \in \{0, 1, 5, 10, 20\}$ and record both MSE and CKA for each model. Each value of $\mu$ represents a different point on the trade-off between accuracy and fairness.
#
# ### Pareto frontier

# %%
mus = [0, 1, 5, 10, 20]
mse_list, cka_list = [], []

for mu in mus:
    model = fairkl.FairKernelRidge(sigma=1.0, lam=0.01, mu=mu, sigma_q=1.0)
    model.fit(X, y, q=q, epochs=200, lr=0.005)
    yh = np.array(model.predict(X)).ravel()
    mse = float(np.mean((yh - y) ** 2))
    cka_val = float(fairkl.cka_rbf(yh.reshape(-1, 1), q))
    mse_list.append(mse)
    cka_list.append(cka_val)
    print(f"μ = {mu:5.1f}   MSE = {mse:.3f}   CKA = {cka_val:.3f}")

# %% [markdown]
# The printout shows the expected pattern: as $\mu$ increases, CKA drops (fairer predictions) but MSE rises (less accurate predictions). Let us visualize this as a Pareto frontier.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cka_list, mse_list, "o-", color="C0", lw=2, markersize=8)
for i, mu in enumerate(mus):
    ax.annotate(
        f"μ={mu}",
        (cka_list[i], mse_list[i]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
    )
ax.set_xlabel("CKA   (0 = fair,  1 = unfair)")
ax.set_ylabel("MSE")
ax.set_title("Fairness vs Accuracy — Pareto Frontier", fontsize=12)
style_ax(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# Each point on the curve is a trained model. Moving left along the x-axis means fairer predictions (lower CKA); moving down means more accurate predictions (lower MSE). No model can be both perfectly fair and perfectly accurate on this dataset — the curve quantifies exactly how much accuracy we must sacrifice for a given level of fairness.
#
# **Key takeaway.** The penalty weight $\mu$ is the practitioner's dial. Setting $\mu = 0$ recovers standard KRR with maximum accuracy but no fairness guarantee. Increasing $\mu$ trades accuracy for fairness. The Pareto frontier makes this trade-off explicit and helps practitioners choose an operating point that satisfies their application's constraints.
#
# ---
#
# **Next**: [Part 2](tutorial_fair_krr_part2.py) wraps the model with scikit-learn for cross-validated hyperparameter selection.
