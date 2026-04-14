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
# # FairModelWrapper — bring your own neural network
#
# `FairModelWrapper` takes **any** `keras.Model` (MLP, CNN, transformer) and adds a CKA fairness penalty via the standard `compile` / `fit` API. Unlike `FairLinear` and `FairKernelRidge`, which hard-wire a linear or kernel-ridge predictor, the wrapper leaves the architecture choice entirely to the user — you only lose the closed-form tricks those specialised models use; you gain arbitrary depth, nonlinearity, and compatibility with any Keras optimizer, callback, or dataset pipeline.
#
# This notebook trains a small MLP on a synthetic tabular task where the target is partially driven by a sensitive attribute, and shows how increasing the fairness weight $\mu$ progressively decouples the predictions from that attribute.

# %%
from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import matplotlib.pyplot as plt
import numpy as np

from _style import SCATTER_KW, style_ax
from fairkl.metrics.cka import CKALoss, cka_rbf
from fairkl.models import FairModelWrapper


# %% [markdown]
# ## Synthetic data
#
# We build a nonlinear regression task where `y = tanh(x) + 3 * q + noise`. The sensitive attribute `q` has a strong linear effect on the target; without a fairness penalty any sufficiently expressive model will pick it up. We stack `q` into `X` so the network *can* see it — this is the hard case where we rely on the CKA term to suppress the dependence.

# %%
rng = np.random.default_rng(0)
n = 800
x_feat = rng.standard_normal((n, 3)).astype("float32")
q = rng.standard_normal((n, 1)).astype("float32")
X = np.hstack([x_feat, q]).astype("float32")
y = (
    np.tanh(x_feat[:, 0])
    + 0.5 * x_feat[:, 1]
    + 3.0 * q.ravel()
    + 0.2 * rng.standard_normal(n)
).astype("float32")[:, None]

Xtr, Xte = X[:600], X[600:]
ytr, yte = y[:600], y[600:]
qtr, qte = q[:600], q[600:]

print(f"Corr(y, q)      = {np.corrcoef(y.ravel(), q.ravel())[0, 1]:+.3f}")
print(f"Train / test    = {Xtr.shape[0]} / {Xte.shape[0]}")


# %% [markdown]
# ## Build a plain Keras MLP
#
# Any Keras model works. Below is a stock two-hidden-layer MLP. We wrap it with `FairModelWrapper(mu=...)` and call the familiar `compile` / `fit` pair — the fairness term is added internally via `add_loss`.


# %%
def build_mlp(d: int = 4) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(d,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


# %% [markdown]
# ## Sweep $\mu$
#
# Train the same architecture at a range of fairness weights. `mu=0` is the unfair baseline (pure MSE); larger `mu` trades task performance for independence between predictions and `q`. The fairness loss is supplied explicitly via `fairness_loss=CKALoss(...)` — any `keras.losses.Loss` taking `(y_true=q, y_pred=f)` works here, so swapping in `HSICLoss` or `MMDLoss` is a one-line change.

# %%
mus = [0.0, 0.5, 2.0, 8.0]
results = {}
fairness_loss = CKALoss(sigma_f=1.0, sigma_q=1.0)

for mu in mus:
    keras.utils.set_random_seed(0)
    mlp = build_mlp(d=X.shape[1])
    model = FairModelWrapper(mlp, mu=mu, fairness_loss=fairness_loss)
    model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse", metrics=["mse"])
    history = model.fit(Xtr, ytr, q=qtr, epochs=40, batch_size=128, verbose=0)

    # Keras tracks the MSE metric independently of the total loss; the
    # fairness contribution is therefore (total loss - MSE) at each epoch.
    total = np.asarray(history.history["loss"])
    task = np.asarray(history.history["mse"])
    fair = np.maximum(total - task, 0.0)

    yh = np.asarray(model.predict(Xte, verbose=0))
    mse = float(np.mean((yh - yte) ** 2))
    cka = float(cka_rbf(yh, qte, sigma_f=1.0, sigma_q=1.0))
    results[mu] = {
        "yh": yh.ravel(),
        "mse": mse,
        "cka": cka,
        "total": total,
        "task": task,
        "fair": fair,
    }
    print(f"mu = {mu:5.2f}  |  test MSE = {mse:.3f}  |  CKA(yhat, q) = {cka:.3f}")


# %% [markdown]
# ## Training loss curves
#
# Each column below is one `mu`. The blue line is the **total** objective that the optimizer sees (what Adam actually minimises). The orange line is the **task** term (MSE of the predictions), and the green line is the **fairness** contribution (`mu * CKA`, recovered as `total - MSE`). At `mu=0` the fairness line is flat at zero and total ≡ task. As `mu` grows the fairness term takes a larger fraction of the objective early on and decays as the network learns predictions that are simultaneously accurate and independent of `q`.

# %%
fig, axes = plt.subplots(1, len(mus), figsize=(3.6 * len(mus), 3.2), sharex=True)
for ax, mu in zip(axes, mus, strict=True):
    r = results[mu]
    ax.plot(r["total"], label="total loss", color="tab:blue", lw=2)
    ax.plot(r["task"], label="task (MSE)", color="tab:orange", lw=1.5, ls="--")
    ax.plot(r["fair"], label="fairness (μ·CKA)", color="tab:green", lw=1.5, ls=":")
    ax.set_title(f"μ = {mu}")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    style_ax(ax)
axes[0].set_ylabel("loss (log scale)")
axes[-1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Trade-off curve
#
# The classic fairness / accuracy trade-off: more regularisation toward independence (higher CKA weight) costs predictive accuracy. The curve lets you pick a $\mu$ that matches your tolerance.

# %%
fig, ax = plt.subplots(figsize=(5.5, 4))
mses = [results[m]["mse"] for m in mus]
ckas = [results[m]["cka"] for m in mus]
ax.plot(ckas, mses, marker="o", color="tab:blue")
for m, c, e in zip(mus, ckas, mses, strict=True):
    ax.annotate(f"μ={m}", (c, e), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("CKA(yhat, q)  —  lower is fairer")
ax.set_ylabel("Test MSE")
ax.set_title("Fairness / accuracy trade-off on held-out data")
style_ax(ax)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Predictions vs. the sensitive attribute
#
# At `mu=0` the predictions track `q` almost linearly (slope ≈ 3, matching the data-generating process). As `mu` grows, the slope flattens — the predictor has learnt to rely on the other features instead of `q`.

# %%
fig, axes = plt.subplots(1, len(mus), figsize=(3.5 * len(mus), 3.2), sharey=True)
for ax, mu in zip(axes, mus, strict=True):
    ax.scatter(qte.ravel(), results[mu]["yh"], color="tab:blue", **SCATTER_KW)
    ax.set_title(f"μ = {mu}")
    ax.set_xlabel("q (sensitive)")
    style_ax(ax)
axes[0].set_ylabel("yhat")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Swapping in HSIC or MMD
#
# The `fairness_loss=` argument accepts any `keras.losses.Loss` that takes `(y_true=q, y_pred=f)`. `CKALoss` is the default because it is bounded in $[0, 1]$ (the penalty weight is interpretable regardless of data scale); `HSICLoss` and `MMDLoss` are also available if you prefer an unnormalised dependence measure.

# %%
from fairkl.metrics.hsic import HSICLoss

keras.utils.set_random_seed(0)
mlp = build_mlp(d=X.shape[1])
hsic_model = FairModelWrapper(
    mlp, mu=0.2, fairness_loss=HSICLoss(sigma_f=1.0, sigma_q=1.0)
)
hsic_model.compile(optimizer=keras.optimizers.Adam(3e-3), loss="mse")
hsic_model.fit(Xtr, ytr, q=qtr, epochs=20, batch_size=128, verbose=0)

yh = np.asarray(hsic_model.predict(Xte, verbose=0))
mse_h = float(np.mean((yh - yte) ** 2))
cka_h = float(cka_rbf(yh, qte, sigma_f=1.0, sigma_q=1.0))
print(f"HSIC-penalised MLP  |  test MSE = {mse_h:.3f}  |  CKA(yhat, q) = {cka_h:.3f}")
