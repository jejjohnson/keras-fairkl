---
applyTo: "docs/**/*.py,docs/**/*.md,notebooks/**/*.py"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **jupytext percent-format `.py` files**. They are the single source of truth for all code. Pre-executed `.ipynb` files (with inline figure outputs) are committed alongside them and rendered by mkdocs.

**Execution model**: `mkdocs-jupyter` renders notebooks with `execute: false`. Authors run notebooks locally via `jupytext --to notebook --execute`, producing `.ipynb` files with inline outputs. Both `.py` and `.ipynb` are committed.

## Directory Layout

```
docs/
├── notebooks/
│   ├── _style.py              # shared plot styling
│   ├── demo_foo.py            # jupytext percent-format (source)
│   ├── demo_foo.ipynb         # pre-executed (committed)
│   └── benchmark_bar.py
│   └── benchmark_bar.ipynb
└── guide.md
```

## Workflow

1. Write the `.py` source (jupytext percent format)
2. Execute locally: `jupytext --to notebook --execute foo.py -o foo.ipynb`
3. Commit both the `.py` source and the executed `.ipynb`
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()`. Do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

## Jupytext Header

Every notebook `.py` file must start with this header:

```python
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
```

## Cell Markers

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import keras.ops as ops
```

## Notebook Structure

Every example notebook should follow this order:

1. **Title & overview** (markdown) — what the notebook demonstrates, prerequisites
2. **Imports** (code) — set `KERAS_BACKEND`, import libraries
3. **Problem setup** (markdown + code) — data, parameters
4. **Core computation** (markdown + code) — the actual demonstration
5. **Figures & tables** (code) — generate and display with `plt.show()`
6. **Summary / takeaways** (markdown)

## Figures

Use `plt.show()` to render figures inline. The executed `.ipynb` captures the output as base64-encoded images.

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, "o-", color="C0", lw=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
style_ax(ax)
plt.tight_layout()
plt.show()
```

Do **not** use `savefig` or `IMG_DIR`. Do **not** add markdown cells with `![image](...)` embeds.

## Plot Styling

Import shared styling from `_style.py`:

```python
from _style import SCATTER_KW, style_ax
```

- `style_ax(ax)` — applies grid and minor ticks
- `SCATTER_KW` — consistent scatter plot kwargs (s=30, black edges, alpha=0.5)

## Tables & Statistics

For comparison tables, print them in a code cell:

```python
# %%
for name, stats in results.items():
    print(f"{name:20s}  time={stats['time_ms']:8.2f} ms")
```

The code cell output in the `.ipynb` is the rendered table.

## mkdocs.yml Nav

Point nav entries to `.ipynb` files (not `.py`):

```yaml
nav:
  - Examples:
      - Demo: notebooks/demo_foo.ipynb
```

## Checklist for New Notebooks

- [ ] Jupytext header present
- [ ] `KERAS_BACKEND` set before imports
- [ ] Every figure uses `plt.show()` (no `savefig`)
- [ ] No `IMG_DIR` or `Path(__file__)` setup
- [ ] No markdown image embed cells (`![](...)`)
- [ ] Notebook listed in `mkdocs.yml` nav (pointing to `.ipynb`)
- [ ] Both `.py` and executed `.ipynb` committed
