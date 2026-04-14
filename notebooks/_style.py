"""Shared plot styling for fairkl documentation notebooks."""

from __future__ import annotations


def style_ax(ax):
    """Apply gaussx-style grid and ticks to an axis."""
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()


SCATTER_KW = dict(s=30, edgecolors="k", linewidths=0.5, alpha=0.5, zorder=5)
GROUP_COLORS = {"A": "C1", "B": "C0"}  # orange / blue
