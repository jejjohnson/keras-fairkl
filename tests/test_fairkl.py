"""Smoke test for fairkl package import."""

from __future__ import annotations

import fairkl


def test_import():
    assert fairkl is not None
