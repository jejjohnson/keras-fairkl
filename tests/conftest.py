"""Pytest configuration: set Keras backend to JAX."""

from __future__ import annotations

import os


os.environ.setdefault("KERAS_BACKEND", "jax")
