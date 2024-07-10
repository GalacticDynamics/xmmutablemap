from __future__ import annotations

import importlib.metadata

import immutable_map_jax as m


def test_version():
    assert importlib.metadata.version("immutable_map_jax") == m.__version__
