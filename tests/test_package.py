from __future__ import annotations

import importlib.metadata

import xmmutablemap as m


def test_version():
    assert importlib.metadata.version("xmmutablemap") == m.__version__
