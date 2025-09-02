"""Test package."""

import importlib.metadata

import xmmutablemap as m


def test_version():
    assert importlib.metadata.version("xmmutablemap") == m.__version__
