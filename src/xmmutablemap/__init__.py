"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

xmmutablemap: Immutable Map, compatible with JAX & Equinox
"""

__all__ = ["ImmutableMap", "__version__", "frozendict"]

from ._core import ImmutableMap, frozendict

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"
