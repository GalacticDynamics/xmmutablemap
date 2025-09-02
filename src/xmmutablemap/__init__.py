"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

xmmutablemap: Immutable Map, compatible with JAX & Equinox
"""

__all__ = ["ImmutableMap", "__version__"]

from ._core import ImmutableMap

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
