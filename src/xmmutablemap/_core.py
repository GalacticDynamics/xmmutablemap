"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

xmmutablemap: Immutable Map, compatible with JAX & Equinox
"""

__all__ = ("ImmutableMap", "frozendict")

import sys
from collections.abc import Mapping
from typing import Any, TypeVar

from jax.tree_util import register_pytree_node_class

if sys.version_info >= (3, 15):
    # PEP 814 built-in. Not exercised by CI, which runs 3.10 and 3.13; the
    # backport below is what those cover.
    from builtins import frozendict  # pragma: no cover
else:
    from ._frozendict import frozendict

K = TypeVar("K")
V = TypeVar("V")


@register_pytree_node_class
class ImmutableMap(frozendict[K, V]):
    """Immutable mapping that JAX understands as a PyTree.

    A `frozendict` (`PEP 814 <https://peps.python.org/pep-0814/>`_ on Python
    3.15+, a matching stand-in below that) registered as a JAX PyTree node.
    Everything a `frozendict` does, it does; the difference is that JAX
    traverses it as a container rather than treating it as an opaque leaf.

    Parameters
    ----------
    *args : Mapping[K, V] | Iterable[tuple[K, V]]
        At most one positional argument, as for `dict`.
    **kwargs : V
        Key-value pairs.

    Examples
    --------
    >>> from xmmutablemap import ImmutableMap
    >>> d = ImmutableMap(a=1, b=2)
    >>> d
    ImmutableMap({'a': 1, 'b': 2})

    It is a `frozendict`, and compares equal to any equal mapping:

    >>> from xmmutablemap import frozendict
    >>> isinstance(d, frozendict)
    True
    >>> d == {"a": 1, "b": 2}
    True

    Being immutable, it is hashable, order-independently:

    >>> hash(ImmutableMap(a=1, b=2)) == hash(ImmutableMap(b=2, a=1))
    True

    Unlike a plain `frozendict`, JAX sees the structure:

    >>> import jax
    >>> jax.tree.structure(d)
    PyTreeDef(CustomNode(ImmutableMap[('a', 'b')], [*, *]))

    """

    __slots__ = ()

    def __or__(self, other: Any, /) -> "ImmutableMap[K, V]":
        """Merge with another mapping, keeping this type.

        `frozendict.__or__` hardcodes its result type, so a subclass would
        otherwise get a plain `frozendict` back.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d | {"b": 3, "c": 4}
        ImmutableMap({'a': 1, 'b': 3, 'c': 4})

        >>> try:
        ...     d | ()
        ... except TypeError:
        ...     print("Cannot combine with non-mapping")
        Cannot combine with non-mapping

        """
        if not isinstance(other, Mapping):
            return NotImplemented
        return type(self)(dict(self) | dict(other))

    def copy(self) -> "ImmutableMap[K, V]":
        """Return a shallow copy, which for an immutable map is itself.

        `frozendict.copy` returns a plain `frozendict` for a subclass; return
        `self` instead, as the built-in does for an exact `frozendict`.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.copy() is d
        True

        """
        return self

    # ===========================================
    # JAX PyTree

    def tree_flatten(self) -> tuple[tuple[V, ...], tuple[K, ...]]:
        """Flatten dict to the values (and keys).

        This is used for JAX's tree flattening.

        Returns
        -------
        tuple[tuple[V, ...], tuple[K, ...]]
            A tuple of (values, keys).
            The keys are treated as auxiliary data.

        Examples
        --------
        >>> import jax
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.tree_flatten()
        ((1, 2), ('a', 'b'))

        >>> jax.tree.flatten(d)
        ([1, 2], PyTreeDef(CustomNode(ImmutableMap[('a', 'b')], [*, *])))

        """
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[K, ...], children: tuple[V, ...]
    ) -> "ImmutableMap[K, V]":
        """Unflatten into an ImmutableMap from the keys and values.

        This is used for JAX's tree un-flattening.

        Parameters
        ----------
        aux_data : tuple[K, ...]
            The keys.
        children : tuple[V, ...]
            The values.

        Examples
        --------
        >>> import jax
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> flat = d.tree_flatten()
        >>> ImmutableMap.tree_unflatten(*flat)
        ImmutableMap({1: 'a', 2: 'b'})

        >>> jax.tree.unflatten(jax.tree.structure(d), flat)
        ImmutableMap({'a': (1, 2), 'b': ('a', 'b')})

        """
        return cls(tuple(zip(aux_data, children, strict=True)))
