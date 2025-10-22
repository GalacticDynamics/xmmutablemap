"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

xmmutablemap: Immutable Map, compatible with JAX & Equinox
"""

__all__ = ("ImmutableMap",)

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any, TypeVar, overload

from jax.tree_util import register_pytree_node_class

_T = TypeVar("_T")
K = TypeVar("K")
V = TypeVar("V")


@register_pytree_node_class
class ImmutableMap(Mapping[K, V]):
    """Immutable string-keyed dictionary.

    Parameters
    ----------
    *args : tuple[str, V]
        Key-value pairs.
    **kwargs : V
        Key-value pairs.

    Examples
    --------
    >>> from xmmutablemap import ImmutableMap
    >>> d = ImmutableMap(a=1, b=2)
    >>> d
    ImmutableMap({'a': 1, 'b': 2})

    """

    def __init__(
        self,
        /,
        *args: Mapping[K, V] | tuple[K, V] | Iterable[tuple[K, V]],
        **kwargs: V,
    ) -> None:
        self._data: dict[K, V] = dict(*args, **kwargs)  # type: ignore[assignment]

    # ===========================================
    # Collection Protocol

    def __contains__(self, key: Any) -> bool:
        """Check if the key is in the map.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> "a" in d
        True
        >>> "c" in d
        False

        """
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        """Return an iterator over the keys.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> [k for k in d]
        ['a', 'b']
        >>> list(d)
        ['a', 'b']

        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items in the map.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> len(d)
        2

        """
        return len(self._data)

    # ===========================================
    # Mapping Protocol

    def __getitem__(self, key: K) -> V:
        """Get an item by key.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d["a"]
        1
        >>> d["c"]
        Traceback (most recent call last):
          ...
        KeyError: 'c'

        """
        return self._data[key]

    def keys(self) -> KeysView[K]:
        """Return the keys.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.keys()
        dict_keys(['a', 'b'])

        """
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        """Return the values.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.values()
        dict_values([1, 2])

        """
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        """Return the items.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.items()
        dict_items([('a', 1), ('b', 2)])

        """
        return self._data.items()

    @overload
    def get(self, key: K, /) -> V | None: ...

    @overload
    def get(self, key: K, /, default: V | _T) -> V | _T: ...

    def get(self, key: K, /, default: V | _T | None = None) -> V | _T | None:
        """Get an item by key.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> d.get("a")
        1
        >>> d.get("c")
        >>> d.get("c", 3)
        3

        """
        return self._data.get(key, default)

    # ===========================================
    # Extending Mapping

    def __or__(self, value: Any, /) -> "ImmutableMap[K, V]":
        """Return a new ImmutableMap combining this with another mapping.

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
        if not isinstance(value, Mapping):
            return NotImplemented

        return type(self)(self._data | dict(value))

    def __ror__(self, value: Any) -> Any:
        return value | self._data

    # ===========================================
    # Other

    def __hash__(self) -> int:
        """Hash.

        Normally, dictionaries are not hashable because they are mutable.
        However, this dictionary is immutable, so we can hash it.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> isinstance(hash(d), int)
        True

        """
        return hash(tuple(self._data.items()))

    def __repr__(self) -> str:
        """Return the representation.

        Examples
        --------
        >>> from xmmutablemap import ImmutableMap
        >>> d = ImmutableMap(a=1, b=2)
        >>> repr(d)
        "ImmutableMap({'a': 1, 'b': 2})"

        """
        return f"{self.__class__.__name__}({self._data!r})"

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
        return tuple(self._data.values()), tuple(self._data.keys())

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
