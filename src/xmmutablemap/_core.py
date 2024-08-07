"""
Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

xmmutablemap: Immutable Map, compatible with JAX & Equinox
"""

__all__: list[str] = []

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Annotated, Any, TypeVar, overload
from typing_extensions import Doc

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
    >>> from galax.utils import ImmutableMap
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
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ===========================================
    # Mapping Protocol

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def keys(self) -> KeysView[K]:
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        return self._data.items()

    @overload
    def get(self, key: K, /) -> V | None: ...

    @overload
    def get(self, key: K, /, default: V | _T) -> V | _T: ...

    def get(self, key: K, /, default: V | _T | None = None) -> V | _T | None:
        return self._data.get(key, default)

    # ===========================================
    # Extending Mapping

    def __or__(self, value: Any, /) -> "ImmutableMap[K, V]":
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
        """
        return hash(tuple(self._data.items()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    # ===========================================
    # JAX PyTree

    def tree_flatten(
        self,
    ) -> tuple[
        Annotated[tuple[V, ...], Doc("The values.")],
        Annotated[tuple[K, ...], Doc("The keys as auxiliary data.")],
    ]:
        """Flatten dict to the values (and keys)."""
        return tuple(self._data.values()), tuple(self._data.keys())

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Annotated[tuple[K, ...], Doc("The keys.")],
        children: Annotated[tuple[V, ...], Doc("The values.")],
    ) -> "ImmutableMap[K, V]":
        """Unflatten into an ImmutableMap from the keys and values."""
        return cls(tuple(zip(aux_data, children, strict=True)))
