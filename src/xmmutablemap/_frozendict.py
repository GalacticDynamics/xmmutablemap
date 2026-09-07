"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

Backport of the :class:`frozendict` built-in added in Python 3.15 (`PEP 814
<https://peps.python.org/pep-0814/>`_).

On Python 3.15+ :mod:`xmmutablemap` re-exports the built-in and this module is
unused. On older Pythons this pure-Python stand-in takes its place, so that
:class:`~xmmutablemap.ImmutableMap` can subclass ``frozendict`` on every
supported version.

Behaviour is matched to the built-in wherever it is observable, including the
order-independent hash, equality with :class:`dict`, the ``frozendict({...})``
repr (and bare ``frozendict()`` when empty), and ``|`` returning a plain
``frozendict`` rather than the subclass.
"""

__all__ = ("frozendict",)

from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import Any, TypeVar, cast, overload

_T = TypeVar("_T")
K = TypeVar("K")
V = TypeVar("V")


# The lowercase name is deliberate: this stands in for the built-in, and
# `type(self).__name__` is what the repr prints.
class frozendict(Mapping[K, V]):  # noqa: N801  # pylint: disable=invalid-name
    """Immutable, hashable mapping.

    Parameters
    ----------
    *args : Mapping[K, V] | Iterable[tuple[K, V]]
        At most one positional argument, as for `dict`.
    **kwargs : V
        Key-value pairs.

    Examples
    --------
    >>> from xmmutablemap import frozendict
    >>> frozendict(a=1, b=2)
    frozendict({'a': 1, 'b': 2})

    >>> frozendict()
    frozendict()

    """

    # `__slots__` alone gives instances no `__dict__`, so attributes can be
    # neither set nor deleted -- with the same message the built-in raises.
    # No `__setattr__` override is needed to reproduce that.
    __slots__ = ("_data",)

    _data: dict[K, V]

    def __init__(
        self,
        /,
        *args: Mapping[K, V] | Iterable[tuple[K, V]],
        **kwargs: V,
    ) -> None:
        self._data = dict(*args, **kwargs)  # type: ignore[assignment]

    # ===========================================
    # Mapping protocol

    def __getitem__(self, key: K) -> V:
        """Get an item by key.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1)["a"]
        1

        """
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        """Iterate over the keys, in insertion order.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> list(frozendict(a=1, b=2))
        ['a', 'b']

        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> len(frozendict(a=1, b=2))
        2

        """
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        """Check whether a key is present.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> "a" in frozendict(a=1)
        True

        """
        return key in self._data

    def __reversed__(self) -> Iterator[K]:
        """Iterate over the keys in reverse insertion order.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> list(reversed(frozendict(a=1, b=2)))
        ['b', 'a']

        """
        return reversed(self._data)

    def keys(self) -> KeysView[K]:
        """Return a view of the keys.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1, b=2).keys()
        dict_keys(['a', 'b'])

        """
        return self._data.keys()

    def values(self) -> ValuesView[V]:
        """Return a view of the values.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1, b=2).values()
        dict_values([1, 2])

        """
        return self._data.values()

    def items(self) -> ItemsView[K, V]:
        """Return a view of the items.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1, b=2).items()
        dict_items([('a', 1), ('b', 2)])

        """
        return self._data.items()

    @overload
    def get(self, key: K, /) -> V | None: ...

    @overload
    def get(self, key: K, /, default: V | _T) -> V | _T: ...

    def get(self, key: K, /, default: V | _T | None = None) -> V | _T | None:
        """Get an item by key, returning `default` if absent.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> d = frozendict(a=1)
        >>> d.get("a")
        1
        >>> d.get("b")
        >>> d.get("b", 2)
        2

        """
        return self._data.get(key, default)

    def copy(self) -> "frozendict[K, V]":
        """Return a shallow copy.

        As the built-in does, an exact `frozendict` returns itself; a subclass
        returns a plain `frozendict`.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> d = frozendict(a=1)
        >>> d.copy() is d
        True

        """
        # An exact-type check, not `isinstance`: the built-in uses
        # `PyFrozenDict_CheckExact` here, so a *subclass* must not return
        # `self`. `ImmutableMap` overrides `copy` precisely because of this.
        # pylint: disable-next=unidiomatic-typecheck
        return self if type(self) is frozendict else frozendict(self._data)

    @classmethod
    def fromkeys(
        cls, iterable: Iterable[K], value: Any = None, /
    ) -> "frozendict[K, Any]":
        """Build a `frozendict` with keys from `iterable`, all set to `value`.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict.fromkeys(["a", "b"], 0)
        frozendict({'a': 0, 'b': 0})

        """
        return cls(dict.fromkeys(iterable, value))

    # ===========================================
    # Operators

    def __or__(self, other: Any, /) -> "frozendict[Any, Any]":
        """Merge with another mapping, returning a new `frozendict`.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1) | {"b": 2}
        frozendict({'a': 1, 'b': 2})

        """
        if not isinstance(other, Mapping):
            return NotImplemented
        # The built-in's C implementation hardcodes the result type, so a
        # subclass of `frozendict` gets a plain `frozendict` back. Match that.
        return frozendict(self._data | dict(other))

    def __ror__(self, other: Any, /) -> Any:
        """Merge into another mapping.

        `dict.__or__` accepts only a `dict` (and, on 3.15+, a `frozendict`).
        This stand-in is neither, so ``{...} | frozendict(...)`` gets
        ``NotImplemented`` from the left operand and lands here.

        The left operand drives the merge, which reproduces the built-in's
        results -- including keeping the left-hand type.

        A `Mapping` that implements no ``|`` of its own raises `TypeError`
        here, exactly as it does for the built-in: ``__ror__`` is not a
        promise to merge into any mapping whatsoever. The one difference is
        cosmetic -- the message names ``dict``, this class's backing store,
        where the built-in names ``frozendict``.

        Examples
        --------
        >>> from collections import OrderedDict
        >>> from xmmutablemap import frozendict
        >>> {"b": 2} | frozendict(a=1)
        {'b': 2, 'a': 1}
        >>> type(OrderedDict(b=2) | frozendict(a=1)).__name__
        'OrderedDict'

        """
        if not isinstance(other, Mapping):
            return NotImplemented
        # `other` is some Mapping; let it drive the merge so that e.g.
        # `OrderedDict | frozendict` stays an OrderedDict.
        return cast("Any", other) | self._data

    # ===========================================
    # Other

    def __eq__(self, other: object) -> bool:
        """Compare equal to any mapping with the same items.

        The built-in's C comparison returns ``NotImplemented`` for anything
        that is not a `dict` or `frozendict`, and leans on the other operand's
        reflected comparison for the rest -- which is how ``frozendict(a=1) ==
        MappingProxyType({"a": 1})`` comes out `True`. A `mappingproxy` knows
        nothing about *this* class, so the reflected call would not come back;
        handle any `Mapping` here directly. Non-mappings still get
        ``NotImplemented``, as the built-in gives them.

        Examples
        --------
        >>> from types import MappingProxyType
        >>> from xmmutablemap import frozendict
        >>> d = frozendict(a=1, b=2)
        >>> d == {"a": 1, "b": 2} == MappingProxyType({"a": 1, "b": 2})
        True
        >>> d == {"a": 1}
        False
        >>> d.__eq__(1) is NotImplemented
        True

        """
        if isinstance(other, frozendict):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash, order-independently, as the built-in does.

        The hash *value* differs from the built-in's -- hashes are not stable
        across interpreters or releases -- but the semantics match: equal
        mappings hash equally whatever their insertion order.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> hash(frozendict(a=1, b=2)) == hash(frozendict(b=2, a=1))
        True

        """
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        """Return the representation.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1, b=2)
        frozendict({'a': 1, 'b': 2})
        >>> frozendict()
        frozendict()

        """
        name = type(self).__name__
        return f"{name}({self._data!r})" if self._data else f"{name}()"
