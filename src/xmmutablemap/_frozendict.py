"""Copyright (c) 2024 Galactic Dynamics Maintainers. All rights reserved.

Backport of the :class:`frozendict` built-in added in Python 3.15 (`PEP 814
<https://peps.python.org/pep-0814/>`_).

On Python 3.15+ :mod:`xmmutablemap` re-exports the built-in and this module is
unused. On older Pythons this pure-Python stand-in takes its place, so that
:class:`~xmmutablemap.ImmutableMap` can subclass ``frozendict`` on every
supported version.

Behaviour is matched to the built-in (checked against CPython 3.15.0),
including construction in ``__new__``, ``frozendict(d) is d`` for an exact
``frozendict``, the hash (that of the items' ``frozenset``), equality with
:class:`dict`, the ``frozendict({...})`` repr (and bare ``frozendict()`` when
empty), ``|`` merging only a ``dict`` or ``frozendict`` and returning a plain
``frozendict`` rather than the subclass, ``copy.copy``, and pickling.

Known differences, all either inherent to a pure-Python class or cosmetic:

- It subclasses :class:`collections.abc.Mapping` (so static type checkers see
  a ``Mapping``); the built-in is only registered as one. This shows in the
  MRO, ``dir()``, and the ``ABCMeta`` metaclass.
- ``.keys().mapping`` (and the other views') wraps the backing ``dict``, not
  the ``frozendict``.
- Some error messages name ``dict`` (e.g. for unhashable keys) where the
  built-in names ``frozendict``; the exception types match.
- ``__module__``, ``inspect.signature``, and the like differ.
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
from copyreg import __newobj__  # type: ignore[attr-defined]
from typing import Any, SupportsIndex, TypeVar, cast, overload
from typing_extensions import Self

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
    # ponytail: `_data` itself stays rebindable, being a slot; it is private.
    __slots__ = ("_data",)

    _data: dict[K, V]

    # Built in `__new__`, not `__init__`, as the built-in is: calling
    # `__init__` again must not change the contents, and a subclass's own
    # `__init__` sees an already-filled mapping.
    def __new__(
        cls,
        /,
        *args: Mapping[K, V] | Iterable[tuple[K, V]],
        **kwargs: V,
    ) -> "Self":
        if len(args) > 1:
            msg = f"frozendict expected at most 1 argument, got {len(args)}"
            raise TypeError(msg)
        # Like `tuple(t)`, an exact frozendict of an exact frozendict is itself.
        # pylint: disable-next=unidiomatic-typecheck
        if cls is frozendict and args and not kwargs and type(args[0]) is frozendict:
            return cast("Self", args[0])
        self = super().__new__(cls)
        self._data = dict(*args, **kwargs)  # type: ignore[assignment]
        return self

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

    def copy(self, /) -> "frozendict[K, V]":
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

    def __copy__(self) -> "frozendict[K, V]":
        """Support `copy.copy`.

        The `copy` module treats an exact built-in `frozendict` as atomic and
        copies a subclass through its pickle protocol. Match both.

        Examples
        --------
        >>> import copy
        >>> from xmmutablemap import frozendict
        >>> d = frozendict(a=1)
        >>> copy.copy(d) is d
        True

        """
        # pylint: disable-next=unidiomatic-typecheck
        if type(self) is frozendict:
            return self
        func, args, state, *_ = self.__reduce_ex__(4)
        new = func(*args)
        if state:
            new.__dict__.update(state)
        return cast("frozendict[K, V]", new)

    def __getnewargs__(self) -> tuple[dict[K, V]]:
        """Return the arguments `__new__` needs to rebuild this.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1).__getnewargs__()
        ({'a': 1},)

        """
        return (dict(self._data),)

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> tuple[Any, ...]:
        """Reduce for pickling and `copy`, exactly as the built-in does.

        The data is the `__new__` argument, and only a subclass's instance
        `__dict__` (if any) is state. Spelled out here, rather than left to
        `object`, so pickles rebuild through `__new__`; protocols 0 and 1,
        which cannot, are refused as the built-in refuses them.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> frozendict(a=1).__reduce_ex__(4)[1:]
        ((<class '...frozendict'>, {'a': 1}), None, None, None)

        """
        if protocol < 2:  # type: ignore[operator]
            # The built-in, needing `__newobj__`, refuses protocols 0 and 1,
            # naming `frozendict` even for a subclass.
            msg = "cannot pickle 'frozendict' object"
            raise TypeError(msg)
        # ponytail: extra `__slots__` on a subclass are not pickled; add them
        # to the state if a subclass ever needs them.
        state = getattr(self, "__dict__", None) or None
        return __newobj__, (type(self), *self.__getnewargs__()), state, None, None

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
        # The built-in merges only a `dict` or `frozendict`; any other
        # mapping must bring its own `|` (as `mappingproxy` does).
        if not isinstance(other, (dict, frozendict)):
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

        As with the built-in, only a `dict` (or subclass) merges this way;
        any other `Mapping` gets ``NotImplemented``, so ``|`` raises
        `TypeError` unless that mapping brings its own ``|``.

        Examples
        --------
        >>> from collections import OrderedDict
        >>> from xmmutablemap import frozendict
        >>> {"b": 2} | frozendict(a=1)
        {'b': 2, 'a': 1}
        >>> type(OrderedDict(b=2) | frozendict(a=1)).__name__
        'OrderedDict'

        """
        if not isinstance(other, dict):
            return NotImplemented
        # Let the dict drive the merge so that e.g. `OrderedDict | frozendict`
        # stays an OrderedDict.
        return other | self._data

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

        The value is that of the items' `frozenset`, which is exactly the
        built-in's hash.

        Examples
        --------
        >>> from xmmutablemap import frozendict
        >>> hash(frozendict(a=1, b=2)) == hash(frozendict(b=2, a=1))
        True
        >>> hash(frozendict(a=1)) == hash(frozenset({("a", 1)}))
        True

        """
        try:
            return hash(frozenset(self._data.items()))
        except TypeError:
            # Re-raise as the built-in does, naming the unhashable value
            # rather than the `(key, value)` tuple holding it.
            for value in self._data.values():
                hash(value)
            raise  # pragma: no cover  # only if a value's hash is flaky

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
