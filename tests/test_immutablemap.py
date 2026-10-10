"""Test :mod:`xmmutablemap`."""

import copy
import pickle
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import jax
import pytest

from xmmutablemap import ImmutableMap, frozendict


class Plain(Mapping[str, int]):
    """A valid Mapping with no `__or__`."""

    def __init__(self, d: dict[str, int]) -> None:  # noqa: D107
        self._d = d

    def __getitem__(self, k: str) -> int:  # noqa: D105
        return self._d[k]

    def __iter__(self) -> Iterator[str]:  # noqa: D105
        return iter(self._d)

    def __len__(self) -> int:  # noqa: D105
        return len(self._d)


class TestImmutableMap:
    """Test :class:`ImmutableMap`."""

    @pytest.fixture
    def d(self) -> ImmutableMap[str, Any]:
        """Example immutable map."""
        return ImmutableMap(a=1, b=2)

    # ===============================================================

    @pytest.mark.parametrize(
        ("arg", "kwargs"),
        [
            ((), {}),
            ({"a": 1, "b": 2}, {}),
            ([("a", 1), ("b", 2)], {}),
            ((("a", 1), ("b", 2)), {}),
        ],
    )
    def test_init(
        self,
        arg: tuple[str, Any] | dict[str, Any] | list[tuple[str, Any]],
        kwargs: dict[str, Any],
    ) -> None:
        """Test initialization.

        Should be able to initialize with all the same input types as a regular
        dictionary.
        """
        d = ImmutableMap(arg, **kwargs)
        assert isinstance(d, ImmutableMap)
        assert d == dict(arg, **kwargs)

    def test_getitem(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__getitem__`."""
        assert d["a"] == 1
        assert d["b"] == 2

    def test_iter(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__iter__`."""
        assert list(d) == ["a", "b"]

    def test_len(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__len__`."""
        assert len(d) == 2

    def test_hash(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__hash__`."""
        assert isinstance(hash(d), int)
        assert hash(d) == hash(ImmutableMap(d))

        # Not hashable if values aren't hashable.
        d = ImmutableMap(a=1, b={"c"})
        with pytest.raises(TypeError, match="unhashable type: 'set'"):
            hash(d)

    def test_eq_with_other_mappings(self) -> None:
        """Test mapping interoperability for `__eq__`."""
        d = ImmutableMap(a=1, b=2)
        other_dict = {"a": 1, "b": 2}
        other_ordered_dict = OrderedDict([("a", 1), ("b", 2)])
        other_proxy = MappingProxyType({"a": 1, "b": 2})

        assert d == {"a": 1, "b": 2}
        assert d == OrderedDict([("a", 1), ("b", 2)])
        assert d == MappingProxyType({"a": 1, "b": 2})
        assert other_dict == d
        assert other_ordered_dict == d
        assert other_proxy == d

    def test_eq_with_unequal_mappings(self) -> None:
        """Test `__eq__` returns `False` for unequal mappings."""
        d = ImmutableMap(a=1, b=2)

        # Different length.
        assert d != {"a": 1}
        assert d != {"a": 1}

        # Same length, but a missing key.
        assert d != {"a": 1, "c": 2}
        assert d != {"a": 1, "c": 2}

        # Same keys, but a differing value.
        assert d != {"a": 1, "b": 3}
        assert d != {"a": 1, "b": 3}

    def test_eq_with_non_mapping(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__eq__` returns `NotImplemented`/`False` for non-mappings."""
        assert d != 1
        assert d.__eq__(1) is NotImplemented

    def test_eq_and_hash_ignore_insertion_order(self) -> None:
        """Test equality/hash contract for same items in different orders."""
        d1 = ImmutableMap(a=1, b=2)
        d2 = ImmutableMap(b=2, a=1)
        assert d1 == d2
        assert hash(d1) == hash(d2)

        # The invariant is what makes set/dict membership work.
        assert len({d1, d2}) == 1
        assert d2 in {d1: "value"}

    def test_keys(self, d: ImmutableMap[str, Any]) -> None:
        """Test `keys`."""
        assert list(d.keys()) == ["a", "b"]

    def test_values(self, d: ImmutableMap[str, Any]) -> None:
        """Test `values`."""
        assert list(d.values()) == [1, 2]

    def test_items(self, d: ImmutableMap[str, Any]) -> None:
        """Test `items`."""
        assert list(d.items()) == [("a", 1), ("b", 2)]

    def test_repr(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__repr__`."""
        assert repr(d) == "ImmutableMap({'a': 1, 'b': 2})"

    def test_or(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__or__`."""
        assert d | ImmutableMap(c=3) == ImmutableMap(a=1, b=2, c=3)
        assert d | {"c": 3} == ImmutableMap(a=1, b=2, c=3)
        assert d | OrderedDict([("c", 3)]) == ImmutableMap(a=1, b=2, c=3)
        assert d | MappingProxyType({"c": 3}) == ImmutableMap(a=1, b=2, c=3)

        # Looser than `frozendict`: any Mapping merges, even one with no `|`.
        assert d | Plain({"c": 3}) == ImmutableMap(a=1, b=2, c=3)
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = frozendict(a=1) | Plain({"c": 3})
        # ...but only on the left; the reflected `|` is `frozendict`'s.
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = Plain({"c": 3}) | d

        # Should raise TypeError if not a mapping.
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = d | 1

    def test_ror(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__ror__`."""
        # Reverse order
        assert {"c": 3} | d == {"c": 3, "a": 1, "b": 2}
        assert OrderedDict([("c", 3)]) | d == OrderedDict(
            [("c", 3), ("a", 1), ("b", 2)],
        )

    # === Test pytree methods ===

    def test_tree_flatten(self, d: ImmutableMap[str, Any]) -> None:
        """Test `tree_flatten`."""
        assert d.tree_flatten() == ((1, 2), ("a", "b"))

    def test_tree_flatten_sorts_keys(self) -> None:
        """Leaves come out in sorted-key order, like JAX's `dict`/`frozendict`."""
        d = ImmutableMap(b=1, a=2)
        assert d.tree_flatten() == ((2, 1), ("a", "b"))
        assert jax.tree.leaves(d) == [2, 1]
        assert list(jax.tree.map(lambda x: x, d)) == ["a", "b"]

    def test_tree_flatten_unsortable_keys(self) -> None:
        """Keys that can't be ordered raise, as JAX does for `dict`."""
        with pytest.raises(TypeError):
            ImmutableMap({1: "x", "b": 2}).tree_flatten()

    def test_tree_unflatten(self, d: ImmutableMap[str, Any]) -> None:
        """Test `tree_unflatten`."""
        d1 = ImmutableMap.tree_unflatten(("a", "b"), (1, 2))
        assert d1 == ImmutableMap(a=1, b=2)

        # round-trip
        d = ImmutableMap(a=1, b=2)
        flattened = d.tree_flatten()
        d2 = ImmutableMap.tree_unflatten(flattened[1], flattened[0])
        assert d2 == d


class TestFrozenDictCompat:
    """Test that `ImmutableMap` is a drop-in `frozendict`.

    These run against the built-in `frozendict` on Python 3.15+ and against
    the backport below that, so they pin the behaviour that must not diverge
    between the two.
    """

    def test_is_a_frozendict(self) -> None:
        """`ImmutableMap` is a `frozendict` subclass."""
        d = ImmutableMap(a=1, b=2)
        assert isinstance(d, frozendict)
        assert issubclass(ImmutableMap, frozendict)

    def test_eq_across_mapping_types(self) -> None:
        """Equality holds against any equal mapping."""
        d = ImmutableMap(a=1, b=2)
        assert d == {"a": 1, "b": 2}
        assert d == frozendict(a=1, b=2)
        assert d == MappingProxyType({"a": 1, "b": 2})
        assert d != {"a": 1}

    def test_hash_matches_frozendict(self) -> None:
        """An `ImmutableMap` hashes as the equal `frozendict` does.

        `ImmutableMap(...) == frozendict(...)`, so if the hashes disagreed the
        two would collide incorrectly in sets and dicts.
        """
        assert hash(ImmutableMap(a=1, b=2)) == hash(frozendict(a=1, b=2))
        assert len({ImmutableMap(a=1, b=2), frozendict(a=1, b=2)}) == 1

    def test_no_mutating_methods(self) -> None:
        """The mutating half of the `dict` API is absent."""
        d = ImmutableMap(a=1, b=2)
        for name in ("clear", "pop", "popitem", "setdefault", "update"):
            assert not hasattr(d, name)

        with pytest.raises(TypeError):
            d["c"] = 3  # type: ignore[index]

        with pytest.raises(TypeError):
            del d["a"]  # type: ignore[attr-defined]

    def test_type_preserved_by_or_and_copy(self) -> None:
        """`|` and `copy` keep the subclass.

        `frozendict`'s C implementation hardcodes its result type, so without
        the overrides these would silently degrade to a plain `frozendict` --
        which is not a PyTree, so JAX would start treating it as a leaf.
        """
        d = ImmutableMap(a=1, b=2)
        assert type(d | {"c": 3}) is ImmutableMap
        assert type(d | frozendict(c=3)) is ImmutableMap
        assert type(d | MappingProxyType({"c": 3})) is ImmutableMap
        assert type(d | OrderedDict([("c", 3)])) is ImmutableMap
        assert d.copy() is d

    def test_repr_roundtrips(self) -> None:
        """The repr matches `frozendict`'s, including when empty."""
        assert repr(ImmutableMap(a=1, b=2)) == "ImmutableMap({'a': 1, 'b': 2})"
        assert repr(ImmutableMap()) == "ImmutableMap()"

    @pytest.mark.parametrize("cls", [frozendict, ImmutableMap])
    def test_pickle(self, cls: type) -> None:
        """Instances round-trip through `pickle`, keeping their type.

        The backport defines no `__reduce__`: default slot-state pickling
        covers it. This is what catches that going wrong.
        """
        d = cls(a=1, b=2)
        assert pickle.loads(pickle.dumps(d)) == d  # noqa: S301
        assert type(pickle.loads(pickle.dumps(d))) is cls  # noqa: S301

    def test_reversed_and_fromkeys(self) -> None:
        """`__reversed__` and `fromkeys` come from `frozendict`."""
        assert list(reversed(ImmutableMap(a=1, b=2))) == ["b", "a"]
        assert ImmutableMap.fromkeys(["a", "b"], 0) == {"a": 0, "b": 0}

    def test_unhashable_values(self) -> None:
        """Hashing requires hashable values, as `frozendict` does."""
        with pytest.raises(TypeError, match="unhashable type"):
            hash(ImmutableMap(a=[1]))


class TestFrozenDict:
    """Test the `frozendict` name itself.

    On Python 3.15+ this is the PEP 814 built-in; below that it is the
    backport. These pin the behaviour that must hold either way.
    """

    def test_or_with_non_mapping(self) -> None:
        """`|` with a non-mapping is a `TypeError`, from either side."""
        d = frozendict(a=1)

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = d | 1

        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = 1 | d

    def test_or_returns_plain_frozendict(self) -> None:
        """`|` yields a `frozendict`, and merges from either side."""
        assert frozendict(a=1) | {"b": 2} == {"a": 1, "b": 2}
        assert type(frozendict(a=1) | {"b": 2}) is frozendict
        assert {"b": 2} | frozendict(a=1) == {"b": 2, "a": 1}

    def test_ror_with_mapping_that_has_no_or(self) -> None:
        """A `Mapping` implementing no `|` raises, as it does for the built-in.

        `__ror__` delegates to the left operand, so a mapping that cannot
        merge at all cannot merge here either. Verified against the 3.15
        built-in, which raises the same way.
        """
        with pytest.raises(TypeError, match="'Plain' and 'frozendict'"):
            _ = Plain({"b": 2}) | frozendict(a=1)

    def test_or_only_merges_dicts(self) -> None:
        """`|` merges only a `dict` or `frozendict`, not any `Mapping`."""
        with pytest.raises(TypeError, match="'frozendict' and 'Plain'"):
            _ = frozendict(a=1) | Plain({"b": 2})
        # A mappingproxy has its own `|`, which unwraps it to a dict.
        assert frozendict(a=1) | MappingProxyType({"b": 2}) == {"a": 1, "b": 2}
        assert type(OrderedDict(b=2) | frozendict(a=1)) is OrderedDict

    def test_init_cannot_mutate(self) -> None:
        """Construction happens in `__new__`; calling `__init__` is a no-op."""
        d = frozendict(a=1)
        d.__init__(b=2)  # type: ignore[misc]
        assert d == {"a": 1}

    def test_too_many_args(self) -> None:
        """More than one positional argument names `frozendict`."""
        with pytest.raises(TypeError, match="^frozendict expected at most 1"):
            frozendict({}, {})  # type: ignore[call-overload]

    def test_constructor_returns_same_frozendict(self) -> None:
        """`frozendict(d)` of an exact `frozendict` is `d`, as for `tuple`."""
        d = frozendict(a=1)
        assert frozendict(d) is d
        assert frozendict(d, b=2) is not d
        assert type(frozendict(ImmutableMap(a=1))) is frozendict

    def test_subclass_init_receives_args(self) -> None:
        """A subclass `__init__` runs after `__new__` has built the data."""

        class Sub(frozendict[str, int]):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                self.seen = dict(self)

        assert Sub(a=1).seen == {"a": 1}

    def test_hash_is_frozenset_of_items(self) -> None:
        """The hash is that of the items' frozenset, as for the built-in."""
        d = frozendict(a=1, b=2)
        assert hash(d) == hash(frozenset(d.items()))
        with pytest.raises(TypeError, match="^unhashable type: 'list'$"):
            hash(frozendict(a=[1]))

    def test_copy_module(self) -> None:
        """`copy.copy` returns an exact `frozendict` itself; deepcopy copies."""
        d = frozendict(a=[1])
        assert copy.copy(d) is d
        deep = copy.deepcopy(d)
        assert deep == d
        assert deep["a"] is not d["a"]

        class Sub(frozendict[str, Any]):
            pass

        s = Sub(a=1)
        s.x = 3  # type: ignore[attr-defined]
        c = copy.copy(s)
        assert type(c) is Sub
        assert c is not s
        assert c == s
        assert c.x == 3  # type: ignore[attr-defined]

    def test_getnewargs(self) -> None:
        """Pickling goes through `__getnewargs__`, with a copy of the data."""
        d = frozendict(a=1)
        (data,) = d.__getnewargs__()
        assert data == {"a": 1}
        data["b"] = 2
        assert d == {"a": 1}

    @pytest.mark.parametrize("protocol", [0, 1])
    def test_old_pickle_protocols_refused(self, protocol: int) -> None:
        """Protocols 0 and 1 cannot rebuild through `__new__`, so they raise."""
        with pytest.raises(TypeError, match="cannot pickle 'frozendict' object"):
            pickle.dumps(frozendict(a=1), protocol)

    @pytest.mark.parametrize("cls", [frozendict, ImmutableMap])
    def test_no_attribute_mutation(self, cls: type) -> None:
        """Attributes can be neither set nor deleted, on the subclass too."""
        d = cls(a=1)

        with pytest.raises(AttributeError):
            d.x = 1

        with pytest.raises(AttributeError):
            del d.x
