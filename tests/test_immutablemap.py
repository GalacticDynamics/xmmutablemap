"""Test :mod:`xmmutablemap`."""

from collections import OrderedDict
from types import MappingProxyType
from typing import Any

import pytest

from xmmutablemap import ImmutableMap


class TestImmutableMap:
    """Test :class:`ImmutableMap`."""

    @pytest.fixture(scope="class")
    def d(self) -> ImmutableMap[str, Any]:
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
        assert d._data == dict(arg, **kwargs)

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
        assert hash(d) == hash(tuple(d.items()))

        # Not hashable if values aren't hashable.
        d = ImmutableMap(a=1, b={"c"})
        with pytest.raises(TypeError, match="unhashable type: 'set'"):
            hash(d)

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

        # Should raise TypeError if not a mapping.
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = d | 1

    def test_ror(self, d: ImmutableMap[str, Any]) -> None:
        """Test `__ror__`."""
        # Reverse order
        assert {"c": 3} | d == {"c": 3, "a": 1, "b": 2}
        assert OrderedDict([("c", 3)]) | d == OrderedDict(
            [("c", 3), ("a", 1), ("b", 2)]
        )

    # === Test pytree methods ===

    def test_tree_flatten(self, d: ImmutableMap[str, Any]) -> None:
        """Test `tree_flatten`."""
        assert d.tree_flatten() == ((1, 2), ("a", "b"))

    def test_tree_unflatten(self, d: ImmutableMap[str, Any]) -> None:
        """Test `tree_unflatten`."""
        d1 = ImmutableMap.tree_unflatten(("a", "b"), (1, 2))
        assert d1 == ImmutableMap(a=1, b=2)

        # round-trip
        d = ImmutableMap(a=1, b=2)
        flattened = d.tree_flatten()
        d2 = ImmutableMap.tree_unflatten(flattened[1], flattened[0])
        assert d2 == d
