<h1 align='center'> xmmutablemap </h1>
<h3 align="center"><code>JAX</code>-compatible Immutable Mapping</h3>

JAX prefers immutable objects but neither Python nor JAX provide an immutable
dictionary. 😢 </br> This repository defines a light-weight immutable map
(lower-level than a dict) that JAX understands as a PyTree. 🎉 🕶️

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

```bash
pip install xmmutablemap
```

<details>
  <summary>using <code>uv</code></summary>

```bash
uv add xmmutablemap
```

</details>
<details>
  <summary>from source, using pip</summary>

```bash
pip install git+https://github.com/GalacticDynamics/xmmutablemap.git
```

</details>
<details>
  <summary>building from source</summary>

```bash
cd /path/to/parent
git clone https://github.com/GalacticDynamics/xmmutablemap.git
cd xmmutablemap
pip install -e .  # editable mode
```

</details>

## Documentation

`xmutablemap` provides the class `ImmutableMap`, which is a full implementation
of
[Python's `Mapping` ABC](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes).
If you've used a `dict` then you already know how to use `ImmutableMap`! The
things `ImmutableMap` adds is 1) immutability (and related benefits like
hashability) and 2) compatibility with `JAX`.

```python
from xmmutablemap import ImmutableMap

print(ImmutableMap(a=1, b=2, c=3))
# ImmutableMap({'a': 1, 'b': 2, 'c': 3})

print(ImmutableMap({"a": 1, "b": 2.0, "c": "3"}))
# ImmutableMap({'a': 1, 'b': 2.0, 'c': '3'})
```

## Development

[![Actions Status][actions-badge]][actions-link]

We welcome contributions!

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/xmmutablemap/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/xmmutablemap/actions
[pypi-link]:                https://pypi.org/project/xmmutablemap/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/xmmutablemap
[pypi-version]:             https://img.shields.io/pypi/v/xmmutablemap

<!-- prettier-ignore-end -->
