<h1 align='center'> xmmutablemap </h1>
<h2 align="center"><cod>Jax</code>-compatible Immutable Map</h2>

JAX prefers immutable objects but neither Python nor JAX provide an immutable
dictionary. 😢 </br> This repository defines a light-weight immutable map
(lower-level than a dict) that JAX understands as a PyTree. 🎉 🕶️

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install xmmutablemap
```

## Documentation

<!-- [![Documentation Status][rtd-badge]][rtd-link] -->

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
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/xmmutablemap
[conda-link]:               https://github.com/conda-forge/xmmutablemap-feedstock
<!-- [github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/GalacticDynamics/xmmutablemap/discussions -->
[pypi-link]:                https://pypi.org/project/xmmutablemap/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/xmmutablemap
[pypi-version]:             https://img.shields.io/pypi/v/xmmutablemap
[zenodo-badge]:             https://zenodo.org/badge/755708966.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850557

<!-- prettier-ignore-end -->
