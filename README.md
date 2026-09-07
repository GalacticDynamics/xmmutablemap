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

### Relationship to `frozendict` (PEP 814)

Python 3.15 adds a built-in `frozendict`
([PEP 814](https://peps.python.org/pep-0814/)), which covers most of what
`ImmutableMap` was written for. `ImmutableMap` is now a **subclass of it**, so
the two interoperate directly:

```python
from xmmutablemap import ImmutableMap, frozendict

d = ImmutableMap(a=1, b=2)

print(isinstance(d, frozendict))
# True
print(d == frozendict(a=1, b=2) == {"a": 1, "b": 2})
# True
```

On Python 3.15+, `xmmutablemap.frozendict` _is_ the built-in. On older Pythons
it is a behaviour-matched stand-in, so this code works the same on every
supported version and you can migrate ahead of your Python floor.

The one thing the built-in cannot do is act as a JAX PyTree. JAX's registry is
keyed on the exact type, and `frozendict` is not registered, so JAX treats a
bare `frozendict` as an opaque **leaf** rather than a container:

```python
import jax

print(jax.tree.structure(frozendict(a=1, b=2)))
# PyTreeDef(*)
print(jax.tree.structure(ImmutableMap(a=1, b=2)))
# PyTreeDef(CustomNode(ImmutableMap[('a', 'b')], [*, *]))
```

That registration is the reason this package still exists. Once JAX registers
the built-in upstream, `ImmutableMap` becomes a thin alias and this package can
be retired.

### JAX Integration

One of the key benefits of `ImmutableMap` is its compatibility with JAX. Since
it's immutable and hashable, it can be used in places where JAX would normally
complain about mutable objects like regular dictionaries.

#### Using ImmutableMap as a Default in JAX Dataclasses

Here's an example showing how `ImmutableMap` can be used as a default value in a
dataclass, which is particularly useful with JAX:

```python
import functools
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from xmmutablemap import ImmutableMap


@functools.partial(
    jax.tree_util.register_dataclass, data_fields=["params"], meta_fields=["batch_size"]
)
@dataclass(frozen=True)
class Config:
    """Configuration with immutable default parameters."""

    # This works! ImmutableMap is immutable and hashable
    params: ImmutableMap[str, float] = ImmutableMap(
        learning_rate=0.001, momentum=0.9, weight_decay=1e-4
    )
    batch_size: int = 32


# JAX can safely transform functions using this dataclass
@jax.jit
def train_step(config: Config, data: jnp.ndarray) -> jnp.ndarray:
    """Example training step that uses config parameters."""
    lr = config.params["learning_rate"]
    return data * lr


# This works perfectly
config = Config()
data = jnp.array([1.0, 2.0, 3.0])
result = train_step(config, data)
print(f"Result: {result}")
# Result: [0.001 0.002 0.003]
```

#### Key Benefits for JAX

- **Immutability**: Once created, `ImmutableMap` cannot be modified, preventing
  accidental mutations that could break JAX's functional programming model
- **Hashability**: JAX can safely cache and memoize functions that use
  `ImmutableMap` instances
- **PyTree Support**: `ImmutableMap` is registered as a JAX PyTree, so it works
  seamlessly with JAX transformations like `jit`, `grad`, `vmap`, etc.
- **Safe Defaults**: Can be used as default values in dataclasses without the
  typical pitfalls of mutable defaults

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
