# Project Overview

This repository provides `ImmutableMap`, an immutable and hashable mapping type
that implements `collections.abc.Mapping` and is compatible with JAX PyTrees.

- **Language**: Python
- **Main API**: `ImmutableMap`
  - Constructible from dicts, iterables of pairs, or keyword args.
  - Implements `__len__`, `__iter__`, `__getitem__`.
  - Must remain immutable and hashable.
- **Design goals**: Immutability, hashability, Mapping compliance, JAX
  interoperability.
- **JAX integration**: Objects may be used as PyTrees. Know how to define
  `tree_flatten` and `tree_unflatten`. Performant.

## Folder Structure

- `/src`: Contains the source code.
- `README.md`: Project documentation and usage examples. The Python code blocks
  are also tested as part of the test suite.
- Tests:
  - `noxfile.py`: Nox configuration for sessions like linting, testing, and
    building.
  - `conftest.py`: Pytest configuration and fixtures.
  - `/tests`: Contains the tests.

## Coding Style

- Always use type hints (standard typing, `collections.abc.Mapping`, `TypeVar`,
  etc.).
- Immutability is a core constraint: methods return new objects, never mutate.
- Keep dependencies minimal; JAX is not a core runtime dependency, but the code
  is designed and tested with JAX interoperability in mind.
- Docstrings should be concise and include usage examples.

## Tooling

- This repo uses `uv` for managing virtual environments and running commands.
- This repo uses `nox` for testing and automation.
- Before committing, to do a full linting and testing, run:

  ```bash
  uv run nox -s check
  ```

## Testing

- Use `pytest` for all test suites.
- Add unit tests for every new function or class.
- Encourage property-based testing with `hypothesis` to validate Mapping laws:
  - insertion order
  - equality semantics
  - immutability
- For JAX-related behavior:
  - Confirm PyTree registration (`tree_structure`).
  - Verify compatibility with transformations like `jit` and `vmap`.
  - Tests should run on CPU by default; no accelerators required.

## Final Notes

Prefer clarity over cleverness. Preserve immutability and Mapping semantics.
When in doubt, match Python’s Mapping API and JAX’s PyTree conventions.
