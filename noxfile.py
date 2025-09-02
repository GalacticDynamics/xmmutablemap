"""Nox sessions."""

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = [
    "tests",
    # Linting
    "lint",
    "pylint",
    # Testing
    "test",
    # Packaging
    "build",
]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session, /) -> None:
    """Run the linter."""
    session.run(
        "uv",
        "run",
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """Run PyLint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.run("uv", "sync", "--group", "pylint")
    session.run("uv", "run", "pylint", "xmmutablemap", *session.posargs)


# =============================================================================
# Testing


@nox.session
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.run("uv", "sync", "--group", "test")
    session.run("uv", "run", "pytest", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the lints and tests."""
    session.notify("lint")
    session.notify("test")


# =============================================================================
# Packaging


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
