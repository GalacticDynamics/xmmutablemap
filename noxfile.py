"""Nox sessions."""

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = [
    # Linting
    "lint",
    "pylint",
    "precommit",
    # Testing
    "tests",
    # Packaging
    "build",
]
nox.options.default_venv_backend = "uv"


# =============================================================================
# Linting


@nox.session(venv_backend="uv")
def lint(session: nox.Session, /) -> None:
    """Run the linter."""
    precommit(session)  # reuse pre-commit session
    pylint(session)  # reuse pylint session


@nox.session(venv_backend="uv")
def precommit(session: nox.Session, /) -> None:
    """Run pre-commit."""
    session.run_install(
        "uv",
        "sync",
        "--group=lint",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(venv_backend="uv")
def pylint(session: nox.Session, /) -> None:
    """Run PyLint."""
    session.run_install(
        "uv",
        "sync",
        "--group=lint",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pylint", "xmmutablemap", *session.posargs)


# =============================================================================
# Testing


@nox.session(venv_backend="uv")
def tests(session: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    session.run_install(
        "uv",
        "sync",
        "--group=test",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", *session.posargs)


# =============================================================================
# Packaging


@nox.session(venv_backend="uv")
def rm_build(_: nox.Session, /) -> None:
    """Remove the build directory."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)


@nox.session(venv_backend="uv")
def build(session: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.run_install(
        "uv",
        "sync",
        "--group=build",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("build")
