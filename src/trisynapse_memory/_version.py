"""Resolve the public package version from one canonical source."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _source_tree_version() -> str | None:
    """Read pyproject.toml when running directly from a source checkout."""

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with pyproject.open("rb") as stream:
        value = tomllib.load(stream).get("project", {}).get("version")
    return str(value) if value else None


def get_version() -> str:
    """Return the source version in a checkout or installed distribution metadata."""

    source_version = _source_tree_version()
    if source_version is not None:
        return source_version
    try:
        return version("trisynapse-memory")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = get_version()

