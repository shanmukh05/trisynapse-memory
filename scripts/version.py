#!/usr/bin/env python3
"""Set and verify every Trisynapse Memory release-version surface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"
JS_PACKAGE = ROOT / "packages/js-sdk/package.json"
STUDIO_PACKAGE = ROOT / "packages/studio/package.json"
INSTALL_SH = ROOT / "install.sh"
INSTALL_PS1 = ROOT / "install.ps1"

VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?$")
PRERELEASE_PATTERN = re.compile(
    r"^(?P<base>[0-9]+\.[0-9]+\.[0-9]+)(?P<phase>a|b|rc)(?P<number>[0-9]+)$"
)
PYPROJECT_PATTERN = re.compile(r'(?m)^(version\s*=\s*)"[^"]+"\s*$')
INSTALL_SH_PATTERN = re.compile(
    r'(?m)^(?P<prefix>VERSION="\$\{TRISYNAPSE_MEMORY_VERSION:-)'
    r'(?P<version>[^}]+)(?P<suffix>\}")$'
)
INSTALL_PS1_PATTERN = re.compile(
    r'(?m)^(?P<prefix>\$Version = if \(\$env:TRISYNAPSE_MEMORY_VERSION\) '
    r'\{ \$env:TRISYNAPSE_MEMORY_VERSION \} else \{ ")'
    r'(?P<version>[^"]+)(?P<suffix>" \})$'
)


class VersionError(RuntimeError):
    """A release version is invalid or inconsistent."""


def canonical_version() -> str:
    with PYPROJECT.open("rb") as stream:
        value = tomllib.load(stream).get("project", {}).get("version")
    if not value:
        raise VersionError("pyproject.toml does not define project.version")
    return str(value)


def validate_version(value: str) -> str:
    if not VERSION_PATTERN.fullmatch(value):
        raise VersionError(
            "version must look like 0.2.0 or 0.2.0rc1; do not include the leading v"
        )
    return value


def npm_version(value: str) -> str:
    """Convert the canonical PEP 440 version to an npm-compatible SemVer value."""

    value = validate_version(value)
    match = PRERELEASE_PATTERN.fullmatch(value)
    if match is None:
        return value
    phase = {"a": "alpha", "b": "beta", "rc": "rc"}[match.group("phase")]
    return f"{match.group('base')}-{phase}.{match.group('number')}"


def _json_version(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["version"])


def _matched_version(path: Path, pattern: re.Pattern[str], label: str) -> str:
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise VersionError(f"could not find {label} in {path.relative_to(ROOT)}")
    return match.group("version")


def _uv_lock_version() -> str:
    with UV_LOCK.open("rb") as stream:
        packages = tomllib.load(stream).get("package", [])
    matches = [item for item in packages if item.get("name") == "trisynapse-memory"]
    if len(matches) != 1:
        raise VersionError("uv.lock must contain exactly one trisynapse-memory package")
    return str(matches[0].get("version"))


def version_surfaces() -> dict[str, str]:
    return {
        "pyproject.toml": canonical_version(),
        "uv.lock": _uv_lock_version(),
        "packages/js-sdk/package.json": _json_version(JS_PACKAGE),
        "packages/studio/package.json": _json_version(STUDIO_PACKAGE),
        "install.sh": _matched_version(INSTALL_SH, INSTALL_SH_PATTERN, "default VERSION"),
        "install.ps1": _matched_version(
            INSTALL_PS1, INSTALL_PS1_PATTERN, "default $Version"
        ),
    }


def check_versions(tag: str | None = None) -> str:
    expected = validate_version(canonical_version())
    surfaces = version_surfaces()
    npm_expected = npm_version(expected)
    expectations = {
        path: npm_expected
        if path in {"packages/js-sdk/package.json", "packages/studio/package.json"}
        else expected
        for path in surfaces
    }
    mismatches = {
        path: value
        for path, value in surfaces.items()
        if value != expectations[path]
    }
    if mismatches:
        rendered = ", ".join(
            f"{path}={value} (expected {expectations[path]})"
            for path, value in mismatches.items()
        )
        raise VersionError(f"version mismatch: {rendered}")
    if tag is not None and tag != f"v{expected}":
        raise VersionError(f"tag {tag!r} does not match expected tag 'v{expected}'")
    return expected


def _replace_once(path: Path, pattern: re.Pattern[str], replacement: str) -> str:
    current = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(replacement, current, count=1)
    if count != 1:
        raise VersionError(f"expected one version field in {path.relative_to(ROOT)}")
    return updated


def _package_json(path: Path, release_version: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = release_version
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except FileNotFoundError as exc:
        raise VersionError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise VersionError(f"command failed ({exc.returncode}): {' '.join(command)}") from exc


def set_version(release_version: str) -> str:
    release_version = validate_version(release_version)
    package_version = npm_version(release_version)
    tracked = [PYPROJECT, UV_LOCK, JS_PACKAGE, STUDIO_PACKAGE, INSTALL_SH, INSTALL_PS1]
    originals = {path: path.read_bytes() for path in tracked}
    updates: dict[Path, str] = {
        PYPROJECT: _replace_once(
            PYPROJECT, PYPROJECT_PATTERN, rf'\g<1>"{release_version}"'
        ),
        JS_PACKAGE: _package_json(JS_PACKAGE, package_version),
        STUDIO_PACKAGE: _package_json(STUDIO_PACKAGE, package_version),
        INSTALL_SH: _replace_once(
            INSTALL_SH,
            INSTALL_SH_PATTERN,
            rf'\g<prefix>{release_version}\g<suffix>',
        ),
        INSTALL_PS1: _replace_once(
            INSTALL_PS1,
            INSTALL_PS1_PATTERN,
            rf'\g<prefix>{release_version}\g<suffix>',
        ),
    }
    try:
        for path, text in updates.items():
            _atomic_write(path, text)
        _run([shutil.which("uv") or "uv", "lock"])
        return check_versions()
    except Exception:
        for path, content in originals.items():
            path.write_bytes(content)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Set or verify the Trisynapse Memory release version."
    )
    subcommands = value.add_subparsers(dest="command", required=True)
    subcommands.add_parser("current", help="Print the canonical version.")
    npm = subcommands.add_parser(
        "npm", help="Print the npm SemVer for a canonical release version."
    )
    npm.add_argument(
        "version", nargs="?", help="Canonical version; defaults to pyproject.toml."
    )
    check = subcommands.add_parser("check", help="Fail when version surfaces disagree.")
    check.add_argument("--tag", help="Also require this tag to equal v<version>.")
    update = subcommands.add_parser("set", help="Update every version surface atomically.")
    update.add_argument("version", help="PEP 440 version without a leading v.")
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        handlers: dict[str, Callable[[], str]] = {
            "current": canonical_version,
            "npm": lambda: npm_version(arguments.version or canonical_version()),
            "check": lambda: check_versions(arguments.tag),
            "set": lambda: set_version(arguments.version),
        }
        release_version = handlers[arguments.command]()
    except VersionError as exc:
        print(f"version error: {exc}", file=sys.stderr)
        return 1
    if arguments.command in {"current", "npm"}:
        print(release_version)
    else:
        verb = "set" if arguments.command == "set" else "verified"
        print(f"Trisynapse Memory version {verb}: {release_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
