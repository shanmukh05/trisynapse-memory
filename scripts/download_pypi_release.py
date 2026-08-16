#!/usr/bin/env python3
"""Download the immutable PyPI artifacts for an already-published release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT = "trisynapse-memory"
NOT_FOUND = 3


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "trisynapse-memory-release"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed PyPI URLs
        return response.read()


def published_files(version: str) -> list[dict[str, object]] | None:
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    try:
        payload = json.loads(download(url))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    files = [
        item
        for item in payload.get("urls", [])
        if item.get("packagetype") in {"bdist_wheel", "sdist"}
    ]
    package_types = {str(item.get("packagetype")) for item in files}
    if package_types != {"bdist_wheel", "sdist"}:
        raise RuntimeError(
            f"PyPI {PROJECT} {version} must contain a wheel and source distribution"
        )
    return files


def save_file(destination: Path, item: dict[str, object]) -> Path:
    filename = str(item["filename"])
    if Path(filename).name != filename:
        raise RuntimeError(f"unsafe PyPI filename: {filename!r}")
    expected = str(dict(item["digests"])["sha256"])
    content = download(str(item["url"]))
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {filename}")

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    files = published_files(arguments.version)
    if files is None:
        print(f"PyPI {PROJECT} {arguments.version} is not published")
        return NOT_FOUND
    for item in files:
        target = save_file(arguments.destination, item)
        print(f"Reused immutable PyPI artifact: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
