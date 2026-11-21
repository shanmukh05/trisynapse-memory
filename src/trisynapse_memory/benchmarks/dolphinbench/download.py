"""Install the pinned, runnable DolphinBench release subset safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO


RELEASE_COMMIT = "28a93b090102a093aea46c627911677e10891095"
MANIFEST_SHA256 = "40107f293243a6cb3941c16e75ce2e3495ed31c68a1015b8537efeb8dfec2ae9"
ARCHIVE_URL = f"https://github.com/mem0ai/dolphinbench/archive/{RELEASE_COMMIT}.tar.gz"

_ROOT_FILES = {
    "LICENSE",
    "README.md",
    "manifest.json",
    "requirements.txt",
    "session_map.json",
}
_ROOT_DIRECTORIES = {
    "authoring/release_manifests",
    "construction",
    "docs",
    "examples",
    "graders",
    "harness",
    "mock_mcp",
    "pricing",
    "registry",
    "tests",
}
_MARKER = ".trisynapse-download.json"
_SUBSET_VERSION = 2


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _allowed(relative: PurePosixPath) -> bool:
    value = relative.as_posix()
    if value in _ROOT_FILES:
        return True
    return any(value == root or value.startswith(root + "/") for root in _ROOT_DIRECTORIES)


def _relative_member(name: str) -> PurePosixPath | None:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive member: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    if len(path.parts) == 1:
        return None
    relative = PurePosixPath(*path.parts[1:])
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return relative


def _copy_member(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        shutil.copyfileobj(source, output)


def _verify_manifest(root: Path, expected_sha256: str) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("DolphinBench archive is missing manifest.json")
    content = manifest_path.read_bytes()
    actual = _sha256_bytes(content)
    if actual != expected_sha256:
        raise ValueError(
            f"DolphinBench manifest checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    manifest = json.loads(content)
    declared = manifest.get("files") or {}
    for relative, checksum in declared.items():
        relative_path = PurePosixPath(relative)
        if not _allowed(relative_path):
            continue
        path = root / relative
        if not path.is_file():
            raise ValueError(f"DolphinBench archive is missing declared file: {relative}")
        if _sha256_bytes(path.read_bytes()) != checksum:
            raise ValueError(f"DolphinBench file checksum mismatch: {relative}")
    return manifest


def _existing_install(target: Path, expected_manifest_sha256: str) -> dict | None:
    marker = target / _MARKER
    if not marker.is_file():
        return None
    installed = json.loads(marker.read_text(encoding="utf-8"))
    if not (
        installed.get("commit") == RELEASE_COMMIT
        and installed.get("manifest_sha256") == expected_manifest_sha256
        and installed.get("subset_version") == _SUBSET_VERSION
    ):
        return None
    _verify_manifest(target, expected_manifest_sha256)
    return installed


def install_from_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    expected_manifest_sha256: str = MANIFEST_SHA256,
) -> dict:
    """Extract and verify the runnable subset from an existing tar archive.

    Existing valid installs are returned unchanged.  Existing invalid or
    unrecognized directories are never overwritten.
    """

    archive_path = Path(archive).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    installed = _existing_install(target, expected_manifest_sha256)
    if installed is not None:
        return installed
    if target.exists():
        raise FileExistsError(f"refusing to replace existing DolphinBench directory: {target}")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dolphinbench-", dir=target.parent))
    extracted: list[str] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                relative = _relative_member(member.name)
                if relative is None:
                    continue
                if not _allowed(relative):
                    continue
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"unsupported archive member type: {member.name}")
                output = temporary / relative.as_posix()
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name}")
                _copy_member(source, output)
                extracted.append(relative.as_posix())
        manifest = _verify_manifest(temporary, expected_manifest_sha256)
        metadata = {
            "commit": RELEASE_COMMIT,
            "manifest_sha256": expected_manifest_sha256,
            "subset_version": _SUBSET_VERSION,
            "files": len(extracted),
            "release_version": manifest.get("release_version") or manifest.get("version"),
        }
        (temporary / _MARKER).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return metadata
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def download_release(
    destination: str | Path = "data/dolphinbench/release",
    *,
    url: str = ARCHIVE_URL,
) -> dict:
    """Download the pinned archive and install its runnable subset."""

    target = Path(destination).expanduser().resolve()
    installed = _existing_install(target, MANIFEST_SHA256)
    if installed is not None:
        return installed
    if target.exists():
        raise FileExistsError(f"refusing to replace existing DolphinBench directory: {target}")
    with tempfile.TemporaryDirectory(prefix="dolphinbench-download-") as directory:
        archive = Path(directory) / "release.tar.gz"
        request = urllib.request.Request(url, headers={"User-Agent": "trisynapse-memory/0.2"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        return install_from_archive(archive, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", default="data/dolphinbench/release", help="Release install directory"
    )
    args = parser.parse_args(argv)
    result = download_release(args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
