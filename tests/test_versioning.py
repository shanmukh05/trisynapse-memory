from __future__ import annotations

from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from trisynapse_memory import MemoryEngine, __version__


ROOT = Path(__file__).resolve().parents[1]


def test_typescript_sdk_uses_trisynapse_npm_scope() -> None:
    package = json.loads(
        (ROOT / "packages/js-sdk/package.json").read_text(encoding="utf-8")
    )
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert package["name"] == "@trisynapse/trisynapse-memory"
    assert 'package="@trisynapse/trisynapse-memory@${version}"' in workflow
    assert 'from "@trisynapse/trisynapse-memory"' in workflow


@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        ("0.2.0", "0.2.0"),
        ("0.2.0a1", "0.2.0-alpha.1"),
        ("0.2.0b2", "0.2.0-beta.2"),
        ("0.2.0rc3", "0.2.0-rc.3"),
    ],
)
def test_npm_version_is_valid_semver(canonical: str, expected: str) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/version.py", "npm", canonical],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_all_version_surfaces_match_package_metadata() -> None:
    expected = package_version("trisynapse-memory")
    result = subprocess.run(
        [sys.executable, "scripts/version.py", "check", "--tag", f"v{expected}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert __version__ == expected
    assert MemoryEngine.VERSION == expected


def test_version_check_rejects_a_mismatched_tag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/version.py", "check", "--tag", "v999.0.0"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "does not match expected tag" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="install.sh targets macOS and Linux")
def test_posix_installer_is_ascii_and_expands_package_safely(tmp_path: Path) -> None:
    installer = ROOT / "install.sh"
    assert installer.read_bytes().isascii()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.log"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        """#!/bin/sh
case "$1:$2" in
  tool:install) printf '%s\n' "$4" > "$INSTALLER_UV_LOG" ;;
  tool:dir) printf '%s\n' "$INSTALLER_BIN_DIR" ;;
  *) exit 2 ;;
esac
""",
        encoding="ascii",
    )
    fake_uv.chmod(0o755)
    fake_command = bin_dir / "trisynapse-memory"
    fake_command.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"status\":\"ready\"}'\n",
        encoding="ascii",
    )
    fake_command.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "INSTALLER_BIN_DIR": str(bin_dir),
            "INSTALLER_UV_LOG": str(uv_log),
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "TRISYNAPSE_MEMORY_VERSION": "9.8.7",
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
    )
    result = subprocess.run(
        [shutil.which("sh") or "/bin/sh", str(installer)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Installing or upgrading trisynapse-memory[all]==9.8.7..." in result.stdout
    assert uv_log.read_text(encoding="ascii").strip() == (
        "trisynapse-memory[all]==9.8.7"
    )
    metadata = tmp_path / "state/trisynapse-memory/install.env"
    assert "version=9.8.7" in metadata.read_text(encoding="utf-8")
