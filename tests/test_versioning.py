from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
import subprocess
import sys

from trisynapse_memory import MemoryEngine, __version__


ROOT = Path(__file__).resolve().parents[1]


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
