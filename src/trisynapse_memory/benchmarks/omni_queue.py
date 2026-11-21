"""Dependency-aware parallel execution of named OmniMemEval runs."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep


@dataclass(frozen=True)
class QueuedRun:
    name: str
    argv: list[str]
    directory: Path
    dependency: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_queue(runs: list[QueuedRun], *, checkout: Path, data: Path, jobs: int, name: str) -> int:
    """Save queue progress and isolate failures by job and dependency."""
    root = data / "omnimemeval" / "queues" / name
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _drain(runs, checkout, data, jobs, root)


def _drain(runs: list[QueuedRun], checkout: Path, data: Path, jobs: int, root: Path) -> int:
    path = root / "status.json"
    status = {run.name: {
        "status": "pending",
        "dependency": run.dependency,
        "run_directory": str(run.directory),
        "log": str(root / f"{run.name}.log"),
    } for run in runs}
    parent = [sys.executable, "-m", "trisynapse_memory.benchmarks.omnimemeval",
              "--checkout", str(checkout), "--data-root", str(data)]
    pending = {run.name: run for run in runs}
    active: dict[str, subprocess.Popen] = {}
    started = _now()
    _write_status(path, {"started_at": started, "jobs": jobs, "runs": status})
    print(f"Queue status: {path}", flush=True)
    interrupted = False
    try:
        while pending or active:
            changed = _collect(active, status)
            changed |= _launch(pending, active, status, jobs, parent, root)
            if changed:
                _write_status(path, {"started_at": started, "jobs": jobs, "runs": status})
            if active:
                sleep(0.5)
            elif pending and not changed:
                raise RuntimeError("Queue has pending runs with unresolved dependencies")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for name, process in active.items():
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            status[name].update(status="interrupted", ended_at=_now())
        _write_status(path, {"started_at": started, "ended_at": _now(), "jobs": jobs, "runs": status})
    if interrupted:
        return 130
    return 0 if all(item["status"] == "complete" for item in status.values()) else 1


def _collect(active: dict[str, subprocess.Popen], status: dict) -> bool:
    done = {name: code for name, process in active.items() if (code := process.poll()) is not None}
    for name, code in done.items():
        del active[name]
        result = "complete" if code == 0 else "failed"
        status[name].update(status=result, exit_code=code, ended_at=_now())
        print(f"{name}: {result}" + (f" · {status[name]['log']}" if code else ""), flush=True)
    return bool(done)


def _launch(pending: dict, active: dict, status: dict, jobs: int, parent: list[str], root: Path) -> bool:
    changed = False
    for name, run in list(pending.items()):
        dependency = status[run.dependency]["status"] if run.dependency else "complete"
        if dependency in {"failed", "blocked"}:
            del pending[name]
            status[name].update(status="blocked", ended_at=_now())
            print(f"{name}: blocked by {run.dependency}", flush=True)
            changed = True
        elif dependency == "complete" and len(active) < jobs:
            changed = True
            del pending[name]
            command = [*parent, *run.argv]
            if (run.directory / "manifest.json").is_file():
                command.append("--resume")
            try:
                with (root / f"{name}.log").open("a", encoding="utf-8") as log:
                    active[name] = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
            except OSError as exc:
                status[name].update(status="failed", error=str(exc), ended_at=_now())
                print(f"{name}: failed to start · {exc}", flush=True)
                continue
            status[name].update(status="running", started_at=_now())
            print(f"{name}: running", flush=True)
    return changed
