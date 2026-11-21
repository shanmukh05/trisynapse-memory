"""Watch local OmniMemEval queue and stage checkpoints."""

from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import json
import re
import signal
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryFile
from time import sleep

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Column, Table
from rich.text import Text

_STAGES = ("Ingestion", "Search", "Answers", "Judging", "Metrics", "Report")
_PERSONA_STAGES = ("Ingestion", "Search", "Answers", "Metrics", "Report")
_PERSONA_BATCH_SIZE = 20
_PERSONA_ADD_DONE = re.compile(r"\[ADD DONE] \S+ PersonaMem add [^\n]*? session=(pm_persona_\d+_chunk_\d+)\s")
_WORK_STATUSES = {"success", "success_empty", "skipped"}
_STYLES = {"complete": "green", "running": "cyan", "failed": "red", "blocked": "yellow",
           "interrupted": "yellow", "pending": "dim"}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return {}


def _count_rows(data: dict) -> int:
    return sum(
        row.get("status", "success") in _WORK_STATUSES
        for value in data.values()
        for row in (value if isinstance(value, list) else [value])
    )


class Checkpoints:
    def __init__(self, checkout: Path) -> None:
        self.checkout = checkout
        self._rows: dict[Path, tuple[int, int, int]] = {}

    def rows(self, path: Path) -> int:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return 0
        key = stat.st_ino, stat.st_mtime_ns
        saved = self._rows.get(path)
        if saved is None or saved[:2] != key:
            saved = (*key, _count_rows(_read_json(path)))
            self._rows[path] = saved
        return saved[2]


@lru_cache
def _persona_inputs(root: Path) -> tuple[dict[int, Path], Counter[int]]:
    dataset = root / "personamem"
    histories = {
        int(path.stem.rsplit("_persona", 1)[1]): path
        for path in (dataset / "data/chat_history_32k").glob("*_persona*.json")
    }
    with (dataset / "benchmark/text/benchmark.csv").open(newline="", encoding="utf-8") as stream:
        counts = Counter(int(row["persona_id"]) for row in csv.DictReader(stream))
    return {persona: histories[persona] for persona in sorted(counts.keys() & histories)}, counts


@lru_cache
def _persona_chunk_total(root: Path) -> int:
    histories, _ = _persona_inputs(root)
    total = 0
    for path in histories.values():
        messages = json.loads(path.read_bytes())["chat_history"]
        # The pinned harness batches first, then drops system and blank messages.
        total += sum(
            any(
                msg["role"] != "system" and msg.get("content", "").strip()
                for msg in messages[offset:offset + _PERSONA_BATCH_SIZE]
            )
            for offset in range(0, len(messages), _PERSONA_BATCH_SIZE)
        )
    return total


@lru_cache
def _work_totals(root: Path, checkout: Path, suite: str, last_unit: int | None) -> tuple[int, int, str] | None:
    try:
        if suite == "locomo":
            conversations = json.loads((root / "locomo/locomo10.json").read_bytes())
            sessions = sum(
                row["conversation"].get(f"session_{index}") is not None
                for row in conversations
                for index in range(35)
            )
            questions = sum(q["category"] != 5 for row in conversations for q in row["qa"])
            return sessions, questions, "sessions"
        if suite == "pmv2":
            histories, counts = _persona_inputs(root)
            selected = list(histories)
            if last_unit is not None:
                selected = selected[:last_unit + 1]
            return len(selected), sum(counts[persona] for persona in selected), "personas"
        if suite == "beam100k" and last_unit is None:
            sessions = questions = 0
            with (checkout / "data/beam/beam_100k.json").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    conversation = json.loads(line)
                    sessions += sum(bool(session) for session in conversation["chat"])
                    probes = conversation["probing_questions"]
                    if isinstance(probes, str):
                        probes = ast.literal_eval(probes)
                    questions += sum(len(group) for group in probes.values())
            return sessions, questions, "sessions"
        if suite == "halumem" and last_unit is None:
            sessions = questions = 0
            with (root / "halumem/HaluMem-Medium.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    user_sessions = json.loads(line)["sessions"]
                    sessions += len(user_sessions)
                    questions += sum(len(session.get("questions", [])) for session in user_sessions)
            return sessions, questions, "sessions"
    except FileNotFoundError:
        pass
    return None


def _checkpoint_count(root: Path, filename: str) -> int:
    try:
        return len({line for line in (root / filename).read_text(encoding="utf-8").splitlines() if line})
    except FileNotFoundError:
        return 0


@dataclass(frozen=True)
class Run:
    name: str
    state: str
    dependency: str | None
    root: Path
    manifest: dict
    planned: dict

    @property
    def stages(self) -> tuple[str, ...]:
        all_stages = _PERSONA_STAGES if self.root.parent.name == "pmv2" else _STAGES
        if self.state == "complete" and self.completed:
            return all_stages[:self.completed]
        return all_stages[:self.planned.get("to_step")]

    @property
    def completed(self) -> int:
        return self.manifest.get("completed_through", 0)

    @property
    def current(self) -> dict:
        attempts = self.manifest.get("attempts") or []
        stages = attempts[-1].get("stages", []) if attempts else []
        if stages and stages[-1]["through"] > self.completed:
            return stages[-1]
        return {}

    @property
    def last_unit(self) -> int | None:
        return self.manifest.get("configuration", {}).get("settings", self.planned).get("last_unit")

    def work(self, stage: int, data: Path, checkpoints: Checkpoints) -> tuple[int, int, str] | None:
        suite = self.root.parent.name
        config = self.manifest.get("configuration", {})
        last_unit = self.last_unit
        dataset = config.get("suite", self.planned.get("suite", suite)) if suite == "beam" else suite
        totals = _work_totals(data, checkpoints.checkout, dataset, last_unit)
        if totals is None or stage > (3 if suite == "pmv2" else 4):
            return None
        units, questions, ingest_unit = totals
        total, unit = (units, ingest_unit) if stage == 1 else (questions, "questions")
        if stage <= self.completed:
            return total, total, unit
        if self.state not in {"running", "failed", "interrupted"} or not self.current:
            return 0, total, unit
        if not self.current["from"] <= stage <= self.current["through"]:
            return 0, total, unit
        backend = config["backend"]
        tag = {"pmv2": "pm", "halumem": "hm"}.get(suite, suite)
        prefix = f"{backend}_{tag}"
        if stage == 1:
            filename = "streaming_completed.txt" if last_unit is not None else "success_records.txt"
            completed = _checkpoint_count(self.root, filename)
        elif stage == 2:
            completed = sum(
                checkpoints.rows(path) for path in (self.root / "tmp").glob(f"{prefix}_search_results_*.json")
            )
        elif stage == 3:
            completed = checkpoints.rows(self.root / f"{prefix}_responses.json")
        else:
            completed = max(
                checkpoints.rows(self.root / f"{prefix}_judged.json"),
                sum(1 for _ in (self.root / "judge-checkpoints").glob("*/*.json")),
            )
        return min(completed, total), total, unit


def _bar(completed: int, total: int, *, pulse: bool = False) -> ProgressBar:
    return ProgressBar(total=max(total, 1), completed=completed, pulse=pulse, width=22)


def _elapsed(start: str | None) -> str:
    if start is None:
        return ""
    seconds = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _stage_table(run: Run, data: Path, checkpoints: Checkpoints) -> Table:
    table = Table(title=f"{run.name} · {run.state}", title_justify="left", box=None, expand=True)
    table.add_column("Stage")
    table.add_column("Progress", width=22)
    table.add_column("Completed work")
    table.add_column("Status")
    for index, label in enumerate(run.stages, 1):
        current = run.current
        in_stage = current and current["from"] <= index <= current["through"]
        state = "complete" if index <= run.completed else run.state if in_stage else "pending"
        work = run.work(index, data, checkpoints)
        completed, total, unit = work if work is not None else (int(state == "complete"), 1, "")
        status = f"running {_elapsed(current.get('started_at'))}" if state == "running" else state
        table.add_row(
            label,
            _bar(completed, total, pulse=state == "running" and work is None),
            f"{completed:,}/{total:,} {unit}" if work is not None else "",
            Text(status, style=_STYLES.get(state)),
        )
        if index == 1 and state == "running" and run.root.parent.name == "pmv2" and run.last_unit is None:
            try:
                chunk_total = _persona_chunk_total(data)
                log = (run.root / "run.log").read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            if chunk_total:
                chunk_count = len(set(_PERSONA_ADD_DONE.findall(log)))
                table.add_row(
                    "  Chunks added", _bar(chunk_count, chunk_total),
                    f"{chunk_count:,}/{chunk_total:,} chunks", "",
                )
    return table


def _queue_view(
    path: Path, data: Path, plan: dict, checkpoints: Checkpoints, selected: str | None,
    *, previous_start: str | None = None,
) -> tuple[Group, bool]:
    current = _read_json(path)
    if not current or current.get("started_at") == previous_start:
        return Group(Panel(Text(f"Waiting for queue {path.parent.name}"), title="OmniMemEval progress")), False
    jobs = current["runs"]
    if selected and selected not in jobs:
        raise ValueError(f"{selected!r} is not in queue {path.parent.name!r}")
    runs = []
    for name, item in jobs.items():
        root = Path(item["run_directory"])
        runs.append(Run(name, item["status"], item.get("dependency"), root,
                        _read_json(root / "manifest.json"), plan.get(name, {})))
    counts = Counter(run.state for run in runs)
    header = Table.grid(Column(), Column(width=22), Column(), padding=(0, 1))
    header.add_row("Experiments", _bar(counts["complete"], len(runs)), f"{counts['complete']}/{len(runs)} complete")
    summary = "   ".join(f"{count} {status}" for status, count in counts.items())
    views = [Panel(Group(header, Text(summary)), title=f"OmniMemEval · {path.parent.name}")]
    details = [run for run in runs if run.name == selected or (not selected and run.state == "running")]
    views.extend(_stage_table(run, data, checkpoints) for run in details)

    overview = Table("Experiment", "Stages", "Done", "Status", box=None, expand=True)
    for run in runs:
        target = len(run.stages)
        state = run.state
        if run.dependency and state in {"blocked", "pending"} and jobs[run.dependency]["status"] != "complete":
            state = f"{state} · {run.dependency}"
        overview.add_row(
            run.name, _bar(run.completed, target), f"{run.completed}/{target}",
            Text(state, style=_STYLES.get(run.state)),
        )
    views.append(overview)
    return Group(*views), bool(current.get("ended_at"))


def _queue_is_locked(status: Path) -> bool:
    try:
        with status.with_name(".queue.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except FileNotFoundError:
        return False
    except BlockingIOError:
        return True
    return False


def _watch_queue(
    args: argparse.Namespace, path: Path, data: Path, plan: dict, checkpoints: Checkpoints, console: Console,
) -> int:
    view, ended = _queue_view(path, data, plan, checkpoints, args.run)
    if args.once or ended or not console.is_terminal:
        console.print(view)
    else:
        with Live(view, console=console, refresh_per_second=2, vertical_overflow="ellipsis") as live:
            while not ended:
                sleep(2)
                view, ended = _queue_view(path, data, plan, checkpoints, args.run)
                live.update(view, refresh=True)
    return 0


def _launch_queue(
    args: argparse.Namespace, path: Path, data: Path, plan: dict, checkpoints: Checkpoints, console: Console,
) -> int:
    command = [
        sys.executable, "-m", "trisynapse_memory.benchmarks.omnimemeval",
        "--checkout", str(args.checkout), "--data-root", str(data),
        "queue", str(args.plan), "--group", args.queue,
        "--jobs", str(args.jobs), "--llm-workers", str(args.llm_workers),
    ]
    if args.dry_run:
        return subprocess.call([*command, "--dry-run"], stdin=subprocess.DEVNULL)
    if _queue_is_locked(path):
        console.print("This queue is already running. Showing its progress.")
        return _watch_queue(args, path, data, plan, checkpoints, console)
    if not console.is_terminal:
        return subprocess.call(command, stdin=subprocess.DEVNULL)

    previous_start = _read_json(path).get("started_at")
    with TemporaryFile(mode="w+", encoding="utf-8") as output, subprocess.Popen(
        # A long-running queue can outlive the terminal attached to the launcher.
        command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            view, _ = _queue_view(path, data, plan, checkpoints, None, previous_start=previous_start)
            with Live(view, console=console, vertical_overflow="ellipsis") as live:
                while process.poll() is None:
                    sleep(1)
                    view, _ = _queue_view(path, data, plan, checkpoints, None, previous_start=previous_start)
                    live.update(view, refresh=True)
                view, _ = _queue_view(path, data, plan, checkpoints, None, previous_start=previous_start)
                live.update(view, refresh=True)
        except BaseException:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        code = process.wait()
        if code:
            output.seek(0)
            lines = output.readlines()
            if not any(line.startswith("Queue status: ") for line in lines) and _queue_is_locked(path):
                console.print("This queue is already running. Showing its progress.")
                return _watch_queue(args, path, data, plan, checkpoints, console)
            console.print(Text("".join(lines[-8:]).rstrip(), style="red"))
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", help="Queue name from --queue-name or --group")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--plan", type=Path, default=Path("configs/omnimemeval/rebuttal.json"))
    parser.add_argument("--run", help="Show stage details for a named experiment")
    parser.add_argument("--once", action="store_true", help="Print one snapshot")
    parser.add_argument("--launch", action="store_true", help="Start the named plan group and show progress")
    parser.add_argument("--dry-run", action="store_true", help="Preview the queue without starting it")
    parser.add_argument("--checkout", type=Path, default=Path("var/omnimemeval"), help="OmniMemEval harness checkout")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous experiments with --launch")
    parser.add_argument("--llm-workers", type=int, default=6, help="LLM workers per experiment with --launch")
    args = parser.parse_args()
    if args.queue in {".", ".."} or Path(args.queue).name != args.queue:
        parser.error("Use a queue name rather than a path")
    if args.launch and (args.once or args.run):
        parser.error("--launch shows the whole queue; use --once and --run when watching an existing queue")
    if args.dry_run and not args.launch:
        parser.error("--dry-run requires --launch")
    data = args.data_root.resolve()
    path = data / "omnimemeval" / "queues" / args.queue / "status.json"
    checkpoints = Checkpoints(args.checkout.resolve())
    console = Console()
    try:
        config = _read_json(args.plan)
        plan = {
            run["name"]: {**config.get("defaults", {}), **run.get("options", {}), "suite": run["suite"]}
            for run in config.get("runs", [])
        }
        if args.launch:
            return _launch_queue(args, path, data, plan, checkpoints, console)
        return _watch_queue(args, path, data, plan, checkpoints, console)
    except KeyboardInterrupt:
        return 130 if args.launch else 0
    except (OSError, ValueError) as exc:
        console.print(Text(str(exc), style="red"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
