"""Prepare, run, and inspect published OmniMemEval user-memory evaluations."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

if os.name != "nt":
    import fcntl

from trisynapse_memory.benchmarks.omni_backend import DEFAULT_EMBEDDING_MODEL, DEFAULT_MEMORY_MODEL
from trisynapse_memory.benchmarks.omni_queue import QueuedRun, run_queue
from trisynapse_memory.engine.memory_log.store import SQLiteMemoryLogStore
from trisynapse_memory.engine.models import RetrievalConfiguration
from trisynapse_memory.prompts import prompt_provenance

UPSTREAM = "https://github.com/MemTensor/OmniMemEval.git"
UPSTREAM_REVISION = "ff13e4a5b12e3c406a81c32bb0834c7e8d73f38e"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
OPENAI_ENDPOINT = "https://api.openai.com/v1"
SUITES = {
    "locomo": ("locomo", "locomo", 6),
    "personamem": ("pmv2", "pmv2", 5),
    "halumem": ("halumem", "halumem", 6),
    "beam100k": ("beam", "beam", 6),
    "beam10m": ("beam", "beam", 6),
    "longmemeval": ("lme", "lme", 6),
}
BEAM_FILES = {
    "beam100k": ("100k", "100K-*.parquet", "beam_100k.json"),
    "beam10m": ("10m", "10M-*.parquet", "beam_10m_10m.json"),
}
LOCAL_INPUTS = {
    "locomo": ("locomo", "locomo", ("locomo10.json",)),
    "longmemeval": ("longmemeval", "longmemeval", ("longmemeval_s_cleaned.json",)),
    "halumem": ("halumem", "halumem", ("HaluMem-Medium.jsonl",)),
    "personamem": ("personamem", "personamem_v2", ("benchmark/text/benchmark.csv", "data/chat_history_32k")),
}
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}")

# Backend registration, request plumbing, and checkpoints are the upstream overrides.
# Scoring, ingestion sequence, prompt rendering, and reports stay upstream.
_UPSTREAM_EDITS = {
    "scripts/client_factory/registry.py": (
        '    "memos":       ("memos_client",       "MemosClient"),',
        '    "memos":       ("memos_client",       "MemosClient"),\n'
        '    "trisynapse":  ("trisynapse_client",  "TrisynapseClient"),',
    ),
    "scripts/utils/search_helpers.py": (
        '    "memos": generic_text_search,',
        '    "memos": generic_text_search,\n'
        '    "trisynapse": generic_text_search,',
    ),
    "scripts/locomo/locomo_search.py": (
        '        "memos": generic_text_search,',
        '        "memos": generic_text_search,\n'
        '        "trisynapse": generic_text_search,',
    ),
    "scripts/utils/ingest_helpers.py": (
        '_CONV_ID_LIBS = frozenset({"memos", "everos"})',
        '_CONV_ID_LIBS = frozenset({"memos", "everos", "trisynapse"})',
    ),
    "scripts/locomo/locomo_ingestion.py": (
        '        if frame == "memos":\n            client.add(speaker_a_messages, speaker_a_user_id, sess_id)',
        '        if frame in {"memos", "trisynapse"}:\n            client.add(speaker_a_messages, speaker_a_user_id, sess_id)',
    ),
}
_LOCOMO_JUDGE = "scripts/locomo/locomo_eval.py"
_LOCOMO_JUDGE_EDITS = (
    (
        "from utils.checkpoint import atomic_json_dump",
        "from utils.checkpoint import atomic_json_dump\n"
        "from trisynapse_memory.benchmarks.omni_checkpoints import LoCoMoJudgeCheckpoint",
    ),
    (
        "    graded_responses = []\n",
        "    checkpoint = LoCoMoJudgeCheckpoint(group_id, eval_model, num_runs)\n"
        "    graded_responses = []\n",
    ),
    (
        "    for i, response in enumerate(group_responses, 1):\n",
        "    for i, response in enumerate(group_responses, 1):\n"
        "        previous = checkpoint.load(i, response)\n"
        "        if previous is not None:\n"
        "            graded_responses.append(previous)\n"
        "            continue\n",
    ),
    (
        "        graded_responses.append(graded_response)\n",
        "        checkpoint.save(i, response, convert_numpy_types(graded_response))\n"
        "        graded_responses.append(graded_response)\n",
    ),
)


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def _install_backend(checkout: Path) -> None:
    if not (checkout / ".git").exists():
        raise FileNotFoundError(f"OmniMemEval checkout missing at {checkout}; run setup first")
    if _git(checkout, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise ValueError(f"OmniMemEval must use revision {UPSTREAM_REVISION} at {checkout}")
    with (checkout / ".trisynapse-backend.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _apply_backend_hooks(checkout)


def _apply_backend_hooks(checkout: Path) -> None:
    edits = {path: (edit,) for path, edit in _UPSTREAM_EDITS.items()}
    edits[_LOCOMO_JUDGE] = _LOCOMO_JUDGE_EDITS
    unfamiliar = set(_git(checkout, "diff", "--name-only", "HEAD").splitlines()) - edits.keys()
    if unfamiliar:
        raise ValueError(f"Unexpected edits in OmniMemEval: {', '.join(sorted(unfamiliar))}")
    for relative, replacements in edits.items():
        path = checkout / relative
        upstream = _git(checkout, "show", f"HEAD:{relative}") + "\n"
        patched = upstream
        for original, replacement in replacements:
            if patched.count(original) != 1:
                raise ValueError(f"OmniMemEval integration hook changed: {relative}")
            patched = patched.replace(original, replacement)
        current = path.read_text(encoding="utf-8")
        if current.rstrip() == upstream.rstrip():
            path.write_text(patched, encoding="utf-8")
        elif current.rstrip() != patched.rstrip():
            raise ValueError(f"OmniMemEval checkout has edits outside the adapter hook: {relative}")
    bridge = checkout / "scripts/client_factory/trisynapse_client.py"
    content = "from trisynapse_memory.benchmarks.omni_backend import TrisynapseClient\n"
    if bridge.exists():
        if bridge.read_text(encoding="utf-8") != content:
            raise ValueError(f"OmniMemEval backend bridge has local edits: {bridge}")
    else:
        bridge.write_text(content, encoding="utf-8")


def _setup(checkout: Path, *, install: bool) -> None:
    if not (checkout / ".git").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        if any(checkout.iterdir()):
            raise ValueError(f"Use an empty directory for the pinned OmniMemEval checkout: {checkout}")
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
    )
    if revision.returncode:
        _git(checkout, "fetch", "--depth=1", UPSTREAM, UPSTREAM_REVISION)
        _git(checkout, "checkout", "--detach", "FETCH_HEAD")
    _install_backend(checkout)
    if install:
        _install_requirements(checkout / "requirements_user_memory.txt")
    print(f"OmniMemEval {UPSTREAM_REVISION} ready at {checkout}")


def _install_requirements(requirements: Path) -> None:
    if find_spec("pip") is not None:
        install = [sys.executable, "-m", "pip", "install"]
    elif uv := shutil.which("uv"):
        install = [uv, "pip", "install", "--python", sys.executable]
    else:
        raise RuntimeError("OmniMemEval setup requires pip or uv in the active environment.")
    subprocess.run([*install, "-r", str(requirements)], check=True)


def _link_input(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"{target} points to a different input; use a separate OmniMemEval checkout")
        return
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise


def _convert_beam(parquets: list[Path], target: Path) -> None:
    import pyarrow.parquet as pq

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as output:
        temp = Path(output.name)
        try:
            for parquet in parquets:
                for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1):
                    output.write(json.dumps(batch.to_pylist()[0], ensure_ascii=False) + "\n")
            output.close()
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


def _prepare(
    suite: str, checkout: Path, data: Path, *, download: bool = False
) -> list[Path]:
    directory = checkout / "data"
    if suite in BEAM_FILES:
        scale, pattern, filename = BEAM_FILES[suite]
        target = directory / "beam" / filename
        sources = sorted((data / suite / "data").glob(pattern))
        if sources and not target.exists():
            _convert_beam(sources, target)
        if not target.exists() and download:
            subprocess.run(
                [sys.executable, "data/beam/prepare_beam.py", "--scale", scale],
                cwd=checkout,
                check=True,
            )
        targets = [target]
    else:
        source_dir, target_dir, filenames = LOCAL_INPUTS[suite]
        targets = [directory / target_dir / filename for filename in filenames]
        for filename, target in zip(filenames, targets, strict=True):
            _link_input(data / source_dir / filename, target)
        if download and not all(target.exists() for target in targets):
            prepare_script = next((directory / target_dir).glob("prepare_*.py"))
            subprocess.run([sys.executable, str(prepare_script)], cwd=checkout, check=True)
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(f"Missing dataset input {target}; run prepare {suite} --download")
    return targets


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _dataset_inputs(targets: list[Path], checkout: Path) -> dict[str, str]:
    inputs = {}
    for target in targets:
        files = sorted(target.glob("*.json")) if target.is_dir() else [target]
        if not files:
            raise FileNotFoundError(f"No PersonaMem histories found in {target}")
        for path in files:
            inputs[path.relative_to(checkout).as_posix()] = _file_hash(path)
    return inputs


def _backend_files() -> dict[str, Path]:
    package = Path(__file__).resolve().parents[1]
    paths = [
        *package.rglob("*.py"),
        *(package / "prompts").glob("*.md"),
        *(package / "benchmarks" / "configs").glob("*.yaml"),
    ]
    return {path.relative_to(package.parent).as_posix(): path for path in sorted(paths)}


def _source_hash() -> str:
    digest = sha256()
    for name, path in _backend_files().items():
        digest.update(name.encode() + b"\0" + bytes.fromhex(_file_hash(path)))
    return digest.hexdigest()


def _save_source(path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with ZipFile(temporary, mode="w", compression=ZIP_DEFLATED) as bundle:
        for name, source in _backend_files().items():
            bundle.write(source, name)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_path(args: argparse.Namespace, data: Path) -> Path:
    _, result_dir, _ = SUITES[args.suite]
    return data / "omnimemeval" / "runs" / result_dir / f"{args.backend}-{args.name}"


def _safe_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("use 1-80 letters, digits, dots, hyphens or underscores")
    return value


def _model_name(value: str) -> str:
    if not _MODEL_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid model identifier")
    return value


def _runtime_env(args: argparse.Namespace, run_dir: Path, checkout: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    env = os.environ.copy()
    env_file = args.env or (Path(".env.local") if Path(".env.local").is_file() else None)
    if env_file is not None:
        if not env_file.is_file():
            raise FileNotFoundError(env_file)
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env.setdefault(key, value)
    for role in ("ANSWER", "EVAL"):
        env.setdefault(f"{role}_API_KEY", env.get("OPENAI_API_KEY", ""))
        env.setdefault(f"{role}_BASE_URL", OPENAI_ENDPOINT)
    nltk_cache = checkout / "resources" / "nltk"
    if nltk_cache.is_dir():
        env.setdefault("NLTK_DATA", str(nltk_cache))
    script_dir = str(Path(sys.executable).absolute().parent)
    source_dir = str(Path(__file__).resolve().parents[2])
    env.update({
        "PATH": script_dir + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": source_dir,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRISYNAPSE_OMNI_RUN_DIR": str(run_dir),
        "TRISYNAPSE_OMNI_RUN_VERSION": args.name,
        "TRISYNAPSE_OMNI_MEMORY_VERSION": args.reuse if args.from_step == 2 and args.reuse else args.name,
        "TRISYNAPSE_OMNI_MEMORY_MODEL": args.memory_model,
        "TRISYNAPSE_OMNI_EMBEDDING": args.embedding_model,
    })
    return env


def _check_credentials(args: argparse.Namespace, env: dict[str, str], start: int) -> None:
    needs_answer = start <= 3 <= args.to_step
    needs_memory = args.backend == "trisynapse" and start == 1 and args.memory_model != "none"
    if (needs_answer or needs_memory) and not env["ANSWER_API_KEY"]:
        raise ValueError("Set ANSWER_API_KEY or OPENAI_API_KEY in the runtime env")
    if args.suite != "personamem" and start <= 4 <= args.to_step and not env["EVAL_API_KEY"]:
        raise ValueError("Set EVAL_API_KEY or OPENAI_API_KEY in the runtime env")


def _command(args: argparse.Namespace, checkout: Path, env_file: Path, first: int, last: int) -> list[str]:
    runner, _, _ = SUITES[args.suite]
    command = [
        "bash", str(checkout / "scripts" / f"run_{runner}_eval.sh"),
        "--lib", args.backend, "--version", args.name, "--env", str(env_file),
        "--top-k", str(args.top_k), "--workers", str(args.workers), "--llm-workers", str(args.llm_workers),
        "--num-runs", str(args.num_runs), "--save-model-input", "1",
        "--from-step", str(first), "--to-step", str(last),
    ]
    if args.suite in BEAM_FILES:
        command.extend(["--scale", BEAM_FILES[args.suite][0]])
    if args.last_unit is not None:
        command.extend(["--streaming", "1", "--end-idx", str(args.last_unit)])
    return command


def _retrieval_settings(path: Path | None) -> dict:
    settings = (
        RetrievalConfiguration.model_validate_json(path.read_text(encoding="utf-8"))
        if path is not None else RetrievalConfiguration()
    )
    return settings.model_dump(mode="json", exclude={"revision", "updated_at"})


def _snapshot(args: argparse.Namespace, checkout: Path, inputs: list[Path], env: dict[str, str]) -> dict:
    return {
        "format_version": 1,
        "suite": args.suite,
        "backend": args.backend,
        "version": args.name,
        "reuse": args.reuse,
        "harness": {"repository": UPSTREAM, "revision": UPSTREAM_REVISION},
        "dataset_sha256": _dataset_inputs(inputs, checkout),
        "backend_source_sha256": _source_hash(),
        "backend_prompts": prompt_provenance(("facts", "group_summary")) if args.backend == "trisynapse" else [],
        "retrieval": _retrieval_settings(args.retrieval_config) if args.backend == "trisynapse" else None,
        "models": {
            "memory": args.memory_model if args.backend == "trisynapse" else None,
            "embedding": args.embedding_model if args.backend == "trisynapse" else None,
            "answer": args.answer_model,
            "judge": args.judge_model if args.suite != "personamem" else None,
        },
        "providers": {
            role.lower(): {
                "host": urlsplit(env[f"{role}_BASE_URL"]).hostname,
                "endpoint_sha256": sha256(env[f"{role}_BASE_URL"].encode()).hexdigest(),
            }
            for role in ("ANSWER", "EVAL")
        },
        "settings": {"top_k": args.top_k, "num_runs": args.num_runs, "last_unit": args.last_unit},
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in (
                "openai", "pandas", "torch", "sentence-transformers", "transformers",
                "lancedb", "tiktoken", "numpy", "nltk", "rouge-score", "bert-score",
            )
        },
    }


def _connect_results(checkout: Path, data: Path) -> None:
    destination = data / "omnimemeval" / "runs"
    destination.mkdir(parents=True, exist_ok=True)
    path = checkout / "results"
    _link_input(destination, path)
    if path.resolve() != destination:
        raise ValueError(f"OmniMemEval results {path} must point to {destination}")


def _validate_run(args: argparse.Namespace) -> int:
    _, _, last_step = SUITES[args.suite]
    args.to_step = last_step if args.to_step is None else args.to_step
    if not 1 <= args.from_step <= args.to_step <= last_step:
        raise ValueError(f"Stage bounds must be between 1 and {last_step}, in ascending order")
    if args.from_step > 1 and not (args.resume or args.reuse):
        raise ValueError("Use --resume or --reuse with --from-step greater than 1")
    if args.reuse and (args.suite != "locomo" or args.from_step not in {2, 3, 4} or args.reuse == args.name):
        raise ValueError("--reuse needs a different LoCoMo run name and --from-step 2, 3, or 4")
    if args.reuse and args.backend != "trisynapse" and args.from_step == 2:
        raise ValueError("Sharing an ingested memory snapshot requires the Trisynapse backend")
    if min(args.top_k, args.workers, args.llm_workers, args.num_runs) < 1:
        raise ValueError("Top-k, worker counts, and run counts must be positive")
    if args.last_unit is not None and (args.last_unit < 0 or args.suite == "locomo"):
        raise ValueError("--last-unit requires a non-negative index on a streaming suite")
    if args.last_unit is not None and args.to_step == 1:
        raise ValueError("Streaming runs combine ingestion and search; use --to-step 2 or later")
    if args.backend == "trisynapse" and args.memory_model == "none" and args.to_step > 2:
        raise ValueError("The offline memory model is limited to ingest and search smoke runs")
    if args.backend != "trisynapse" and args.retrieval_config is not None:
        raise ValueError("--retrieval-config is only supported by the Trisynapse backend")
    return last_step


def _run(args: argparse.Namespace, checkout: Path, data: Path) -> None:
    last_step = _validate_run(args)
    run_dir = _run_path(args, data)
    env = _runtime_env(args, run_dir, checkout)
    inputs = _prepare(args.suite, checkout, data)
    _connect_results(checkout, data)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _execute(args, checkout, run_dir, env, inputs, last_step)


def _reuse_locomo(args: argparse.Namespace, target: Path, configuration: dict) -> None:
    source = target.with_name(f"{args.backend}-{args.reuse}")
    if not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"No reusable run at {source}")
    with (source / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        original = manifest["configuration"]
        if original["harness"] != configuration["harness"] or original["dataset_sha256"] != configuration["dataset_sha256"]:
            raise ValueError("The source run uses a different dataset or harness revision")
        if original["backend_source_sha256"] != configuration["backend_source_sha256"]:
            raise ValueError("The source run uses different Trisynapse code")
        retained_models = ("memory", "embedding", "answer") if args.from_step == 4 else ("memory", "embedding")
        if any(original["models"][role] != configuration["models"][role] for role in retained_models):
            raise ValueError("The source run uses different models for stages being reused")
        if original["providers"]["answer"] != configuration["providers"]["answer"]:
            raise ValueError("The source run uses a different answer or ingestion endpoint")
        if args.from_step >= 3 and (
            original["retrieval"] != configuration["retrieval"]
            or original["settings"]["top_k"] != configuration["settings"]["top_k"]
        ):
            raise ValueError("Reusing search results requires identical retrieval settings and top-k")
        if manifest.get("completed_through", 0) < args.from_step - 1:
            raise ValueError(f"The source run has not finished stage {args.from_step - 1}")
        if args.from_step == 2:
            if original["reuse"] is not None or manifest["completed_through"] != 1:
                raise ValueError("Use a fresh ingest-only run as the shared memory snapshot")
            _copy_locomo_memory(source, target, configuration["retrieval"])
        _copy_locomo_outputs(source, target, args.backend, args.from_step)


def _copy_locomo_memory(source: Path, target: Path, settings: dict) -> None:
    retrieval = RetrievalConfiguration.model_validate(settings)
    with TemporaryDirectory(dir=target, prefix=".memory-") as staging:
        snapshot = Path(staging) / "memory"
        shutil.copytree(source / "memory", snapshot, ignore=shutil.ignore_patterns("*.jsonl"))
        stores = list(snapshot.glob("*/trace.sqlite3"))
        if not stores:
            raise ValueError(f"No ingested Trace stores in {source}")
        for database in stores:
            with closing(SQLiteMemoryLogStore(database.parent)) as store:
                current = store.get_retrieval_configuration()
                store.save_retrieval_configuration(retrieval, expected_revision=current.revision)
        snapshot.rename(target / "memory")


def _copy_locomo_outputs(source: Path, target: Path, backend: str, from_step: int) -> None:
    artifacts = [
        (1, "ingestion_stats.json", False),
        (2, "search_results.json", True),
        (2, "search_status.json", False),
        (3, "responses.json", True),
        (3, "response_status.json", False),
    ]
    for stage, suffix, required in artifacts:
        if stage >= from_step:
            continue
        name = f"{backend}_locomo_{suffix}"
        path = source / name
        if path.is_file() or required:
            shutil.copyfile(path, target / name)


def _completed_stages(run_dir: Path, completed: int, last_step: int) -> int:
    while completed < last_step and (run_dir / f".step_{completed + 1}_done").is_file():
        completed += 1
    return completed


def _stage_ranges(first: int, last: int, *, streaming: bool):
    if streaming and first == 1:
        yield 1, 2
        first = 3
    for stage in range(first, last + 1):
        yield stage, stage


def _run_stage(command: list[str], checkout: Path, env: dict, log) -> int:
    with subprocess.Popen(
        command, cwd=checkout, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _execute(args: argparse.Namespace, checkout: Path, run_dir: Path, env: dict, inputs: list[Path], last_step: int) -> None:
    manifest_path = run_dir / "manifest.json"
    settings = _snapshot(args, checkout, inputs, env)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError(f"Run already exists at {run_dir}; use a new --name or resume an unfinished run")
        if manifest["configuration"] != settings:
            raise ValueError("Run configuration or inputs changed. Use a new --name.")
        completed = _completed_stages(run_dir, manifest.get("completed_through", 0), last_step)
        manifest["completed_through"] = completed
        if args.from_step > completed + 1:
            raise ValueError("Complete the preceding stages before resuming")
    else:
        if args.resume:
            raise FileNotFoundError(f"No run manifest at {manifest_path}")
        manifest = {"created_at": _now(), "configuration": settings, "attempts": [], "completed_through": 0}
        if args.reuse:
            _reuse_locomo(args, run_dir, settings)
            manifest["completed_through"] = args.from_step - 1
        _save_source(run_dir / "trisynapse-source.zip")
    start = max(args.from_step, manifest["completed_through"] + 1)
    if start > args.to_step:
        manifest["status"] = "complete" if manifest["completed_through"] == last_step else "partial"
        _write_json(manifest_path, manifest)
        print(f"Requested stages already completed: {run_dir}")
        return
    if settings["retrieval"] is not None:
        _write_json(run_dir / "retrieval-config.json", settings["retrieval"])
    env_file = run_dir / "harness.env"
    env_file.write_text(f"ANSWER_MODEL={args.answer_model}\nEVAL_MODEL={args.judge_model}\n", encoding="utf-8")
    attempt = {"started_at": _now(), "exit_code": None, "stages": []}
    manifest["attempts"].append(attempt)
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)
    try:
        _check_credentials(args, env, start)
        with (run_dir / "run.log").open("a", encoding="utf-8") as log:
            for first, last in _stage_ranges(start, args.to_step, streaming=args.last_unit is not None):
                command = _command(args, checkout, env_file, first, last)
                stage = {"from": first, "through": last, "started_at": _now(), "command": command}
                attempt["stages"].append(stage)
                _write_json(manifest_path, manifest)
                stage["exit_code"] = _run_stage(command, checkout, env, log)
                stage["ended_at"] = _now()
                attempt["exit_code"] = stage["exit_code"]
                if stage["exit_code"]:
                    raise RuntimeError(f"OmniMemEval stage {first} failed. See {run_dir / 'run.log'}")
                manifest["completed_through"] = last
                _write_json(manifest_path, manifest)
        manifest["status"] = "complete" if args.to_step == last_step else "partial"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        attempt["ended_at"] = _now()
        manifest["artifacts"] = sorted(
            path.name for path in run_dir.iterdir() if not path.name.startswith(".") and path != manifest_path
        )
        _write_json(manifest_path, manifest)
    print(f"Run artifacts: {run_dir}")


def _plan_command(plan_path: Path, defaults: dict, run: dict, overrides: dict) -> list[str]:
    options = {**defaults, **run.get("options", {}), **overrides}
    if config := options.get("retrieval_config"):
        config_path = (plan_path.parent / config).resolve()
        _retrieval_settings(config_path)
        cwd = Path.cwd()
        options["retrieval_config"] = str(config_path.relative_to(cwd) if config_path.is_relative_to(cwd) else config_path)
    command = ["run", run["suite"], "--name", run["name"]]
    for key, value in options.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def _print_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    plan_path = args.file.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shown = 0
    for run in plan["runs"]:
        if args.name and run["name"] != args.name:
            continue
        command = _plan_command(plan_path, plan["defaults"], run, {})
        _validate_run(parser.parse_args(command))
        print(shlex.join(["trisynapse-memory", "bench", "omni", *command]))
        shown += 1
    if not shown:
        raise ValueError(f"No matching runs in {plan_path}")


def _queue_plan(parser: argparse.ArgumentParser, args: argparse.Namespace, checkout: Path, data: Path) -> int:
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    plan_path = args.file.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in plan["runs"]}
    if len(runs) != len(plan["runs"]):
        raise ValueError(f"Run names must be unique in {plan_path}")
    selected = list(args.name or [])
    groups = args.group or []
    for group in groups:
        group_runs = plan.get("groups", {}).get(group)
        if not group_runs:
            raise ValueError(f"No matching runs for group {group!r} in {plan_path}")
        selected.extend(group_runs)
    overrides = {
        key: getattr(args, key) for key in ("workers", "llm_workers") if getattr(args, key) is not None
    }
    ordered: dict[str, QueuedRun] = {}
    visiting: set[str] = set()

    def add(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise ValueError(f"Circular run dependency at {name}")
        if name not in runs:
            raise ValueError(f"Unknown run {name!r} in {plan_path}")
        visiting.add(name)
        command = _plan_command(plan_path, plan["defaults"], runs[name], overrides)
        run = parser.parse_args(command)
        _validate_run(run)
        if run.reuse:
            add(run.reuse)
            source = parser.parse_args(ordered[run.reuse].argv)
            _, _, end = SUITES[source.suite]
            if (
                (source.suite, source.backend) != (run.suite, run.backend)
                or (source.to_step or end) < run.from_step - 1
                or (run.from_step == 2 and source.to_step != 1)
            ):
                raise ValueError(f"{run.reuse} cannot supply the stages requested by {name}")
        ordered[name] = QueuedRun(name, command, _run_path(run, data), run.reuse)
        visiting.remove(name)

    for name in selected:
        add(name)
    if args.dry_run:
        print(f"Queue: {len(ordered)} runs, up to {args.jobs} at a time")
        for run in ordered.values():
            command = ["trisynapse-memory", "bench", "omni", "--checkout", str(checkout), "--data-root", str(data)]
            resume = ["--resume"] if (run.directory / "manifest.json").is_file() else []
            suffix = f"  # after {run.dependency}" if run.dependency else ""
            print(shlex.join([*command, *run.argv, *resume]) + suffix)
        return 0
    name = args.queue_name or (groups[0] if len(groups) == 1 else "selected")
    return run_queue(list(ordered.values()), checkout=checkout, data=data, jobs=args.jobs, name=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trisynapse-memory bench omni", description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("var/omnimemeval"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="action", required=True)
    setup = commands.add_parser("setup", help="Prepare the pinned upstream harness")
    setup.add_argument("--install", action="store_true", help="Install upstream evaluation dependencies")
    prepare = commands.add_parser("prepare", help="Stage local input or download the upstream dataset")
    prepare.add_argument("suite", choices=SUITES)
    prepare.add_argument("--download", action="store_true")
    plan = commands.add_parser("plan", help="Validate and print run commands without executing them")
    plan.add_argument("file", type=Path, nargs="?", default=Path("configs/omnimemeval/rebuttal.json"))
    plan.add_argument("--name", type=_safe_name, help="Print one named run")
    queue = commands.add_parser("queue", help="Run named experiments in parallel, resuming existing progress")
    queue.add_argument("file", type=Path, nargs="?", default=Path("configs/omnimemeval/rebuttal.json"))
    selection = queue.add_mutually_exclusive_group(required=True)
    selection.add_argument("--group", action="append", type=_safe_name, help="Run this plan group; repeat to select more")
    selection.add_argument("--name", action="append", type=_safe_name, help="Run this experiment; repeat to select more")
    queue.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous experiments")
    queue.add_argument("--workers", type=int, help="Override CPU workers per experiment")
    queue.add_argument("--llm-workers", type=int, help="Override LLM workers per experiment")
    queue.add_argument("--queue-name", type=_safe_name, help="Name for the queue status and logs")
    queue.add_argument("--dry-run", action="store_true", help="Print the queue without executing it")
    for action in ("run", "inspect"):
        descriptions = {"run": "Run OmniMemEval and save provenance", "inspect": "Read a run manifest"}
        command = commands.add_parser(action, help=descriptions[action])
        command.add_argument("suite", choices=SUITES)
        command.add_argument("--backend", type=_safe_name, default="trisynapse")
        command.add_argument("--name", type=_safe_name, required=action == "inspect")
        if action == "run":
            command.add_argument("--env", type=Path, help="Credential env file, default .env.local if present")
            command.add_argument("--memory-model", type=_model_name, default=DEFAULT_MEMORY_MODEL)
            command.add_argument("--embedding-model", type=_model_name, default=DEFAULT_EMBEDDING_MODEL)
            command.add_argument("--retrieval-config", type=Path, help="Trisynapse retrieval settings JSON")
            command.add_argument("--answer-model", type=_model_name, default=DEFAULT_MEMORY_MODEL)
            command.add_argument("--judge-model", type=_model_name, default=DEFAULT_JUDGE_MODEL)
            command.add_argument("--top-k", type=int, default=20)
            command.add_argument("--workers", type=int, default=2)
            command.add_argument("--llm-workers", type=int, default=10)
            command.add_argument("--num-runs", type=int, default=1)
            command.add_argument("--from-step", type=int, default=1)
            command.add_argument("--to-step", type=int)
            command.add_argument("--last-unit", type=int, help="Last zero-based unit in streaming mode")
            command.add_argument("--reuse", type=_safe_name, help="Use stages from an existing LoCoMo run")
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    checkout, data = args.checkout.resolve(), args.data_root.resolve()
    try:
        if os.name == "nt" and args.action not in {"inspect", "plan"} and not getattr(args, "dry_run", False):
            raise RuntimeError("OmniMemEval execution requires macOS, Linux, or WSL. Use the Linux environment in WSL.")
        if args.action == "setup":
            _setup(checkout, install=args.install)
        elif args.action == "plan":
            _print_plan(parser, args)
        elif args.action == "queue":
            return _queue_plan(parser, args, checkout, data)
        elif args.action == "inspect":
            print((_run_path(args, data) / "manifest.json").read_text(encoding="utf-8"), end="")
        else:
            _install_backend(checkout)
            if args.action == "prepare":
                for path in _prepare(args.suite, checkout, data, download=args.download):
                    print(path)
            else:
                if args.resume and not args.name:
                    raise ValueError("--resume requires --name")
                args.name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                _run(args, checkout, data)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
