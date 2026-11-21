"""Run the pinned DolphinBench harness with an OpenAI-key semantic grader.

The published runner requires Azure transport for its fixed GPT-5.6 Sol judge.
This benchmark-local entry point retains its prompts, model, reasoning effort,
recording, and submission validation while using OpenAI's Chat Completions API.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


JUDGE_MODEL = "gpt-5.6-sol"
JUDGE_REASONING_EFFORT = "medium"
JUDGE_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def _fixed_grader(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grader = config.get("options", {}).get("grader", {})
    expected = {
        "provider": "openai",
        "model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "endpoint": JUDGE_ENDPOINT,
        "api_key_env": "OPENAI_API_KEY",
    }
    if grader != expected:
        raise ValueError("DolphinBench grader configuration must match the fixed OpenAI judge")


def _call_openai_judge(
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
    timeout: int = 90,
) -> dict[str, Any]:
    from graders import llm_judge
    from graders.judge_recording import record_response

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if config.get("backend", "openai") != "openai":
        raise ValueError("DolphinBench OpenAI-key runner requires the OpenAI judge backend")
    if config.get("judge_model", JUDGE_MODEL) != JUDGE_MODEL:
        raise ValueError("DolphinBench requires GPT-5.6 Sol as its fixed judge")
    if config.get("reasoning_effort", JUDGE_REASONING_EFFORT) != JUDGE_REASONING_EFFORT:
        raise ValueError("DolphinBench requires medium grader reasoning effort")
    settings = {
        "model": JUDGE_MODEL,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
        "max_completion_tokens": llm_judge.DEFAULT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request = urllib.request.Request(
        JUDGE_ENDPOINT,
        data=json.dumps({**settings, "messages": messages}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    response = llm_judge._urlopen_json(request, timeout, model=JUDGE_MODEL)
    record_response(settings, messages, response, (time.monotonic() - started) * 1000)
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""
    if not content.strip():
        raise RuntimeError(
            f"OpenAI judge returned empty content (finish_reason={choice.get('finish_reason')})"
        )
    return llm_judge._extract_json(content)


def main(argv: list[str] | None = None) -> int:
    """Use the upstream CLI and dataset protocol with OpenAI judge transport."""

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("stage", nargs="?")
    parser.add_argument("--config", type=Path, default=Path("run.yaml"))
    args, _ = parser.parse_known_args(argv)
    if args.stage != "init":
        _fixed_grader(args.config)

    from graders import llm_judge
    from harness import runner as upstream

    class OpenAIKeyRunner(upstream.Runner):
        def __init__(
            self,
            release: dict[str, Any],
            directory: Path,
            adapter: Any,
            *,
            identity: dict[str, Any],
            release_root: Path = upstream.ROOT,
            allow_paid: bool = False,
        ) -> None:
            # Upstream's paid-call guard hardcodes Azure transport. Its other
            # execution and submission checks remain active.
            super().__init__(
                release,
                directory,
                adapter,
                identity=identity,
                release_root=release_root,
                allow_paid=False,
            )
            self.allow_paid = allow_paid

    original_runner = upstream.Runner
    original_backend = llm_judge.BACKEND
    original_model = llm_judge.OPENAI_DEFAULT_MODEL
    original_call = llm_judge._call_openai
    upstream.Runner = OpenAIKeyRunner
    llm_judge.BACKEND = "openai"
    llm_judge.OPENAI_DEFAULT_MODEL = JUDGE_MODEL
    llm_judge._call_openai = _call_openai_judge
    try:
        return upstream.main(argv)
    finally:
        upstream.Runner = original_runner
        llm_judge.BACKEND = original_backend
        llm_judge.OPENAI_DEFAULT_MODEL = original_model
        llm_judge._call_openai = original_call


if __name__ == "__main__":
    raise SystemExit(main())
