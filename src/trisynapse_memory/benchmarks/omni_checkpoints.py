"""Durable per-answer checkpoints for OmniMemEval's LoCoMo judge."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any


class LoCoMoJudgeCheckpoint:
    """Keep completed grades when another question in the same group fails."""

    def __init__(self, group_id: str, model: str, repeats: int) -> None:
        root = os.environ.get("TRISYNAPSE_OMNI_RUN_DIR")
        identity = json.dumps([group_id, model, repeats])
        self.path = (
            Path(root) / "judge-checkpoints" / sha256(identity.encode()).hexdigest()[:24]
            if root else None
        )
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _response_key(response: dict[str, Any]) -> str:
        return sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()

    def load(self, index: int, response: dict[str, Any]) -> dict[str, Any] | None:
        if self.path is None:
            return None
        path = self.path / f"{index}.json"
        if not path.is_file():
            return None
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint["response"] != self._response_key(response):
            return None
        return checkpoint["grade"]

    def save(self, index: int, response: dict[str, Any], grade: dict[str, Any]) -> None:
        if self.path is None:
            return
        path = self.path / f"{index}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"response": self._response_key(response), "grade": grade}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
