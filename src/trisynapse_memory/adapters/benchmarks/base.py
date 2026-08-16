"""Dataset adapter interface for production memory benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A normalized question produced by a dataset adapter."""

    id: str
    question: str
    gold: str
    evidence: Any = None
    evidence_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """One isolated memory world containing one or more questions."""

    id: str
    payload: dict[str, Any]
    questions: tuple[BenchmarkQuestion, ...]


@dataclass(frozen=True)
class PreparedCase:
    """The engine-facing context returned after adapter ingestion."""

    episode_ids: tuple[str, ...]
    namespace: MemoryNamespace
    episode_prefix: str | None = None


class BenchmarkAdapter(ABC):
    """Translate one external dataset into the shared benchmark lifecycle.

    Adapters own file/schema interpretation and evidence semantics. They may
    ingest through public ``MemoryEngine`` methods, but they never control
    retrieval, extraction, judging, provider selection, or artifact writing.
    """

    name: ClassVar[str]
    default_filename: ClassVar[str]

    def resolve_dataset(self, data_root: str | Path) -> Path:
        path = Path(data_root)
        if path.is_dir():
            path = path / self.default_filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @abstractmethod
    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        """Parse source data into isolated, normalized cases."""

    @abstractmethod
    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        """Ingest a case through the shipped engine and return query scope."""

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
    ) -> dict[str, Any]:
        """Return dataset-specific evidence metrics for one answer."""

        return dict(question.metadata)
