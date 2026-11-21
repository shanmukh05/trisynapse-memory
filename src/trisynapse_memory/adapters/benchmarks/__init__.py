"""Registry of benchmark dataset adapters.

Each adapter turns one public dataset into ordinary memory-log sources and
questions. Adapters must not change production retrieval or write gold answers
into searchable memory. Add a suite by implementing ``BenchmarkAdapter`` and
registering it in ``_ADAPTERS``.
"""

from __future__ import annotations

from trisynapse_memory.adapters.benchmarks.base import BenchmarkAdapter
from trisynapse_memory.adapters.benchmarks.beam import Beam10MAdapter, Beam100KAdapter
from trisynapse_memory.adapters.benchmarks.halumem import HaluMemAdapter
from trisynapse_memory.adapters.benchmarks.locomo import LoCoMoAdapter
from trisynapse_memory.adapters.benchmarks.longmemeval import LongMemEvalAdapter
from trisynapse_memory.adapters.benchmarks.memorydoc import MemoryDocAdapter
from trisynapse_memory.adapters.benchmarks.personamem import PersonaMemAdapter

_ADAPTERS: dict[str, BenchmarkAdapter] = {
    adapter.name: adapter
    for adapter in (
        LoCoMoAdapter(),
        LongMemEvalAdapter(),
        HaluMemAdapter(),
        MemoryDocAdapter(),
        PersonaMemAdapter(),
        Beam100KAdapter(),
        Beam10MAdapter(),
    )
}


def get_adapter(name: str) -> BenchmarkAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unsupported benchmark suite '{name}'; choose one of: {supported}") from exc


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


__all__ = ["BenchmarkAdapter", "adapter_names", "get_adapter"]
