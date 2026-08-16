"""Registry of benchmark-specific dataset adapters."""

from __future__ import annotations

from trisynapse_memory.adapters.benchmarks.base import BenchmarkAdapter
from trisynapse_memory.adapters.benchmarks.halumem import HaluMemAdapter
from trisynapse_memory.adapters.benchmarks.locomo import LoCoMoAdapter
from trisynapse_memory.adapters.benchmarks.longmemeval import LongMemEvalAdapter
from trisynapse_memory.adapters.benchmarks.memorydoc import MemoryDocAdapter

_ADAPTERS: dict[str, BenchmarkAdapter] = {
    adapter.name: adapter
    for adapter in (LoCoMoAdapter(), LongMemEvalAdapter(), HaluMemAdapter(), MemoryDocAdapter())
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
