"""DolphinBench integration for the Trisynapse Native harness.

This package intentionally does not register DolphinBench with the shared QA
benchmark runner.  DolphinBench owns its own sequential interaction, MCP, and
grading protocol; :class:`TrisynapseDolphinBenchHarness` implements that public
adapter boundary directly.
"""

from .harness import TrisynapseDolphinBenchHarness

__all__ = ["TrisynapseDolphinBenchHarness"]
