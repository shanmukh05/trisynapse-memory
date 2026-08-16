"""Package-wide external integration and dataset-adapter boundary.

Live integrations are exported here; benchmark adapters are available from
``trisynapse_memory.adapters.benchmarks`` to avoid eagerly loading their registry.
"""

from trisynapse_memory.adapters.agent_events import AgentEvent, capture_agent_event
from trisynapse_memory.adapters.trisynapse_live import open_vault_engine

__all__ = ["AgentEvent", "capture_agent_event", "open_vault_engine"]
