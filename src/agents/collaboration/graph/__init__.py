"""多 Agent 主图入口统一导入。"""

from src.agents.collaboration.graph.entry_node import (
    MultiAgentEntryNode,
    build_multi_agent_entry_node,
    build_multi_agent_state_update,
)

__all__ = [
    "MultiAgentEntryNode",
    "build_multi_agent_entry_node",
    "build_multi_agent_state_update",
]
