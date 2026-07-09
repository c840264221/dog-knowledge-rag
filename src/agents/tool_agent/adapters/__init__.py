"""
ToolAgent 适配器子包。

功能：
    收拢旧 state 到新 ToolAgent 契约的适配逻辑。
"""

from src.agents.tool_agent.adapters.registry_adapter import (
    TOOL_AGENT_TOOL_CATALOG_STATE_KEY,
    build_tool_agent_tool_catalog,
    build_tool_agent_tool_catalog_state_update,
    dump_tool_metadata_for_agent,
    get_registered_tool_metadata,
    list_registered_tool_metadata,
    tool_requires_confirmation,
)
from src.agents.tool_agent.adapters.runtime_adapter import (
    TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY,
    build_failed_tool_result,
    build_runtime_call_id,
    build_tool_agent_runtime_state_update,
    dump_tool_agent_execution_records_for_state,
    execute_tool_call_with_runtime,
    execute_tool_calls_with_runtime,
)
from src.agents.tool_agent.adapters.state_adapter import (
    TOOL_AGENT_RESPONSE_STATE_KEY,
    build_tool_agent_response_from_state,
    build_tool_agent_response_state_update,
    dump_tool_agent_response_for_state,
)

__all__ = [
    "TOOL_AGENT_RESPONSE_STATE_KEY",
    "TOOL_AGENT_RUNTIME_RECORDS_STATE_KEY",
    "TOOL_AGENT_TOOL_CATALOG_STATE_KEY",
    "build_failed_tool_result",
    "build_runtime_call_id",
    "build_tool_agent_response_from_state",
    "build_tool_agent_response_state_update",
    "build_tool_agent_runtime_state_update",
    "build_tool_agent_tool_catalog",
    "build_tool_agent_tool_catalog_state_update",
    "dump_tool_agent_execution_records_for_state",
    "dump_tool_agent_response_for_state",
    "dump_tool_metadata_for_agent",
    "execute_tool_call_with_runtime",
    "execute_tool_calls_with_runtime",
    "get_registered_tool_metadata",
    "list_registered_tool_metadata",
    "tool_requires_confirmation",
]
