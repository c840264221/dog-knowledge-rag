"""多 Agent 主图适配器统一导入入口。"""

from src.agents.collaboration.adapters.resume_input_adapter import (
    MultiAgentResumeAction,
    resolve_multi_agent_resume_input,
)

__all__ = [
    "MultiAgentResumeAction",
    "resolve_multi_agent_resume_input",
]
