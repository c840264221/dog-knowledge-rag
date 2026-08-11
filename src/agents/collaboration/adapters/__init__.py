"""多 Agent 主图适配器统一导入入口。"""

from src.agents.collaboration.adapters.clarification_field_resolver import (
    MultiAgentClarificationFieldResolver,
    allocate_fields_to_steps,
    build_default_multi_agent_clarification_field_resolver,
)
from src.agents.collaboration.adapters.resume_input_adapter import (
    MultiAgentResumeAction,
    resolve_multi_agent_resume_input,
)

__all__ = [
    "MultiAgentClarificationFieldResolver",
    "MultiAgentResumeAction",
    "allocate_fields_to_steps",
    "build_default_multi_agent_clarification_field_resolver",
    "resolve_multi_agent_resume_input",
]
