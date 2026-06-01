from dataclasses import dataclass, field


@dataclass
class RuntimeState:

    current_agent: str | None = None

    current_node: str | None = None

    current_tool: str | None = None

    phase: str | None = None

    execution_history: list = field(
        default_factory=list
    )