from dataclasses import dataclass, field


@dataclass
class RuntimeState:

    current_agent: str | None = None

    current_node: str | None = None

    current_tool: str | None = None

    phase: str | None = None

    retry_count: int = 0


    # 因为单独做了timeline scope 所以运行历史部分废弃 改为用更全面的timeline scope
    # 为了保证兼容性  所以此字段暂不废弃 作为占位用
    execution_history: list = field(
        default_factory=list
    )