from dataclasses import dataclass


@dataclass
class RuntimeReport:

    trace_id: str

    current_agent: str

    node_path: list

    tool_count: int

    error_count: int

    tool_latency: float

    timeline: list