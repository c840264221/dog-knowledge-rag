from dataclasses import dataclass
from src.runtime.state.runtime_state import RuntimeState


@dataclass
class RuntimeSnapshot:

    trace_id: str | None

    user_id: str | None

    session_id: str | None

    component: str | None

    runtime_state: RuntimeState

    metrics: dict

    timeline: list

    metadata: dict | None

