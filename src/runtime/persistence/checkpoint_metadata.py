from dataclasses import dataclass
from datetime import datetime


@dataclass
class CheckpointMetadata:

    trace_id: str

    created_at: datetime

    updated_at: datetime

    status: str

    interrupt_count: int = 0