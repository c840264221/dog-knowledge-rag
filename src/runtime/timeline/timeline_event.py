from dataclasses import dataclass

from datetime import datetime


@dataclass
class TimelineEvent:

    event_type: str

    name: str

    timestamp: str

    metadata: dict | None = None

    @classmethod
    def create(cls,event_type,name,metadata=None):
        return cls(
            event_type=event_type,

            name=name,

            timestamp=datetime.now().isoformat(),

            metadata=metadata or {}
        )