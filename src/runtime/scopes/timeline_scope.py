from src.runtime.scopes.base_scope import (
    BaseScope
)

from src.runtime.timeline.timeline_event import (
    TimelineEvent
)


class TimelineScope(BaseScope):

    def __init__(self):

        self.events = []

    def add_event(self,event_type,name,metadata=None):

        event = TimelineEvent.create(
            event_type,
            name,
            metadata
        )

        self.events.append(event)

    def get_events(self):

        return self.events

    def restore(self, events):
        self.events = events

    def clear(self):

        self.events.clear()

    async def startup(self):
        pass

    async def shutdown(self):
        self.clear()