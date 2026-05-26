from collections import defaultdict


class EventBus:

    def __init__(self):

        self.listeners = defaultdict(list)

    def subscribe(self,event_type,listener):

        self.listeners[event_type].append(listener)

    async def emit(self, event):

        event_type = type(event)

        listeners = self.listeners.get(event_type,[])

        for listener in listeners:

            await listener.handle(event)

event_bus = EventBus()