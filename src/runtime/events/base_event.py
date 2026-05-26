import time
import uuid


class BaseEvent:

    def __init__(self):

        self.event_id = str(
            uuid.uuid4()
        )

        self.timestamp = time.time()