from src.runtime.context import (
    runtime_ctx
)

from src.runtime.scopes.timeline_scope import (
    TimelineScope
)

from src.logger import logger


class TimelineReporter:

    @staticmethod
    def report():

        timeline = runtime_ctx.get().service(
            TimelineScope
        )
        events = timeline.get_events()

        logger.info(
            "========== Runtime Timeline =========="
        )

        for i, event in enumerate(events,start=1):
            logger.info(
                f"[{i}] "
                f"{event.timestamp} | "
                f"{event.event_type} | "
                f"{event.name} | "
                # f"{event['message']}"
            )

        logger.info(
            "======================================"
        )