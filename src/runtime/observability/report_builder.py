from src.runtime.observability.runtime_report import (
    RuntimeReport
)

from src.runtime.scopes.metrics_scope import (
    MetricsScope
)

from src.runtime.scopes.timeline_scope import (
    TimelineScope
)


class ReportBuilder:

    @staticmethod
    def build(ctx):

        metrics = ctx.service(
            MetricsScope
        )

        timeline = ctx.service(
            TimelineScope
        )

        state = ctx.state()

        return RuntimeReport(

            trace_id=ctx.trace_id,

            current_agent=state.get_agent(),

            node_path=[],

            tool_count=metrics.get_metrics().get(
                "tool_count",
                0
            ),

            error_count=metrics.get_metrics().get(
                "error_count",
                0
            ),

            tool_latency=metrics.get_metrics().get(
                "tool_latency",
                0
            ),

            timeline=timeline.get_events()
        )