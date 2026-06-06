from src.runtime.persistence.runtime_snapshot import (
    RuntimeSnapshot
)

from src.runtime.context import (
    RuntimeContext
)

from src.runtime.scopes.metrics_scope import (
    MetricsScope
)


class RuntimeSerializer:

    @staticmethod
    def to_snapshot(ctx: RuntimeContext) -> RuntimeSnapshot:

        # 实例化时间线作用域
        timeline_scope = ctx.timeline()

        return RuntimeSnapshot(

            trace_id=ctx.trace_id,

            user_id=ctx.user_id,

            session_id=ctx.session_id,

            component=ctx.component,

            runtime_state=ctx.state().export(),

            metadata=ctx.metadata,

            metrics=ctx.service(MetricsScope).get_metrics(),

            timeline=timeline_scope.get_events()
        )

    @staticmethod
    def from_snapshot(snapshot: RuntimeSnapshot):
        ctx = RuntimeContext()

        ctx.trace_id = snapshot.trace_id

        ctx.user_id = snapshot.user_id

        ctx.session_id = snapshot.session_id

        ctx.component = snapshot.component

        ctx.metadata = snapshot.metadata

        state_scope = ctx.state()

        # state_scope.state = snapshot.runtime_state

        state_scope.restore(
            snapshot.runtime_state
        )

        metrics_scope = ctx.service(
            MetricsScope
        )

        metrics_scope.restore(
            snapshot.metrics
        )

        # 时间线
        timeline_scope = ctx.timeline()
        timeline_scope.restore(
            snapshot.timeline
        )

        return ctx