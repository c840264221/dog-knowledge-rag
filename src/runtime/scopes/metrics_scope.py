from src.runtime.scopes.base_scope import (
    BaseScope
)

from src.runtime.context.request_scope import (
    RequestScope
)


class MetricsScope(BaseScope):

    KEY = "runtime_metrics"

    def __init__(self,request_scope: RequestScope):
        self.scope = request_scope

    def init_metrics(self):

        self.scope.set(

            self.KEY,

            {
                "tool_count": 0,

                "llm_count": 0,

                "error_count": 0,

                "tool_latency": 0,

                "llm_latency": 0,
            }
        )

    def get_metrics(self):

        return self.scope.get(
            self.KEY,
            {}
        )

    def update(self, key, value):

        metrics = self.get_metrics()

        metrics[key] = value

        self.scope.set(
            self.KEY,
            metrics
        )

    def increment(self, key, amount=1):

        metrics = self.get_metrics()

        metrics[key] = metrics.get(
            key,
            0
        ) + amount

        self.scope.set(
            self.KEY,
            metrics
        )

    async def startup(self):

        self.init_metrics()

    async def shutdown(self):

        self.scope.remove(
            self.KEY
        )