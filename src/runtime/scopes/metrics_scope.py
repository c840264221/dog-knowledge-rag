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

                "llm_input_tokens": 0,

                "llm_output_tokens": 0,

                "llm_total_tokens": 0,

                # 保存每一次逻辑 LLM 调用的结构化明细，供成本和重复调用审计。
                "llm_calls": [],
            }
        )

    def append_llm_call(self, call_record: dict) -> None:
        """
        向当前请求追加一条 LLM 调用明细。

        功能：
            复制现有列表后追加记录，避免调用方直接修改 RequestScope 中保存
            的列表对象。旧检查点没有 llm_calls 字段时会自动从空列表开始。

        参数含义：
            call_record：已经转换成普通字典的标准 LLM 调用记录。

        返回值含义：
            None：只更新当前请求的指标数据。
        """

        metrics = self.get_metrics()
        raw_calls = metrics.get("llm_calls", [])
        calls = list(raw_calls) if isinstance(raw_calls, list) else []
        calls.append(dict(call_record))
        metrics["llm_calls"] = calls
        self.scope.set(self.KEY, metrics)

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

    def restore(self, data):
        self.scope.set(
            self.KEY,
            data
        )

    async def startup(self):
        metrics = self.get_metrics()

        if metrics:
            return

        self.init_metrics()

    async def shutdown(self):

        self.scope.remove(
            self.KEY
        )
