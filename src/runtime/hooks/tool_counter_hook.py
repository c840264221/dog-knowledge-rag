from src.runtime.hooks.base_hook import (
    BaseHook
)

from src.runtime.context import runtime_ctx

from src.runtime.scopes.metrics_scope import (
    MetricsScope
)

from src.logger import logger


class ToolCounterHook(BaseHook):

    METRIC_KEY = "tool_before_hook_count"

    async def execute(

        self,

        *args,

        **kwargs
    ):
        """
        统计 tool.before hook 的触发次数。

        功能：
            从当前 RuntimeContext 中获取 MetricsScope（指标作用域），
            并递增 tool_before_hook_count 字段。
            这个 hook 主要用于学习 RuntimeHook（运行时钩子）的使用方式，
            不直接修改 tool_count，避免和 MetricsListener 的工具成功统计重复。

        参数：
            *args：
                hook 触发时传入的位置参数。当前实现暂不使用。
            **kwargs：
                hook 触发时传入的关键字参数。当前实现暂不使用。

        返回值：
            None：
                无业务返回值，只更新 MetricsScope 中的计数字段。
        """

        ctx = runtime_ctx.get()

        metrics_scope = ctx.service(
            MetricsScope
        )

        metrics_scope.increment(
            self.METRIC_KEY
        )

        count = metrics_scope.get_metrics().get(
            self.METRIC_KEY,
            0,
        )

        logger.info(
            f"[Hook] "
            f"tool.before 钩子已触发 "
            f"{count} 次"
        )
