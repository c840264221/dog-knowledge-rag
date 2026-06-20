from collections import defaultdict

import inspect
import logging


class RuntimeHookManager:

    def __init__(
        self,
        raise_hook_errors: bool = False,
    ):
        """
        初始化运行时钩子管理器。

        功能：
            创建 hooks 注册表，用于按照 hook_name 保存多个 hook。
            支持配置 hook 执行失败时是否向外抛出异常。

        参数：
            raise_hook_errors：
                是否在 hook 报错时继续向外抛出异常。
                False 表示记录错误并继续执行后续 hook。
                True 表示 hook 报错后立即抛出异常。

        返回值：
            None：构造函数无返回值。

        专业名词：
            RuntimeHookManager（运行时钩子管理器）：
                用于管理系统运行过程中的 hook 扩展点。

            hook（钩子）：
                在某个运行时机被触发的扩展逻辑，例如 before_node、after_node、on_error。

            fault isolation（故障隔离）：
                某个 hook 失败时，不影响其他 hook 继续执行。

            strict mode（严格模式）：
                hook 报错时不吞掉异常，而是直接抛给调用方。
        """

        self._hooks = defaultdict(list)
        self.raise_hook_errors = raise_hook_errors

    def register(self, hook_name: str, hook):
        """
        注册运行时钩子。

        功能：
            将 hook 注册到指定 hook_name 下。
            当 emit 触发该 hook_name 时，会依次执行该 hook_name 下的所有 hook。

        参数：
            hook_name：
                钩子名称，字符串格式，例如 before_node、after_node、on_error。

            hook：
                钩子对象，需要实现 execute(*args, **kwargs) 方法。
                execute 可以是同步函数，也可以是异步函数。

        返回值：
            None：无业务返回值，只修改内部 hooks 注册表。
        """

        self._hooks[hook_name].append(
            hook
        )

    async def emit(self, hook_name: str, *args, **kwargs):
        """
        触发指定名称的运行时钩子。

        功能：
            根据 hook_name 找到已注册的 hooks。
            依次调用每个 hook 的 execute 方法。
            同时兼容同步 execute 和异步 execute。
            默认隔离单个 hook 的异常，避免影响后续 hook。

        参数：
            hook_name：
                钩子名称，字符串格式，例如 before_node、after_node、on_error。

            *args：
                传递给 hook.execute 的位置参数。

            **kwargs：
                传递给 hook.execute 的关键字参数。

        返回值：
            None：无业务返回值，只负责触发 hooks。

        专业名词：
            emit（触发）：
                执行指定 hook_name 下的所有钩子。

            sync hook（同步钩子）：
                execute 方法使用普通 def 定义的 hook。

            async hook（异步钩子）：
                execute 方法使用 async def 定义的 hook。

            awaitable（可等待对象）：
                可以被 await 等待的对象，例如 coroutine、Task、Future。

            coroutine（协程）：
                async 函数调用后返回的异步执行对象。
        """

        hooks = self._hooks.get(
            hook_name,
            []
        )

        for hook in hooks:

            try:
                result = hook.execute(
                    *args,
                    **kwargs
                )

                if inspect.isawaitable(result):
                    await result

            except Exception as exc:
                self._log_hook_error(
                    hook_name=hook_name,
                    hook=hook,
                    error=exc,
                )

                if self.raise_hook_errors:
                    raise

    def _log_hook_error(
        self,
        hook_name: str,
        hook,
        error: Exception,
    ):
        """
        记录 hook 执行异常。

        功能：
            优先使用项目 logger 记录异常。
            如果项目 logger 因循环导入或其他原因不可用，则降级使用 Python 标准 logging。

        参数：
            hook_name：
                钩子名称，字符串格式。

            hook：
                当前执行失败的 hook 对象。

            error：
                捕获到的异常对象。

        返回值：
            None：无业务返回值，只负责记录日志。

        专业名词：
            lazy import（延迟导入）：
                不在模块顶部导入依赖，而是在函数内部真正需要时再导入，
                用于降低循环导入风险。

            fallback logger（兜底日志器）：
                当项目自定义 logger 不可用时，使用标准 logging 作为备用方案。
        """

        message = (
            "RuntimeHookManager hook 执行失败: "
            f"hook_name={hook_name}, "
            f"hook={hook.__class__.__name__}, "
            f"error={error}"
        )

        try:
            from src.logger import logger

            logger.exception(
                message
            )

        except Exception:
            fallback_logger = logging.getLogger(
                __name__
            )

            fallback_logger.exception(
                message
            )