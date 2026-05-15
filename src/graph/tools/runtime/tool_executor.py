import asyncio

from src.logger import logger


def safe_execute_tool(
    func,
    args=None,
    timeout=5
):
    logger.debug(f"args: {args}, args_type:{type(args)},args is not None---{args is not None}")
    try:

        # 异步工具调用
        if asyncio.iscoroutinefunction(func):

            result = asyncio.run(
                asyncio.wait_for(
                    func(args) if len(args) != 0 else func(),
                    timeout=timeout
                )
            )

        # 同步工具调用
        else:
            result = (
                func(args)
                if len(args) != 0
                else func()
            )

        return result

    except asyncio.TimeoutError:

        logger.warning(
            f"工具执行超时: {func.__name__}"
        )

        return "工具执行超时"

    except Exception as e:

        logger.exception(
            f"工具执行失败: {e}"
        )

        return f"工具执行失败: {str(e)}"