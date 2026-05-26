import time

from functools import wraps

from src.logger import logger


def safe_node(fallback=None,raise_error=False):

    def decorator(func):

        @wraps(func)
        async def wrapper(state, *args, **kwargs):

            start = time.time()

            node_name = func.__name__

            logger.info(
                f"[Node Start] {node_name}"
            )

            try:

                result = await func(state,*args,**kwargs)

                elapsed = (
                    time.time() - start
                )

                logger.info(
                    f"[Node Success] "
                    f"{node_name} "
                    f"({elapsed:.2f}s)"
                )

                return result

            except Exception as e:

                elapsed = (
                    time.time() - start
                )

                logger.exception(
                    f"[Node Error] "
                    f"{node_name} "
                    f"({elapsed:.2f}s): {e}"
                )

                # ===== 写入错误状态 =====

                error_state = {

                    "error": str(e),

                    "failed_node": node_name
                }

                # ===== fallback =====

                if fallback:

                    fallback_result = await fallback(state,e)

                    if isinstance(fallback_result,dict):
                        error_state.update(fallback_result)

                # ===== 是否继续抛异常 =====

                if raise_error:
                    raise

                return error_state

        return wrapper

    return decorator