import asyncio
from typing import Type
from functools import wraps
from src.logger import logger


def retry_async(max_attempts: int=3, delay:int=1, backoff:float=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1,max_attempts+1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    logger.info(f"尝试 {attempt}/{max_attempts} 失败: {e}, {_delay}秒后重试...")
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return None
        return async_wrapper
    return decorator