from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError
)


class TimeoutMiddleware:

    def execute(self,func,timeout):

        with ThreadPoolExecutor(max_workers=1) as executor:

            future = executor.submit(func)

            try:

                return future.result(
                    timeout=timeout
                )

            except TimeoutError:

                raise Exception(
                    "工具执行超时"
                )