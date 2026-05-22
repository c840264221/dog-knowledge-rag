import time

from src.logger import logger


class RetryMiddleware:

    def run(self,func,retries):

        last_error = None

        for i in range(retries):

            try:

                return func()

            except Exception as e:

                last_error = e

                logger.info(f'重试{i+1}次')

                time.sleep(1)

        raise last_error