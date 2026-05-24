class BaseRuntimeError(Exception):
    """
    Runtime基础错误
    """

    def __init__(self,message: str,retryable: bool = False):

        super().__init__(message)

        self.message = message

        self.retryable = retryable