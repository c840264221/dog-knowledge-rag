# 定义基类异常
class DogAgentError(Exception):
    """
    所有Agent异常基类
    """

    def __init__(self,message: str,recoverable: bool = True):

        super().__init__(message)

        self.message = message

        # 是否是可恢复的异常
        self.recoverable = recoverable