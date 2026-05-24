from abc import ABC
from abc import abstractmethod


class BaseMiddleware(ABC):

    @abstractmethod
    async def process(self,ctx,next_func):
        pass