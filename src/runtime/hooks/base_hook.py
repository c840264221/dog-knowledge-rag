from abc import ABC

class BaseHook(ABC):

    async def execute(self,*args,**kwargs):
        pass