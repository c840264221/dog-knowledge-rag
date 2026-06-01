from src.runtime.hooks.base_hook import BaseHook

class TestHook(BaseHook):

    async def execute(

        self,

        *args,

        **kwargs
    ):

        print("hook触发")