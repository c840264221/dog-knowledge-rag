from collections import defaultdict


class RuntimeHookManager:

    def __init__(self):

        self._hooks = defaultdict(list)

    def register(self,hook_name:str,hook):

        self._hooks[hook_name].append(
            hook
        )

    async def emit(self,hook_name:str,*args,**kwargs):

        hooks = self._hooks.get(

            hook_name,

            []
        )

        for hook in hooks:

            await hook.execute(

                *args,

                **kwargs
            )