from src.runtime.scopes.base_scope import (
    BaseScope
)

from src.runtime.state.runtime_state import (
    RuntimeState
)


class StateScope(BaseScope):

    def __init__(self):

        self.state = RuntimeState()

    def get_state(self):

        return self.state

    def set_agent(self, agent_name):

        self.state.current_agent = agent_name

    def set_node(self, node_name):

        self.state.current_node = node_name

    def set_tool(self, tool_name):

        self.state.current_tool = tool_name

    def set_phase(self, phase):

        self.state.phase = phase

    def add_history(self, item):

        self.state.execution_history.append(
            item
        )

    async def startup(self):
        pass

    async def shutdown(self):

        self.state.execution_history.clear()