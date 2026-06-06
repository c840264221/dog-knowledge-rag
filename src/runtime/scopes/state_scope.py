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

    def get_agent(self):
        return self.get_state().current_agent

    def set_node(self, node_name):

        self.state.current_node = node_name

    def get_node(self):
        return self.get_state().current_node

    def set_tool(self, tool_name):

        self.state.current_tool = tool_name

    def set_phase(self, phase):

        self.state.phase = phase

    def add_history(self, item):

        self.state.execution_history.append(
            item
        )

    def set_retry_count(self, retry_count):
        self.state.retry_count = retry_count

    def get_retry_count(self):
        return self.get_state().retry_count

    def export(self):
        return {

            "current_agent":
                self.state.current_agent,

            "current_node":
                self.state.current_node,

            "current_tool":
                self.state.current_tool,

            "phase":
                self.state.phase,

            "retry_count":
                self.state.retry_count
        }

    def restore(self, data):
        self.state.current_agent = (
            data.get("current_agent")
        )

        self.state.current_node = (
            data.get("current_node")
        )

        self.state.current_tool = (
            data.get("current_tool")
        )

        self.state.phase = (
            data.get("phase")
        )

        self.state.retry_count = (
            data.get("retry_count", 0)
        )

    async def startup(self):
        pass

    async def shutdown(self):

        self.state.execution_history.clear()