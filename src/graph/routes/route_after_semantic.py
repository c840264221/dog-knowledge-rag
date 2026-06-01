from src.graph.states.state import DogState

from src.runtime.context import runtime_ctx

def route_after_semantic(state: DogState) -> str:

    next_agent = state.get("next_agent", "general_agent")


    runtime_ctx.get().state().set_agent(
        next_agent
    )

    return next_agent