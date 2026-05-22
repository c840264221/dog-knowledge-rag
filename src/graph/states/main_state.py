from typing import TypedDict

# 多 Agent 版本的状态，继承所有业务字段
class MainState(TypedDict):

    question: str

    current_agent: str

    next_agent: str

    answer: str