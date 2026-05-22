from typing import TypedDict,List, Optional, Dict, Any, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RecommendationState(TypedDict):
    # 用户问题
    question: str

    # 根据语意解析出的关键词
    intent: str


    # （以后就不需要了 现在为了能让程序正常运行不会报key_error先加上）
    # 检索模式
    strategy: str
    # 狗狗的名字
    dog_name: Optional[str]

    # 过滤条件
    filters: dict

    # 标签 用于手动过滤
    tags: List[str]

    # 搜索到的数据  是Document类型  包含content内容和metadata
    docs: list

    # 检索结果是否可以 例如数量不够则为False反之则为True
    retrieval_ok: str

    # LLM返回的答案
    answer: str

    # 存储用户对交互问题的回答
    user_feedback: str

    # 是否询问过用户  此字段防止重复询问
    has_asked_user: bool

    # 重试次数
    retry_count: int

    # messages: List[BaseMessage]  # 存储对话历史
    messages: Annotated[List[BaseMessage], add_messages]

    user_id:str