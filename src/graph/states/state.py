# from typing import TypedDict, List, Optional, Dict, Any, Annotated, Sequence
# from langchain_core.documents import Document
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages
#
#
# class DogState(TypedDict, total=False):
#     # 用户问题
#     question: str
#
#     # 根据语意解析出的关键词
#     intent: str
#
#     # 检索模式
#     strategy: str
#
#     #过滤条件
#     filters: dict
#
#     # 标签 用于手动过滤
#     tags: List[str]
#
#     #  特点  由于知识库不完整暂时用不上 后续可以根据features在向量搜索之前来进行精确过滤
#     features: List[str]
#
#     # 狗狗的名字
#     dog_name: Optional[str]
#
#     # 搜索到的数据  是Document类型  包含content内容和metadata
#     docs: List[Document]
#
#     # LLM返回的答案
#     answer: str
#
#     # 重试次数
#     retry_count: int
#
#     # 检索结果是否可以 例如数量不够则为False反之则为True
#     retrieval_ok: bool
#
#     # 搜索结果取前几项
#     top_k: int
#
#     # 存储用户对交互问题的回答
#     user_feedback: Optional[str]
#
#     # 是否询问过用户  此字段防止重复询问
#     has_asked_user: bool
#
#     # ===== 工具调用新增字段 =====
#     tool_calls: List[Dict[str, Any]]  # 存储解析出的工具调用请求 [{name, args}]
#     tool_results: List[str]  # 存储工具执行的结果文本
#     need_tool: bool  # 当前是否需要执行工具
#     tool_round: int  # 已执行的工具轮次（防止无限循环）
#     pending_prompt: str  # 保留之前人机交互用
#     tool_confirmed:str
#     tool_executed: bool
#     waiting_user_input: bool
#
#
#     # messages: List[BaseMessage]  # 存储对话历史
#     messages: Annotated[List[BaseMessage], add_messages]
#
#     # 多智能体的下一工作的worker
#     next_worker:str
#
#     # 下一个工作的agent
#     next_agent:str
#
#     user_id:str
#
#     trace_id: str
#
#     memory_context: str


# 导入新的dog_state  这个旧版的先不删除  这样可以兼容旧版的graph设计
from src.graph.states.dog_state import DogState


__all__ = [
    "DogState",
]
