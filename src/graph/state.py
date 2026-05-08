from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class DogState(TypedDict, total=False):
    # 用户问题
    question: str

    # 根据语意解析出的关键词
    intent: str

    # 检索模式
    strategy: str

    #过滤条件
    filters: dict

    # 标签 用于手动过滤
    tags: List[str]

    #  特点  由于知识库不完整暂时用不上 后续可以根据features在向量搜索之前来进行精确过滤
    features: List[str]

    # 狗狗的名字
    dog_name: Optional[str]

    # 搜索到的数据  是Document类型  包含content内容和metadata
    docs: List[Document]

    # LLM返回的答案
    answer: str

    # 重试次数
    retry_count: int

    # 检索结果是否可以 例如数量不够则为False反之则为True
    retrieval_ok: bool

    # 搜索结果取前几项
    top_k: int