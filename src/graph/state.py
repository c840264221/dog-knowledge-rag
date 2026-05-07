from typing import TypedDict, List, Dict, Any


class DogState(TypedDict):
    question: str

    # parser输出
    intent: str
    filters: Dict
    tags: List[str]
    features: List[str]
    dog_name: str

    # 检索策略
    strategy: str

    # 中间结果
    docs: List[Any]

    # 重试次数
    retry_count:int

    # 输出
    answer: str