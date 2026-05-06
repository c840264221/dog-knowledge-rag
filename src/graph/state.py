from typing import TypedDict, List, Dict, Any


class DogState(TypedDict):
    question: str

    # parser输出
    intent: str
    filters: Dict
    tags: List[str]
    features: List[str]
    dog_name: str

    # 中间结果
    docs: List[Any]

    # 输出
    answer: str