from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """
    EmbeddingResult：向量化结果

    中文释义：
    - embedding：向量，一组浮点数
    - model_name：模型名称
    - provider：模型提供方
    - text：原始文本
    """

    embedding: list[float] = Field(
        description="文本生成的向量结果"
    )

    model_name: str = Field(
        description="Embedding 模型名称"
    )

    provider: str = Field(
        description="Embedding 模型提供方"
    )

    text: str = Field(
        description="原始输入文本"
    )