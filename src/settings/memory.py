from pydantic_settings import SettingsConfigDict

from src.settings.base import BaseAppSettings


class MemorySettings(BaseAppSettings):
    """
    MemorySettings：Memory 模块配置。

    功能：
    - 管理 Memory 系统相关配置
    - 管理 MemoryRanker 的排序权重
    - 后续可以继续扩展 memory top_k、candidate_k、decay_rate 等配置

    字段说明：
    - semantic_weight: 语义分数权重
    - memory_weight: 记忆强度与时间衰减分数权重
    - confidence_weight: 可信度分数权重
    - importance_weight: 记忆重要程度权重
    - minimum_semantic_score: 允许记忆进入精排的最低语义相关分数
    - default_top_k: 默认最终召回数量
    - default_candidate_k: 默认候选召回数量
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    semantic_weight: float = 3.0

    memory_weight: float = 1.0

    confidence_weight: float = 0.5

    importance_weight: float = 0.5

    minimum_semantic_score: float = 0.45

    default_top_k: int = 5

    default_candidate_k: int = 20
