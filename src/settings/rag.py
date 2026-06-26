from pydantic import Field
from pydantic_settings import SettingsConfigDict

from src.settings.base import BaseAppSettings


class RagSettings(BaseAppSettings):
    """
    RagSettings：RAG 模块配置。

    功能：
        管理 RAG 检索、召回质量评估、重试策略等相关配置。

    技术名词：
        RAG：
            Retrieval-Augmented Generation，检索增强生成。
            先从知识库召回相关内容，再交给 LLM 生成回答。

        Retrieval Quality：
            召回质量。表示召回内容是否足够支撑答案生成。

        Threshold：
            阈值。用于判断某个分数是否达到要求。

        Weight：
            权重。用于计算综合质量分。

    字段说明：
        quality_min_score:
            最低质量分。低于该值时认为召回质量不足。

        quality_min_context_chars:
            最小上下文字符数。context_text 太短时认为信息不足。

        quality_max_distance:
            最大允许向量距离。当前默认假设 retrieval_score 是 distance，
            分数越小越相似。

        quality_enable_metadata_check:
            是否启用 metadata 命中检查。

        quality_enable_section_check:
            是否启用 section 主题匹配检查。

        quality_enable_score_check:
            是否启用 retrieval_score 检查。

        quality_non_empty_weight:
            召回非空权重。

        quality_metadata_hit_weight:
            metadata 命中权重。

        quality_section_hit_weight:
            section 命中权重。

        quality_score_weight:
            retrieval_score 合格权重。

        quality_context_length_weight:
            context_text 长度合格权重。

        retry_max_count:
            Graph 层最大重试次数。

        retry_core_filter_fields:
            retry_node 放宽过滤条件时保留的核心字段。
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    quality_min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    quality_min_context_chars: int = Field(
        default=100,
        ge=0,
    )

    quality_max_distance: float | None = 0.8

    quality_enable_metadata_check: bool = True

    quality_enable_section_check: bool = True

    quality_enable_score_check: bool = True

    quality_non_empty_weight: float = 0.2

    quality_metadata_hit_weight: float = 0.3

    quality_section_hit_weight: float = 0.2

    quality_score_weight: float = 0.2

    quality_context_length_weight: float = 0.1

    retry_max_count: int = Field(
        default=3,
        ge=0,
    )

    retry_first_top_k: int = Field(
        default=10,
        ge=1,
    )

    retry_second_top_k: int = Field(
        default=15,
        ge=1,
    )

    retry_third_top_k: int = Field(
        default=25,
        ge=1,
    )

    retry_core_filter_fields: list[str] = Field(
        default_factory=lambda: [
            "dog_name",
            "size",
        ]
    )