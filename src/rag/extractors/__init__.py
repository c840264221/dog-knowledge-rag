"""
RAG Extractors 模块导出文件。

Extractor（提取器）：
用于从 RagDocument.content 中提取结构化信息，
例如 metadata（元数据）、tags（标签）、实体信息等。

这个 __init__.py 的作用：
1. 标记 src/rag/extractors 是一个 Python package（包）
2. 统一导出当前目录下可被外部使用的 Extractor 类
3. 让外部代码可以用更简洁的方式导入
"""

from src.rag.extractors.dog_breed_metadata_extractor import (
    DogBreedMetadataExtractor,
)


__all__ = [
    "DogBreedMetadataExtractor",
]