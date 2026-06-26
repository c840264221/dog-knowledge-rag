"""
DogBreedMetadataExtractor 单元测试。

本测试只验证狗狗品种 metadata 提取逻辑：
1. 不调用 LLM
2. 不访问 Chroma
3. 不访问数据库
4. 不依赖 Container
"""

from src.rag.extractors.dog_breed_metadata_extractor import (
    DogBreedMetadataExtractor,
)

from src.rag.schemas import (
    RagDocument,
)


def build_test_rag_document(
        content: str,
        metadata: dict | None = None,
) -> RagDocument:
    """
    构建测试用 RagDocument。

    功能：
        根据当前项目 RagDocument schema 构建测试对象。
        如果你的 RagDocument 有 doc_id、source 等必填字段，
        这里会根据 model_fields 自动补充。

    参数：
        content: str
            Markdown 正文。

        metadata: dict | None
            初始 metadata。

    返回值：
        RagDocument：
            测试用 RAG 文档对象。
    """

    payload = {
        "content": content,
        "metadata": metadata or {},
    }

    model_fields = getattr(
        RagDocument,
        "model_fields",
        {},
    )

    if "doc_id" in model_fields:
        payload[
            "doc_id"
        ] = "test-doc-001"

    if "document_id" in model_fields:
        payload[
            "document_id"
        ] = "test-doc-001"

    if "id" in model_fields:
        payload[
            "id"
        ] = "test-doc-001"

    if "source" in model_fields:
        payload[
            "source"
        ] = "affenpinscher.md"

    if "source_path" in model_fields:
        payload[
            "source_path"
        ] = "affenpinscher.md"

    if "title" in model_fields:
        payload[
            "title"
        ] = "Affenpinscher"

    return RagDocument(
        **payload,
    )


def test_extract_metadata_from_current_akc_markdown_format():
    """
    测试是否能从当前 AKC Markdown 格式中提取 metadata。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    content = """
# Affenpinscher

## 🏷️ 标签
confident / famously funny / fearless

## 📏 基本信息
- 身高: 9-11.5 inches
- 体重: 7-10 pounds
- 寿命: 12-15 years

## 🧬 性格特征
- Affectionate With Family: 3
- Good With Young Children: 3
- Good With Other Dogs: 3
- Shedding Level: 3
- Coat Grooming Frequency: 3
- Drooling Level: 1
- Coat Type: Wiry
- Coat Length: Short, Medium
- Openness To Strangers: 5
- Playfulness Level: 3
- Watchdog/Protective Nature: 3
- Adaptability Level: 4
- Trainability Level: 3
- Energy Level: 3
- Barking Level: 3
- Mental Stimulation Needs: 3
"""

    extractor = DogBreedMetadataExtractor()

    metadata = extractor.extract_metadata_from_content(
        content=content,
    )

    assert metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert metadata[
        "dog_tags"
    ] == "confident / famously funny / fearless"

    assert metadata[
        "height_min_inches"
    ] == 9

    assert metadata[
        "height_max_inches"
    ] == 11.5

    assert metadata[
        "weight_min_pounds"
    ] == 7

    assert metadata[
        "weight_max_pounds"
    ] == 10

    assert metadata[
        "lifespan_min_years"
    ] == 12

    assert metadata[
        "lifespan_max_years"
    ] == 15

    assert metadata[
        "size"
    ] == "small"

    assert metadata[
        "trainability_level"
    ] == 3

    assert metadata[
        "energy_level"
    ] == 3

    assert metadata[
        "barking_level"
    ] == 3

    assert metadata[
        "shedding_level"
    ] == 3

    assert metadata[
        "coat_type"
    ] == "Wiry"

    assert metadata[
        "coat_length"
    ] == "Short, Medium"

    assert metadata[
        "good_for_beginner"
    ] is True

    assert metadata[
        "good_for_apartment"
    ] is True

    assert metadata[
        "metadata_extractor"
    ] == "dog_breed_metadata_extractor_v2"

    assert metadata[
        "metadata_schema"
    ] == "akc_markdown_dog_breed_v1"


def test_extract_should_return_new_document_with_enhanced_metadata():
    """
    测试 extract 是否返回 metadata 增强后的新 RagDocument。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    content = """
# Affenpinscher

## 🏷️ 标签
confident / famously funny / fearless

## 📏 基本信息
- 身高: 9-11.5 inches
- 体重: 7-10 pounds
- 寿命: 12-15 years

## 🧬 性格特征
- Trainability Level: 3
- Energy Level: 3
- Barking Level: 3
- Adaptability Level: 4
- Mental Stimulation Needs: 3
"""

    document = build_test_rag_document(
        content=content,
        metadata={
            "source_type": "markdown",
        },
    )

    extractor = DogBreedMetadataExtractor()

    result = extractor.extract(
        document=document,
    )

    assert result.metadata[
        "source_type"
    ] == "markdown"

    assert result.metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert result.metadata[
        "dog_tags"
    ] == "confident / famously funny / fearless"

    assert result.metadata[
        "size"
    ] == "small"

    assert result.metadata[
        "good_for_beginner"
    ] is True

    assert result.metadata[
        "good_for_apartment"
    ] is True


def test_extract_should_not_overwrite_existing_metadata_by_default():
    """
    测试默认情况下不覆盖已有 metadata。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    content = """
# Affenpinscher

## 📏 基本信息
- 体重: 7-10 pounds

## 🧬 性格特征
- Energy Level: 3
"""

    document = build_test_rag_document(
        content=content,
        metadata={
            "dog_name": "Manual Name",
            "energy_level": 5,
        },
    )

    extractor = DogBreedMetadataExtractor(
        overwrite_existing=False,
    )

    result = extractor.extract(
        document=document,
    )

    assert result.metadata[
        "dog_name"
    ] == "Manual Name"

    assert result.metadata[
        "energy_level"
    ] == 5

    assert result.metadata[
        "weight_min_pounds"
    ] == 7

    assert result.metadata[
        "weight_max_pounds"
    ] == 10


def test_extract_should_overwrite_existing_metadata_when_enabled():
    """
    测试开启 overwrite_existing 后覆盖已有 metadata。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    content = """
# Affenpinscher

## 🧬 性格特征
- Energy Level: 3
"""

    document = build_test_rag_document(
        content=content,
        metadata={
            "dog_name": "Manual Name",
            "energy_level": 5,
        },
    )

    extractor = DogBreedMetadataExtractor(
        overwrite_existing=True,
    )

    result = extractor.extract(
        document=document,
    )

    assert result.metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert result.metadata[
        "energy_level"
    ] == 3


def test_extract_should_filter_none_and_complex_metadata_values():
    """
    测试 None 和复杂类型不会写入最终 metadata。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    content = """
# Unknown Breed

## 📖 关于该犬种
Only plain text here.
"""

    document = build_test_rag_document(
        content=content,
        metadata={
            "valid_field": "valid",
            "invalid_list": [
                "a",
                "b",
            ],
            "invalid_dict": {
                "a": 1,
            },
        },
    )

    extractor = DogBreedMetadataExtractor()

    result = extractor.extract(
        document=document,
    )

    assert result.metadata[
        "dog_name"
    ] == "Unknown Breed"

    assert result.metadata[
        "valid_field"
    ] == "valid"

    assert "invalid_list" not in result.metadata

    assert "invalid_dict" not in result.metadata

    assert "height_min_inches" not in result.metadata

    assert "weight_min_pounds" not in result.metadata


def test_extract_many_should_process_document_list():
    """
    测试 extract_many 是否能批量处理文档列表。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    first_document = build_test_rag_document(
        content="""
# Affenpinscher

## 🧬 性格特征
- Energy Level: 3
""",
    )

    second_document = build_test_rag_document(
        content="""
# Beagle

## 🧬 性格特征
- Barking Level: 4
""",
    )

    extractor = DogBreedMetadataExtractor()

    results = extractor.extract_many(
        documents=[
            first_document,
            second_document,
        ],
    )

    assert len(
        results
    ) == 2

    assert results[0].metadata[
        "dog_name"
    ] == "Affenpinscher"

    assert results[0].metadata[
        "energy_level"
    ] == 3

    assert results[1].metadata[
        "dog_name"
    ] == "Beagle"

    assert results[1].metadata[
        "barking_level"
    ] == 4