import json
from pathlib import Path

import pytest

from src.rag.evaluation import (
    RagEvalDatasetLoader,
    load_rag_eval_cases,
)


@pytest.fixture
def sample_eval_dataset_path(
    tmp_path: Path,
) -> Path:
    """
    创建临时 RAG 评估数据集文件。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        Path:
            临时评估数据集 JSON 文件路径。
    """

    dataset_path = tmp_path / "dog_rag_eval_cases.json"

    dataset = [
        {
            "case_id": "dog_eval_001",
            "question": "适合新手养的小型犬有哪些？",
            "expected_dog_names": [
                "Poodle"
            ],
            "expected_filters": {
                "size": "small",
                "trainability": "high"
            },
            "top_k": 5,
            "tags": [
                "small_dog",
                "beginner"
            ],
            "note": "测试小型犬和新手友好问题。"
        }
    ]

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return dataset_path


def test_load_rag_eval_cases_success(
    sample_eval_dataset_path: Path,
) -> None:
    """
    测试可以成功加载 RAG 评估用例。

    参数含义：
        sample_eval_dataset_path:
            临时 RAG 评估数据集文件路径。

    返回值含义：
        None。
    """

    cases = load_rag_eval_cases(sample_eval_dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "dog_eval_001"
    assert cases[0].question == "适合新手养的小型犬有哪些？"
    assert cases[0].expected_dog_names == ["Poodle"]
    assert cases[0].expected_filters == {
        "size": "small",
        "trainability": "high",
    }
    assert cases[0].top_k == 5


def test_loader_supports_dict_root_with_cases(
    tmp_path: Path,
) -> None:
    """
    测试 loader 支持 {"cases": [...]} 这种 JSON 根结构。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "dog_rag_eval_cases.json"

    dataset = {
        "cases": [
            {
                "case_id": "dog_eval_001",
                "question": "Golden Retriever 适合新手吗？",
                "expected_dog_names": [
                    "Golden Retriever"
                ],
                "expected_filters": {
                    "dog_name": "Golden Retriever"
                },
                "top_k": 5
            }
        ]
    }

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = RagEvalDatasetLoader(dataset_path)
    cases = loader.load()

    assert len(cases) == 1
    assert cases[0].case_id == "dog_eval_001"
    assert cases[0].expected_filters == {
        "dog_name": "Golden Retriever"
    }


def test_loader_raises_when_file_not_exists(
    tmp_path: Path,
) -> None:
    """
    测试数据集文件不存在时会抛出 FileNotFoundError。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "not_exists.json"

    loader = RagEvalDatasetLoader(dataset_path)

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_loader_raises_when_json_invalid(
    tmp_path: Path,
) -> None:
    """
    测试 JSON 格式错误时会抛出 ValueError。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    loader = RagEvalDatasetLoader(dataset_path)

    with pytest.raises(ValueError, match="JSON 格式错误"):
        loader.load()


def test_loader_raises_when_case_id_duplicate(
    tmp_path: Path,
) -> None:
    """
    测试 case_id 重复时会抛出 ValueError。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "duplicate.json"

    dataset = [
        {
            "case_id": "dog_eval_001",
            "question": "问题 1",
            "expected_dog_names": [
                "Poodle"
            ],
            "expected_filters": {},
            "top_k": 5
        },
        {
            "case_id": "dog_eval_001",
            "question": "问题 2",
            "expected_dog_names": [
                "Maltese"
            ],
            "expected_filters": {},
            "top_k": 5
        }
    ]

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = RagEvalDatasetLoader(dataset_path)

    with pytest.raises(ValueError, match="case_id 重复"):
        loader.load()


def test_loader_raises_when_required_field_missing(
    tmp_path: Path,
) -> None:
    """
    测试必填字段缺失时会抛出 ValueError。

    参数含义：
        tmp_path:
            pytest 提供的临时目录 fixture。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "missing_field.json"

    dataset = [
        {
            "case_id": "dog_eval_001",
            "expected_dog_names": [
                "Poodle"
            ],
            "expected_filters": {},
            "top_k": 5
        }
    ]

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    loader = RagEvalDatasetLoader(dataset_path)

    with pytest.raises(ValueError, match="校验失败"):
        loader.load()