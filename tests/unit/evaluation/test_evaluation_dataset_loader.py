import json
from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases


def test_load_agent_evaluation_cases_should_load_valid_dataset(
    tmp_path: Path,
) -> None:
    """
    测试通用加载器能把 JSON 转换成统一评估用例。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "root_route_001",
                    "category": "root_route",
                    "question": "今天成都天气怎么样？",
                    "expected": {
                        "route": "tool_agent",
                    },
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cases = load_agent_evaluation_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "root_route_001"
    assert cases[0].expected["route"] == "tool_agent"


def test_load_agent_evaluation_cases_should_support_cases_root(
    tmp_path: Path,
) -> None:
    """
    测试通用加载器兼容包含 cases 列表的 JSON 根字典。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "root_route_001",
                        "category": "root_route",
                        "question": "结束",
                        "expected": {
                            "route": "FINISH",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_agent_evaluation_cases(dataset_path)

    assert cases[0].expected == {"route": "FINISH"}


def test_load_agent_evaluation_cases_should_reject_duplicate_case_id(
    tmp_path: Path,
) -> None:
    """
    测试同一黄金集中重复的 case_id 会被拒绝。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "cases.json"
    duplicated_case = {
        "case_id": "duplicate_001",
        "category": "root_route",
        "question": "测试问题",
        "expected": {
            "route": "general_agent",
        },
    }
    dataset_path.write_text(
        json.dumps(
            [duplicated_case, duplicated_case],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case_id 重复"):
        load_agent_evaluation_cases(dataset_path)


def test_load_agent_evaluation_cases_should_reject_invalid_json(
    tmp_path: Path,
) -> None:
    """
    测试 JSON 语法错误时返回包含位置的清晰异常。

    参数含义：
        tmp_path:
            pytest 提供的临时目录。

    返回值含义：
        None。
    """

    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text("{ invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON 格式错误"):
        load_agent_evaluation_cases(dataset_path)
