from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.evaluation.schemas import AgentEvaluationCase


class AgentEvaluationDatasetLoader:
    """
    加载统一 Agent Evaluation（评估）黄金数据集。

    功能：
        从 JSON 文件读取评估用例，校验文件结构、用例字段和 case_id 唯一性，
        最终返回经过 Pydantic 校验的 AgentEvaluationCase 列表。

    参数含义：
        dataset_path:
            黄金评估集 JSON 文件路径。

    返回值含义：
        AgentEvaluationDatasetLoader:
            可重复调用 load 方法的数据集加载器。
    """

    def __init__(
        self,
        dataset_path: str | Path,
    ) -> None:
        """
        初始化统一评估数据集加载器。

        参数含义：
            dataset_path:
                黄金评估集 JSON 文件路径。

        返回值含义：
            None。
        """

        self.dataset_path = Path(dataset_path)

    def load(self) -> list[AgentEvaluationCase]:
        """
        读取并校验统一评估数据集。

        参数含义：
            无。

        返回值含义：
            list[AgentEvaluationCase]:
                按 JSON 原始顺序排列的评估用例列表。
        """

        self._validate_dataset_path()

        try:
            raw_dataset = json.loads(
                self.dataset_path.read_text(
                    encoding="utf-8",
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Agent 评估数据集 JSON 格式错误: {self.dataset_path}, "
                f"line={exc.lineno}, column={exc.colno}"
            ) from exc

        raw_cases = self._extract_raw_cases(raw_dataset)
        cases = self._build_cases(raw_cases)
        self._validate_unique_case_ids(cases)
        return cases

    def _validate_dataset_path(self) -> None:
        """
        校验评估数据集路径存在且指向文件。

        参数含义：
            无。

        返回值含义：
            None。
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Agent 评估数据集不存在: {self.dataset_path}"
            )

        if not self.dataset_path.is_file():
            raise ValueError(
                f"Agent 评估数据集路径不是文件: {self.dataset_path}"
            )

    def _extract_raw_cases(
        self,
        raw_dataset: Any,
    ) -> list[Any]:
        """
        从 JSON 根结构提取原始用例列表。

        参数含义：
            raw_dataset:
                json.loads 返回的原始 Python 对象。

        返回值含义：
            list[Any]:
                尚未经过 Schema 校验的原始用例列表。
        """

        if isinstance(raw_dataset, list):
            raw_cases = raw_dataset
        elif isinstance(raw_dataset, dict):
            raw_cases = raw_dataset.get("cases")
        else:
            raw_cases = None

        if not isinstance(raw_cases, list):
            raise ValueError(
                "Agent 评估数据集根结构必须是列表，"
                "或包含 cases 列表的字典。"
            )

        if not raw_cases:
            raise ValueError("Agent 评估数据集不能是空列表")

        return raw_cases

    def _build_cases(
        self,
        raw_cases: list[Any],
    ) -> list[AgentEvaluationCase]:
        """
        将原始字典列表转换成统一评估用例对象。

        参数含义：
            raw_cases:
                尚未经过 Schema 校验的原始用例列表。

        返回值含义：
            list[AgentEvaluationCase]:
                Pydantic 校验通过的评估用例列表。
        """

        cases: list[AgentEvaluationCase] = []

        for index, raw_case in enumerate(raw_cases):
            try:
                cases.append(
                    AgentEvaluationCase.model_validate(raw_case)
                )
            except ValidationError as exc:
                raise ValueError(
                    f"Agent 评估数据集第 {index + 1} 条用例校验失败: {exc}"
                ) from exc

        return cases

    def _validate_unique_case_ids(
        self,
        cases: list[AgentEvaluationCase],
    ) -> None:
        """
        校验同一数据集中的 case_id 不重复。

        参数含义：
            cases:
                已通过字段校验的评估用例列表。

        返回值含义：
            None。
        """

        seen_case_ids: set[str] = set()

        for eval_case in cases:
            if eval_case.case_id in seen_case_ids:
                raise ValueError(
                    f"Agent 评估数据集 case_id 重复: {eval_case.case_id}"
                )

            seen_case_ids.add(eval_case.case_id)


def load_agent_evaluation_cases(
    dataset_path: str | Path,
) -> list[AgentEvaluationCase]:
    """
    便捷加载统一 Agent 评估数据集。

    参数含义：
        dataset_path:
            黄金评估集 JSON 文件路径。

    返回值含义：
        list[AgentEvaluationCase]:
            校验通过的统一评估用例列表。
    """

    return AgentEvaluationDatasetLoader(
        dataset_path=dataset_path,
    ).load()
