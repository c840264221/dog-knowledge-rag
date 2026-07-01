from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.rag.evaluation.schemas import RagEvalCase


class RagEvalDatasetLoader:
    """
    RAG 评估数据集加载器。

    用于从 JSON 文件中读取 RAG Evaluation 测试用例，
    并将原始 dict 数据转换成 RagEvalCase 对象。

    参数含义：
        dataset_path: 评估数据集文件路径，例如 data/rag_eval/dog_rag_eval_cases.json。

    返回值含义：
        RagEvalDatasetLoader 实例。
    """

    def __init__(
        self,
        dataset_path: str | Path,
    ) -> None:
        """
        初始化 RAG 评估数据集加载器。

        参数含义：
            dataset_path:
                评估数据集文件路径。
                支持 str 和 pathlib.Path 两种类型。

        返回值含义：
            None。
        """

        self.dataset_path = Path(dataset_path)

    def load(self) -> list[RagEvalCase]:
        """
        加载 RAG 评估用例列表。

        执行流程：
            1. 检查数据集文件是否存在。
            2. 读取 JSON 文件。
            3. 提取 case 列表。
            4. 将每条原始 dict 转换成 RagEvalCase。
            5. 返回 list[RagEvalCase]。

        参数含义：
            无。

        返回值含义：
            list[RagEvalCase]:
                已通过 Pydantic 校验的 RAG 评估用例列表。

        异常：
            FileNotFoundError:
                当数据集文件不存在时抛出。
            ValueError:
                当 JSON 格式错误、根结构错误、case 校验失败时抛出。
        """

        raw_data = self._read_json_file()
        raw_cases = self._extract_raw_cases(raw_data)

        return self._parse_cases(raw_cases)

    def _read_json_file(self) -> Any:
        """
        读取 JSON 数据集文件。

        参数含义：
            无。

        返回值含义：
            Any:
                json.loads 解析后的原始 Python 对象。
                可能是 list，也可能是 dict。

        异常：
            FileNotFoundError:
                当数据集文件不存在时抛出。
            ValueError:
                当 JSON 内容解析失败时抛出。
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"RAG 评估数据集文件不存在: {self.dataset_path}"
            )

        if not self.dataset_path.is_file():
            raise ValueError(
                f"RAG 评估数据集路径不是文件: {self.dataset_path}"
            )

        try:
            return json.loads(
                self.dataset_path.read_text(encoding="utf-8")
            )
        except JSONDecodeError as exc:
            raise ValueError(
                f"RAG 评估数据集 JSON 格式错误: {self.dataset_path}, "
                f"错误位置 line={exc.lineno}, column={exc.colno}, "
                f"message={exc.msg}"
            ) from exc

    def _extract_raw_cases(
        self,
        raw_data: Any,
    ) -> list[dict[str, Any]]:
        """
        从原始 JSON 数据中提取评估用例列表。

        当前支持两种 JSON 结构：

        结构一：
            [
              {"case_id": "...", "question": "..."}
            ]

        结构二：
            {
              "cases": [
                {"case_id": "...", "question": "..."}
              ]
            }

        参数含义：
            raw_data:
                从 JSON 文件中解析出来的原始 Python 对象。

        返回值含义：
            list[dict[str, Any]]:
                原始评估用例 dict 列表。

        异常：
            ValueError:
                当 JSON 根结构不是 list 或不包含 cases 字段时抛出。
        """

        if isinstance(raw_data, list):
            return self._ensure_case_dict_list(raw_data)

        if isinstance(raw_data, dict):
            cases = raw_data.get("cases")

            if cases is None:
                raise ValueError(
                    "RAG 评估数据集 dict 根结构必须包含 cases 字段"
                )

            if not isinstance(cases, list):
                raise ValueError(
                    "RAG 评估数据集 cases 字段必须是 list 类型"
                )

            return self._ensure_case_dict_list(cases)

        raise ValueError(
            "RAG 评估数据集根结构必须是 list，"
            "或者是包含 cases 字段的 dict"
        )

    def _ensure_case_dict_list(
        self,
        raw_cases: list[Any],
    ) -> list[dict[str, Any]]:
        """
        确保原始 case 列表中的每一项都是 dict。

        参数含义：
            raw_cases:
                原始 case 列表。

        返回值含义：
            list[dict[str, Any]]:
                已确认每一项都是 dict 的 case 列表。

        异常：
            ValueError:
                当某一条 case 不是 dict 类型时抛出。
        """

        normalized_cases: list[dict[str, Any]] = []

        for index, raw_case in enumerate(raw_cases, start=1):
            if not isinstance(raw_case, dict):
                raise ValueError(
                    f"第 {index} 条 RAG 评估用例必须是 dict 类型，"
                    f"实际类型: {type(raw_case).__name__}"
                )

            normalized_cases.append(raw_case)

        return normalized_cases

    def _parse_cases(
        self,
        raw_cases: list[dict[str, Any]],
    ) -> list[RagEvalCase]:
        """
        将原始 dict 列表转换成 RagEvalCase 列表。

        参数含义：
            raw_cases:
                从 JSON 文件中读取出来的原始 case dict 列表。

        返回值含义：
            list[RagEvalCase]:
                已完成字段校验和类型转换的评估用例列表。

        异常：
            ValueError:
                当评估数据为空、case_id 重复、字段校验失败时抛出。
        """

        if not raw_cases:
            raise ValueError("RAG 评估数据集不能为空")

        cases: list[RagEvalCase] = []
        seen_case_ids: set[str] = set()

        for index, raw_case in enumerate(raw_cases, start=1):
            try:
                eval_case = RagEvalCase.model_validate(raw_case)
            except ValidationError as exc:
                raise ValueError(
                    f"第 {index} 条 RAG 评估用例校验失败: {exc}"
                ) from exc

            if eval_case.case_id in seen_case_ids:
                raise ValueError(
                    f"RAG 评估用例 case_id 重复: {eval_case.case_id}"
                )

            seen_case_ids.add(eval_case.case_id)
            cases.append(eval_case)

        return cases


def load_rag_eval_cases(
    dataset_path: str | Path,
) -> list[RagEvalCase]:
    """
    快捷加载 RAG 评估用例。

    这是一个函数式入口，适合调试脚本或简单调用场景。
    内部会创建 RagEvalDatasetLoader，并调用 load 方法。

    参数含义：
        dataset_path:
            评估数据集文件路径。

    返回值含义：
        list[RagEvalCase]:
            已加载并校验完成的 RAG 评估用例列表。
    """

    loader = RagEvalDatasetLoader(dataset_path=dataset_path)

    return loader.load()