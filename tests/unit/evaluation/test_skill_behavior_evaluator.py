"""Skill（技能）确定性行为评估器测试。"""

from pathlib import Path

import pytest

from src.evaluation import load_agent_evaluation_cases
from src.evaluation.evaluators import SkillBehaviorEvaluator
from src.evaluation.schemas import AgentEvaluationCase


DATASET_PATH = Path("evaluation/datasets/skill_behavior_cases.json")


@pytest.mark.asyncio
async def test_skill_behavior_dataset_should_pass_real_runtime() -> None:
    """
    检查全部 Skill 黄金用例是否通过真实 SkillRuntime 评估。

    功能：
        加载 JSON 黄金集，依次执行技能选择、输入提取、缺参检查、恢复合并
        和上下文加载，并要求所有字段比较结果通过。

    参数含义：
        无。

    返回值含义：
        None。
    """

    # 从 JSON 读取并经过 AgentEvaluationCase 校验的标准黄金用例。
    eval_cases = load_agent_evaluation_cases(DATASET_PATH)

    # 每条结果都来自真实默认 SkillRuntime，不使用固定 Skill 输出替身。
    results = await SkillBehaviorEvaluator().evaluate_many(eval_cases)

    assert len(results) == 5
    assert all(result.passed for result in results), [
        {
            "case_id": result.case_id,
            "error_message": result.error_message,
            "failed_checks": [
                check.model_dump(mode="python")
                for check in result.failed_checks()
            ],
        }
        for result in results
        if not result.passed
    ]


@pytest.mark.asyncio
async def test_skill_behavior_evaluator_should_expose_resume_output() -> None:
    """
    检查恢复用例是否输出上游技能编号和新旧输入合并结果。

    参数含义：
        无。

    返回值含义：
        None。
    """

    # 这部分输入模拟上一轮已经保存在 Checkpoint 中的技能状态。
    eval_case = AgentEvaluationCase(
        case_id="skill_resume_output_test",
        category="skill_behavior",
        question="它目前会坐下，希望学习等待和召回。",
        input_state={
            "selected_skill_id": "dog-training-plan",
            "existing_inputs": {
                "breed": "Golden Retriever",
                "age": "6岁",
            },
        },
        expected={
            "status": "ready",
            "selection_source": "provided_skill_id",
        },
    )

    result = await SkillBehaviorEvaluator().evaluate_case(eval_case)

    assert result.passed is True
    assert result.output["selected_skill_id"] == "dog-training-plan"
    assert result.output["merged_inputs"] == {
        "breed": "Golden Retriever",
        "age": "6岁",
        "current_behavior": "坐下",
        "training_goal": "学习等待和召回",
    }
    assert result.output["skill_context_loaded"] is True


@pytest.mark.asyncio
async def test_skill_behavior_evaluator_should_reject_unknown_expected_field(
) -> None:
    """
    检查 Skill 评估器是否拒绝没有定义的黄金期望字段。

    参数含义：
        无。

    返回值含义：
        None。
    """

    eval_case = AgentEvaluationCase(
        case_id="skill_unknown_expected_field",
        category="skill_behavior",
        question="制定训练计划",
        expected={
            "unknown_field": True,
        },
    )

    result = await SkillBehaviorEvaluator().evaluate_case(eval_case)

    assert result.passed is False
    assert "不支持的 expected 字段" in str(result.error_message)
