"""任务关系门卫节点单元测试。"""

from __future__ import annotations

import pytest

from src.graph.nodes.task_relation_guard_node import (
    task_relation_guard_node,
)


@pytest.mark.asyncio
async def test_guard_should_keep_raw_input_for_regular_question() -> None:
    """
    验证普通问题保留原始输入并作为记忆抽取文本。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": "我喜欢金毛。",
            "question": "我喜欢金毛。",
        }
    )

    assert result["raw_user_input"] == "我喜欢金毛。"
    assert result["memory_source_text"] == "我喜欢金毛。"
    assert result["memory_retrieval_text"] == "我喜欢金毛。"
    assert result["task_relation_guard_processed"] is True


@pytest.mark.asyncio
async def test_guard_should_normalize_new_task_before_memory() -> None:
    """
    验证新任务控制前缀不会进入记忆抽取文本。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": "新问题：帮我查成都天气。",
            "question": "新问题：帮我查成都天气。",
            "multi_agent_task_result": {
                "status": "awaiting_input",
            },
        }
    )

    assert result["raw_user_input"] == "新问题：帮我查成都天气。"
    assert result["question"] == "帮我查成都天气。"
    assert result["memory_source_text"] == "帮我查成都天气。"
    assert result["memory_retrieval_text"] == "帮我查成都天气。"
    assert result["task_relation_decision"]["relation"] == "new_task"
    assert result["multi_agent_task_result"] == {}


@pytest.mark.asyncio
async def test_guard_should_add_pending_context_for_skill_resume() -> None:
    """
    验证简短 Skill 补充输入会携带旧任务问题进入记忆抽取。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": "继续任务：6岁",
            "question": "继续任务：6岁",
            "skill_status": "awaiting_input",
            "skill_pending_prompt": "请补充狗狗年龄。",
        }
    )

    assert result["question"] == "6岁"
    assert result["memory_source_text"] == (
        "旧任务正在询问：请补充狗狗年龄。\n"
        "用户本轮补充：6岁"
    )
    assert result["memory_retrieval_text"] == result["memory_source_text"]
    assert result["task_relation_decision"]["relation"] == "resume"


@pytest.mark.asyncio
@pytest.mark.parametrize("user_input", ["取消", "成都天气"])
async def test_guard_should_skip_memory_for_control_or_ambiguous_input(
    user_input: str,
) -> None:
    """
    验证取消和模糊输入不会触发没有业务价值的记忆抽取。

    参数含义：
        user_input:
            本轮取消或无法判断关系的用户文本。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": user_input,
            "question": user_input,
            "multi_agent_task_result": {
                "status": "awaiting_input",
            },
        }
    )

    assert result["memory_source_text"] == ""
    assert result["memory_retrieval_text"] == ""
    assert result["task_relation_decision"]["relation"] in {
        "cancel",
        "ambiguous",
    }
    assert result["route_decision"]["route"] == "FINISH"
    assert result["route_decision"]["requires_memory"] is False


@pytest.mark.asyncio
async def test_guard_should_record_degraded_skill_choice_without_memory() -> None:
    """
    验证简化执行会写入审计状态，但不会作为长期记忆抽取。

    参数含义：无。
    返回值含义：None，断言失败时由 pytest 报错。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": "简化执行",
            "question": "简化执行",
            "skill_status": "awaiting_input",
            "skill_selected_id": "dog-training-plan",
            "skill_inputs": {
                "age": "6岁",
                "training_goal": "学习等待",
            },
            "skill_runtime_result": {
                "input_check": {
                    "can_run_degraded": True,
                    "missing_degradable_input_ids": [
                        "breed",
                        "current_behavior",
                    ],
                }
            },
        }
    )

    assert result["task_relation_decision"]["relation"] == "resume"
    assert result["skill_execution_mode"] == "degraded"
    assert result["skill_ignored_input_ids"] == [
        "breed",
        "current_behavior",
    ]
    assert result["skill_degradation_user_input"] == "简化执行"
    assert result["memory_source_text"] == ""
    assert result["memory_retrieval_text"] == ""


@pytest.mark.asyncio
async def test_guard_should_skip_memory_for_multi_agent_degraded_control() -> None:
    """
    验证多智能体简化执行控制指令不会进入记忆抽取。

    参数含义：无。
    返回值含义：None，pytest 根据记忆专用文本是否为空判断是否通过。
    """

    result = await task_relation_guard_node(
        {
            "raw_user_input": "简化执行",
            "question": "简化执行",
            "multi_agent_task_result": {
                "status": "awaiting_input",
            },
        }
    )

    assert result["task_relation_decision"]["relation"] == "resume"
    assert result["task_relation_pending_kind"] == "multi_agent"
    assert result["memory_source_text"] == ""
    assert result["memory_retrieval_text"] == ""
