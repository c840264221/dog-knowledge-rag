"""DogState 多 Agent 跨轮恢复字段测试。"""

from src.graph.graph_run import create_initial_state
from src.graph.states.dog_state import DogState


def test_dog_state_should_include_multi_agent_resume_fields() -> None:
    """
    检查 DogState 是否声明多 Agent 暂停与恢复所需字段。

    参数含义：无。
    返回值含义：None。
    """

    expected_fields = {
        "multi_agent_task_result",
        "multi_agent_resume_action",
        "multi_agent_resume_inputs",
        "multi_agent_resume_ready",
        "multi_agent_pending_prompt",
    }

    assert expected_fields <= set(DogState.__annotations__)


def test_initial_state_should_reset_multi_agent_resume_fields(
    monkeypatch,
) -> None:
    """
    检查每轮干净初始状态不会直接继承上一轮多 Agent 临时数据。

    参数含义：
        monkeypatch:
            pytest 临时替换工具，用来固定测试用户编号。

    返回值含义：
        None。
    """

    monkeypatch.setattr(
        "src.graph.graph_run.get_user_id",
        lambda: "multi_agent_test_user",
    )

    state = create_initial_state(
        question="继续任务",
        trace_id="multi_agent_trace",
    )

    assert state["multi_agent_task_result"] == {}
    assert state["multi_agent_resume_action"] == "none"
    assert state["multi_agent_resume_inputs"] == {}
    assert state["multi_agent_resume_ready"] is False
    assert state["multi_agent_pending_prompt"] == ""
