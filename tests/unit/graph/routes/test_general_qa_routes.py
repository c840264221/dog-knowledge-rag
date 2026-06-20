"""
general_qa_agent routes 单元测试。

Route（路由）：
根据 LangGraph 当前 state 决定下一步走向的函数。

本文件测试：
1. route_general_qa_worker
2. route_after_executing_tool_worker
"""

import pytest

from src.agents.general_qa_agent.routes import (
    route_general_qa_worker,
    route_after_executing_tool_worker,
)


@pytest.mark.parametrize(
    "next_worker",
    [
        "finish",
        "tool_parse",
        "ask_confirm",
        "execute_tool",
        "answer_gen",
    ],
)
def test_route_general_qa_worker_should_return_valid_worker(
    next_worker,
):
    """
    测试 next_worker 是合法 worker 时，是否原样返回。

    参数：
        next_worker：
            参数化传入的合法 worker 名称。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "next_worker": next_worker,
    }

    result = route_general_qa_worker(
        state,
    )

    assert result == next_worker


def test_route_general_qa_worker_should_strip_worker_name():
    """
    测试 next_worker 前后有空格时，是否自动 strip。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "next_worker": "  tool_parse  ",
    }

    result = route_general_qa_worker(
        state,
    )

    assert result == "tool_parse"


def test_route_general_qa_worker_should_fallback_when_missing():
    """
    测试缺少 next_worker 时，是否兜底到 answer_gen。

    fallback（兜底）：
    当 state 缺少字段时，不让 Graph 直接报错，
    而是返回一个安全的默认路由。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {}

    result = route_general_qa_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_general_qa_worker_should_fallback_when_none():
    """
    测试 next_worker 为 None 时，是否兜底到 answer_gen。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "next_worker": None,
    }

    result = route_general_qa_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_general_qa_worker_should_fallback_when_empty_string():
    """
    测试 next_worker 为空字符串时，是否兜底到 answer_gen。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "next_worker": "",
    }

    result = route_general_qa_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_general_qa_worker_should_fallback_when_invalid_worker():
    """
    测试 next_worker 是非法 worker 时，是否兜底到 answer_gen。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "next_worker": "unknown_worker",
    }

    result = route_general_qa_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_after_executing_tool_worker_should_return_ask_confirm_when_tool_calls_left():
    """
    测试还有剩余 tool_calls 时，是否返回 ask_confirm。

    场景：
        execute_tool_node 每次只执行第一个工具。
        如果还有剩余工具，就应该回到 ask_confirm 继续确认下一次工具调用。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "tool_calls": [
            {
                "name": "weather",
                "args": {
                    "city": "北京",
                },
            }
        ],
    }

    result = route_after_executing_tool_worker(
        state,
    )

    assert result == "ask_confirm"


def test_route_after_executing_tool_worker_should_return_answer_gen_when_no_tool_calls_left():
    """
    测试 tool_calls 为空列表时，是否返回 answer_gen。

    场景：
        所有工具都执行完后，应该进入 answer_gen 生成最终回答。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "tool_calls": [],
    }

    result = route_after_executing_tool_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_after_executing_tool_worker_should_return_answer_gen_when_tool_calls_missing():
    """
    测试缺少 tool_calls 字段时，是否安全兜底到 answer_gen。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {}

    result = route_after_executing_tool_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_after_executing_tool_worker_should_return_answer_gen_when_tool_calls_none():
    """
    测试 tool_calls 为 None 时，是否安全兜底到 answer_gen。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "tool_calls": None,
    }

    result = route_after_executing_tool_worker(
        state,
    )

    assert result == "answer_gen"


def test_route_after_executing_tool_worker_should_return_answer_gen_when_tool_calls_invalid_type():
    """
    测试 tool_calls 类型非法时，是否安全兜底到 answer_gen。

    场景：
        tool_calls 正确类型应该是 list。
        如果误传成字符串，不能用 len 字符串判断，否则会错误进入 ask_confirm。

    参数：
        无。

    返回值：
        None：
            pytest 会根据 assert 判断测试是否通过。
    """

    state = {
        "tool_calls": "weather",
    }

    result = route_after_executing_tool_worker(
        state,
    )

    assert result == "answer_gen"