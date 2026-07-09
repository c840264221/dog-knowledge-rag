"""
V1.8.0 ToolAgent 契约冒烟脚本。

功能：
    使用 mock state 验证 ToolAgentResponse 契约和旧 state 适配器是否稳定。

运行方式：
    python -m scripts.smoke_v180_tool_agent_contract
"""

from __future__ import annotations

from src.agents.tool_agent.smoke.v180_smoke_checks import (
    assert_v180_tool_agent_smoke_checks,
)


def main() -> int:
    """
    命令行入口函数。

    功能：
        执行 V1.8.0 ToolAgent 契约冒烟检查，并打印结果摘要。

    参数：
        无。

    返回值：
        int:
            0 表示全部通过；如果失败会抛出 AssertionError。
    """

    results = assert_v180_tool_agent_smoke_checks()

    print("# V1.8.0 工具Agent冒烟测试简易报告")
    print("")
    # 因为调用的函数中用使用了断言 所以如果有错误的话会直接抛出异常 不会走到下面打印这步
    # 并不会影响判断冒烟测试成功与否
    print("- 状态: PASS")
    print(f"- 测试用例数量: {len(results)}")
    print("")
    print("## 测试用例")

    for result in results:
        print(
            f"- {result.case_name}: {result.status}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
