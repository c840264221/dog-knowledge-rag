from __future__ import annotations

import asyncio

from scripts.evaluation.evaluate_v116_multi_agent_behavior import (
    run_multi_agent_behavior_evaluation,
)


def main() -> None:
    """
    运行 V1.17 多 Agent 高风险韧性边界评估。

    功能：
        复用统一多 Agent 行为评估器，展示包含运行中取消、超时耗尽、
        重试耗尽和恢复输入校验的 V1.17 黄金集成绩。

    参数含义：
        无。

    返回值含义：
        None:
            使用进程退出码表示全部多 Agent 行为用例是否通过。
    """

    exit_code = asyncio.run(
        run_multi_agent_behavior_evaluation(
            report_title=(
                "V1.17 Multi-Agent Resilience "
                "Behavior Evaluation Report"
            ),
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
