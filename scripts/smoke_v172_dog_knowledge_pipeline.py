"""
V1.7.2 DogKnowledgeAgent 真实主图 Smoke Test 脚本。

功能：
    用真实主图执行一个狗狗知识问题，
    验证 RootAgent -> DogKnowledgeAgent -> Entry Adapter 链路是否跑通。

使用方式：
    python scripts/smoke_v172_dog_knowledge_pipeline.py

专业名词：
    Smoke Test：冒烟测试，用最小真实调用验证主链路是否能跑通。
    Main Graph：主图，表示 LangGraph 顶层工作流。
    State：状态，表示图执行过程中共享的数据。
    Container：容器，统一管理 Provider / Service 的生命周期。
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.dog_knowledge_agent.smoke.v172_smoke_checks import (
    render_dog_knowledge_smoke_report_markdown,
    validate_dog_knowledge_smoke_state,
)

from src.agents.dog_knowledge_agent.debug.debug_report_smoke_checks import (
    render_dog_knowledge_debug_report_smoke_markdown,
    validate_dog_knowledge_debug_report_smoke_state,
)
from src.runtime.services.checkpoint_config import (
    build_graph_checkpoint_config,
)


async def build_or_get_main_graph() -> Any:
    """
    构建或获取当前项目真实主图。

    功能：
        1. 启动运行时容器。
        2. 从容器中获取 GraphRuntimeService。
        3. 返回已经编译完成、可以执行 ainvoke 的真实主图对象。

    参数：
        无。

    返回值：
        Any:
            当前项目真实主图对象。
    """

    from src.runtime.container.init import container

    await container.startup()

    graph_runtime = container.get(
        "graph_runtime",
    )

    return graph_runtime.graph


async def shutdown_runtime_container() -> None:
    """
    关闭运行时容器。

    功能：
        在 smoke test 执行结束后关闭容器，
        释放 GraphRuntimeService、Provider、checkpointer 等运行时资源。

    参数：
        无。

    返回值：
        None。
    """

    from src.runtime.container.init import container

    await container.shutdown()


def build_smoke_input_state() -> dict[str, Any]:
    """
    构建 Smoke Test 输入状态。

    功能：
        构建一个狗狗知识类问题，
        用于触发 RootAgent 路由到 dog_knowledge_agent。

    参数：
        无。

    返回值：
        dict[str, Any]:
            LangGraph 输入状态。
    """

    return {
        "question": "金毛适合新手养吗？请结合运动量、性格和掉毛情况回答。",
        "user_id": "smoke_user_v172",
        "session_id": "smoke_session_v172",
        "trace_id": "smoke_trace_v172",
    }


def build_smoke_graph_config() -> dict[str, Any]:
    """
    构建 Smoke Test 主图运行配置。

    功能：
        为带 checkpointer 的 LangGraph 主图提供 configurable 配置。
        当前至少需要 thread_id，否则 AsyncSqliteSaver 无法保存检查点。

    参数：
        无。

    返回值：
        dict[str, Any]:
            LangGraph graph.ainvoke 使用的 config 配置。
    """

    return build_graph_checkpoint_config(
        thread_id="smoke_v172_dog_knowledge_pipeline",
        checkpoint_ns="dog_knowledge_pipeline_smoke",
    )


async def run_smoke_test() -> int:
    """
    执行 V1.7.2 DogKnowledgeAgent Smoke Test。

    功能：
        1. 构建真实主图。
        2. 执行狗狗知识类问题。
        3. 检查最终 state 是否包含 dog_knowledge_pipeline_* metadata。
        4. 打印 Markdown 报告。
        5. 根据检查结果返回退出码。

    参数：
        无。

    返回值：
        int:
            0 表示 smoke test 通过。
            1 表示 smoke test 失败。
    """

    try:
        graph = await build_or_get_main_graph()

        input_state = build_smoke_input_state()

        graph_config = build_smoke_graph_config()

        result_state = await graph.ainvoke(
            input_state,
            config=graph_config,
        )

        pipeline_smoke_result = validate_dog_knowledge_smoke_state(
            state=result_state,
        )

        debug_report_smoke_result = validate_dog_knowledge_debug_report_smoke_state(
            state=result_state,
        )

        print(
            render_dog_knowledge_smoke_report_markdown(
                result=pipeline_smoke_result,
            )
        )

        print()
        print(
            render_dog_knowledge_debug_report_smoke_markdown(
                result=debug_report_smoke_result,
                state=result_state,
            )
        )

        if pipeline_smoke_result.passed and debug_report_smoke_result.passed:
            return 0

        return 1

    finally:
        await shutdown_runtime_container()


def main() -> int:
    """
    脚本入口函数。

    功能：
        运行异步 smoke test。

    参数：
        无。

    返回值：
        int:
            0 表示通过。
            1 表示失败。
    """

    return asyncio.run(
        run_smoke_test(),
    )


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )

