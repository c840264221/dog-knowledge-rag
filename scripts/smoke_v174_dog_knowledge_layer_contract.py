from __future__ import annotations

import argparse
import asyncio
from typing import Any

from src.agents.dog_knowledge_agent.smoke.v174_smoke_checks import (
    render_dog_knowledge_layer_contract_smoke_markdown,
    validate_dog_knowledge_layer_contract_state,
)
from src.runtime.services.checkpoint_config import (
    build_graph_checkpoint_config,
)


def build_fake_layer_contract_state() -> dict[str, Any]:
    """
    构建 V1.7.4 分层契约 fake smoke state。

    功能：
        构造一个不依赖 LLM / RAG / 主图的模拟最终 state，
        用于快速验证分层契约检查函数是否可用。

    参数含义：
        无。

    返回值含义：
        dict[str, Any]:
            包含 V1.7.4 分层契约字段的模拟 state。
    """

    return {
        "question": "金毛适合新手养吗？",
        "dog_query_result": {
            "question": "金毛适合新手养吗？",
            "query_type": "exact_lookup",
            "dog_names": ["golden_retriever"],
            "target_fields": ["temperament"],
            "confidence": 0.8,
        },
        "dog_retrieval_result": {
            "query_type": "exact_lookup",
            "evidences": [
                {
                    "evidence_id": "evidence_1",
                    "source": "fake",
                    "content": "金毛通常友好、亲人，适合愿意投入训练的新手。",
                    "score": 0.9,
                }
            ],
            "retrieved_count": 1,
            "confidence": 0.9,
        },
        "dog_generation_result": {
            "generated_answer": "金毛通常适合新手，但需要稳定运动和基础训练。",
            "confidence": 0.85,
            "used_evidence_ids": ["evidence_1"],
        },
        "dog_knowledge_pipeline_result": {
            "question": "金毛适合新手养吗？",
            "query_type": "exact_lookup",
            "status": "answered",
            "answer": "金毛通常适合新手，但需要稳定运动和基础训练。",
            "recommended_breeds": [],
            "evidences": [
                {
                    "evidence_id": "evidence_1",
                    "source": "fake",
                    "content": "金毛通常友好、亲人，适合愿意投入训练的新手。",
                    "score": 0.9,
                }
            ],
            "confidence": 0.85,
            "metadata": {
                "source": "fake_smoke",
            },
        },
        "dog_knowledge_answer": {
            "status": "answered",
            "answer": "金毛通常适合新手，但需要稳定运动和基础训练。",
            "confidence": 0.85,
        },
        "dog_knowledge_answer_public": {
            "status": "answered",
            "answer": "金毛通常适合新手，但需要稳定运动和基础训练。",
            "confidence": 0.85,
        },
        "final_answer": "金毛通常适合新手，但需要稳定运动和基础训练。",
    }


async def run_graph_mode() -> dict[str, Any]:
    """
    运行真实主图并返回最终 state。

    功能：
        启动 runtime container，获取真实主图，执行一个狗狗知识问题，
        用于验证真实链路是否产出 V1.7.4 分层契约字段。

    参数含义：
        无。

    返回值含义：
        dict[str, Any]:
            真实主图执行后的最终 state。
    """

    from src.runtime.container.init import container

    await container.startup()

    try:
        graph_runtime = container.get("graph_runtime")
        graph = graph_runtime.graph

        return await graph.ainvoke(
            {
                "question": "金毛适合新手养吗？请结合性格和运动量回答。",
                "user_id": "smoke_user_v174",
                "session_id": "smoke_session_v174",
                "trace_id": "smoke_trace_v174",
            },
            config=build_graph_checkpoint_config(
                thread_id="smoke_v174_dog_knowledge_layer_contract",
                checkpoint_ns="dog_knowledge_layer_contract_smoke",
            ),
        )
    finally:
        await container.shutdown()


async def run_smoke(mode: str) -> int:
    """
    执行 V1.7.4 分层契约 smoke test。

    功能：
        根据 mode 选择 fake state 或真实主图执行结果，然后调用检查函数，
        打印 Markdown 报告，并返回命令行退出码。

    参数含义：
        mode:
            运行模式。fake 表示使用模拟 state；graph 表示运行真实主图。

    返回值含义：
        int:
            0 表示通过；1 表示失败。
    """

    if mode == "graph":
        state = await run_graph_mode()
    else:
        state = build_fake_layer_contract_state()

    result = validate_dog_knowledge_layer_contract_state(state)

    print(
        render_dog_knowledge_layer_contract_smoke_markdown(result)
    )

    return 0 if result.passed else 1


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    功能：
        读取 smoke 脚本运行模式。

    参数含义：
        无。

    返回值含义：
        argparse.Namespace:
            包含 mode 字段的命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="V1.7.4 DogKnowledgeAgent layer contract smoke test."
    )
    parser.add_argument(
        "--mode",
        choices=("fake", "graph"),
        default="fake",
        help="fake 使用模拟 state；graph 运行真实主图。",
    )

    return parser.parse_args()


def main() -> int:
    """
    脚本入口函数。

    功能：
        解析命令行参数并运行异步 smoke test。

    参数含义：
        无。

    返回值含义：
        int:
            0 表示通过；1 表示失败。
    """

    args = parse_args()

    return asyncio.run(
        run_smoke(mode=args.mode),
    )


if __name__ == "__main__":
    raise SystemExit(main())

