import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.agents.dog_knowledge_agent.response_adapter import (
    DogKnowledgeAgentResponseAdapter,
)
from src.agents.dog_knowledge_agent.schemas import (
    DogKnowledgeAnswer,
)


SmokeMode = Literal[
    "contract",
    "agent",
]

ExpectedQueryType = Literal[
    "exact_lookup",
    "recommendation",
    "comparison",
    "general_qa",
    "fallback",
]


@dataclass(frozen=True)
class DogKnowledgeSmokeCase:
    """
    DogKnowledgeAgent Smoke 测试用例。

    功能：
        用来描述一个 smoke case，包括用户问题、期望的问题类型、是否期望 fallback。

    字段说明：
        name:
            测试用例名称，方便日志输出。

        question:
            用户原始问题。

        expected_query_type:
            期望的标准问题类型。
            例如 exact_lookup、recommendation、fallback。

        expected_is_fallback:
            是否期望该用例走 fallback。

        min_confidence:
            最低置信度要求。
            这个值不是严格评估分数，只是 smoke 阶段用来防止明显异常。
    """

    name: str
    question: str
    expected_query_type: ExpectedQueryType
    expected_is_fallback: bool
    min_confidence: float = 0.0


def build_smoke_cases() -> list[DogKnowledgeSmokeCase]:
    """
    构建 v1.7.3 DogKnowledgeAgent smoke 用例。

    功能：
        覆盖三类核心场景：
        1. 精确查询 exact_lookup。
        2. 推荐查询 recommendation。
        3. 兜底 fallback。

    参数：
        无。

    返回值：
        list[DogKnowledgeSmokeCase]:
            smoke 测试用例列表。
    """

    return [
        DogKnowledgeSmokeCase(
            name="exact_lookup_golden_retriever_lifespan",
            question="金毛寿命多久？",
            expected_query_type="exact_lookup",
            expected_is_fallback=False,
            min_confidence=0.3,
        ),
        DogKnowledgeSmokeCase(
            name="recommendation_beginner_friendly_dogs",
            question="新手适合养什么狗？",
            expected_query_type="recommendation",
            expected_is_fallback=False,
            min_confidence=0.3,
        ),
        DogKnowledgeSmokeCase(
            name="fallback_mars_living_dog",
            question="哪种狗适合在火星生活？",
            expected_query_type="fallback",
            expected_is_fallback=True,
            min_confidence=0.0,
        ),
    ]


def build_contract_pipeline_result(
    case: DogKnowledgeSmokeCase,
) -> dict[str, Any]:
    """
    构建 contract 模式下的模拟 pipeline_result。

    功能：
        不调用真实 DogKnowledgeAgent，只构造类似内部 pipeline 的结果，
        用来验证 Response Adapter 和 Formatter 是否能生成标准 DogKnowledgeAnswer。

    参数：
        case:
            当前 smoke 测试用例。

    返回值：
        dict[str, Any]:
            模拟的 DogKnowledgeAgent pipeline_result。
    """

    if case.expected_query_type == "exact_lookup":
        return {
            "question": case.question,
            "query_type": "exact_lookup",
            "answer": "金毛寻回犬的寿命通常在 10 到 12 年左右。",
            "retrieved_chunks": [
                {
                    "chunk_id": "golden-retriever-lifespan-001",
                    "source_kind": "rag_chunk",
                    "title": "Golden Retriever",
                    "content": "Golden Retrievers usually live around 10 to 12 years.",
                    "score": 0.9,
                    "metadata": {
                        "dog_name": "golden_retriever",
                        "field": "lifespan",
                    },
                }
            ],
            "confidence": 0.85,
            "reason": "命中了 Golden Retriever 的寿命相关知识片段。",
        }

    if case.expected_query_type == "recommendation":
        return {
            "question": case.question,
            "intent": "recommend",
            "recommendations": [
                {
                    "dog_name": "labrador_retriever",
                    "display_name": "Labrador Retriever / 拉布拉多寻回犬",
                    "reason": "性格友好，训练难度相对较低，通常比较适合新手家庭。",
                    "matched_traits": [
                        "新手友好",
                        "容易训练",
                        "家庭友好",
                    ],
                    "warnings": [
                        "运动量较高，需要规律遛狗。",
                    ],
                    "evidence_ids": [
                        "labrador-retriever-001",
                    ],
                    "score": 0.88,
                    "metadata": {
                        "energy": "high",
                        "trainability": "high",
                    },
                }
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": "labrador-retriever-001",
                    "source_kind": "rag_chunk",
                    "title": "Labrador Retriever",
                    "content": "Labrador Retrievers are friendly and trainable family dogs.",
                    "score": 0.86,
                    "metadata": {
                        "dog_name": "labrador_retriever",
                    },
                }
            ],
        }

    return {
        "question": case.question,
        "is_fallback": True,
        "fallback_reason": "问题超出当前 DogKnowledgeAgent 犬种知识库的可靠边界。",
        "debug": {
            "retrieved_chunks": 0,
            "route": "fallback",
        },
    }


def assert_answer_contract(
    answer: DogKnowledgeAnswer,
    case: DogKnowledgeSmokeCase,
) -> None:
    """
    校验 DogKnowledgeAnswer 是否符合 v1.7.3 输出协议。

    参数：
        answer:
            DogKnowledgeAgent 最终标准答案对象。

        case:
            当前 smoke 测试用例。

    返回值：
        None。
        如果校验失败，会抛出 AssertionError。
    """

    assert isinstance(answer, DogKnowledgeAnswer), (
        f"{case.name} 失败：dog_knowledge_answer 必须是 DogKnowledgeAnswer。"
    )

    assert answer.question == case.question, (
        f"{case.name} 失败：question 不一致，"
        f"expected={case.question}, actual={answer.question}"
    )

    assert answer.query_type == case.expected_query_type, (
        f"{case.name} 失败：query_type 不符合预期，"
        f"expected={case.expected_query_type}, actual={answer.query_type}"
    )

    assert answer.is_fallback is case.expected_is_fallback, (
        f"{case.name} 失败：is_fallback 不符合预期，"
        f"expected={case.expected_is_fallback}, actual={answer.is_fallback}"
    )

    assert isinstance(answer.answer, str) and answer.answer.strip(), (
        f"{case.name} 失败：answer 必须是非空字符串。"
    )

    assert 0.0 <= answer.confidence <= 1.0, (
        f"{case.name} 失败：confidence 必须在 0 到 1 之间，actual={answer.confidence}"
    )

    assert answer.confidence >= case.min_confidence, (
        f"{case.name} 失败：confidence 低于 smoke 最低要求，"
        f"min={case.min_confidence}, actual={answer.confidence}"
    )

    if case.expected_query_type == "recommendation":
        assert answer.has_recommendations(), (
            f"{case.name} 失败：推荐类问题必须包含 recommended_breeds。"
        )

    if case.expected_query_type == "exact_lookup":
        assert answer.status in {"success", "partial"}, (
            f"{case.name} 失败：精确查询类问题 status 应该是 success 或 partial，"
            f"actual={answer.status}"
        )

    if case.expected_is_fallback:
        assert answer.status == "fallback", (
            f"{case.name} 失败：fallback case 的 status 必须是 fallback，"
            f"actual={answer.status}"
        )

        assert answer.fallback_reason, (
            f"{case.name} 失败：fallback case 必须包含 fallback_reason。"
        )


def assert_public_contract(
    public_answer: dict[str, Any],
    case: DogKnowledgeSmokeCase,
) -> None:
    """
    校验 dog_knowledge_answer_public 是否符合对外输出协议。

    参数：
        public_answer:
            DogKnowledgeAnswer.to_public_dict() 生成的对外字典。

        case:
            当前 smoke 测试用例。

    返回值：
        None。
        如果校验失败，会抛出 AssertionError。
    """

    required_keys = {
        "question",
        "query_type",
        "status",
        "answer",
        "recommended_breeds",
        "evidences",
        "confidence",
        "is_fallback",
        "metadata",
    }

    missing_keys = required_keys - set(public_answer.keys())

    assert not missing_keys, (
        f"{case.name} 失败：dog_knowledge_answer_public 缺少字段：{missing_keys}"
    )

    assert public_answer["question"] == case.question, (
        f"{case.name} 失败：public question 不一致。"
    )

    assert public_answer["query_type"] == case.expected_query_type, (
        f"{case.name} 失败：public query_type 不符合预期。"
    )

    assert public_answer["is_fallback"] is case.expected_is_fallback, (
        f"{case.name} 失败：public is_fallback 不符合预期。"
    )

    assert isinstance(public_answer["answer"], str) and public_answer["answer"].strip(), (
        f"{case.name} 失败：public answer 必须是非空字符串。"
    )

    assert "debug" not in public_answer, (
        f"{case.name} 失败：public dict 默认不应该暴露 debug 字段。"
    )


def assert_agent_smoke_contract(
    answer: DogKnowledgeAnswer,
    public_answer: dict[str, Any],
    final_answer: str,
    case: DogKnowledgeSmokeCase,
) -> list[str]:
    """
    校验真实 agent 模式下的阶段性输出契约。

    功能：
        agent 模式只强制校验真实链路当前必须稳定的基础字段。
        对 query_type、recommended_breeds、fallback_reason 等未来收敛目标，
        只记录 warning，不直接让 smoke 失败。

    参数：
        answer:
            真实 DogKnowledgeAgent 返回的标准答案对象。

        public_answer:
            对外 public dict 输出。

        final_answer:
            兼容旧上层调用的最终自然语言答案。

        case:
            当前 smoke 测试用例。

    返回值：
        list[str]:
            阶段性契约缺口 warning 列表。
    """

    allowed_query_types = {
        "exact_lookup",
        "recommendation",
        "comparison",
        "general_qa",
        "fallback",
    }

    allowed_statuses = {
        "success",
        "partial",
        "fallback",
        "empty",
        "error",
    }

    warnings: list[str] = []

    assert isinstance(answer, DogKnowledgeAnswer), (
        f"{case.name} 失败：dog_knowledge_answer 必须是 DogKnowledgeAnswer。"
    )

    assert answer.question == case.question, (
        f"{case.name} 失败：question 不一致，"
        f"expected={case.question}, actual={answer.question}"
    )

    assert answer.query_type in allowed_query_types, (
        f"{case.name} 失败：query_type 不在允许范围内，actual={answer.query_type}"
    )

    assert answer.status in allowed_statuses, (
        f"{case.name} 失败：status 不在允许范围内，actual={answer.status}"
    )

    assert isinstance(answer.answer, str) and answer.answer.strip(), (
        f"{case.name} 失败：answer 必须是非空字符串。"
    )

    assert 0.0 <= answer.confidence <= 1.0, (
        f"{case.name} 失败：confidence 必须在 0 到 1 之间，actual={answer.confidence}"
    )

    assert isinstance(final_answer, str) and final_answer.strip(), (
        f"{case.name} 失败：final_answer 必须是非空字符串。"
    )

    assert final_answer == answer.answer, (
        f"{case.name} 失败：final_answer 必须与 dog_knowledge_answer.answer 保持一致。"
    )

    assert_agent_public_smoke_contract(
        public_answer=public_answer,
        answer=answer,
        case=case,
    )

    if answer.query_type != case.expected_query_type:
        warnings.append(
            f"{case.name}: query_type 尚未完全收敛，"
            f"expected={case.expected_query_type}, actual={answer.query_type}"
        )

    if answer.is_fallback is not case.expected_is_fallback:
        warnings.append(
            f"{case.name}: is_fallback 尚未完全收敛，"
            f"expected={case.expected_is_fallback}, actual={answer.is_fallback}"
        )

    if case.expected_query_type == "recommendation" and not answer.has_recommendations():
        warnings.append(
            f"{case.name}: 推荐场景暂未产出 recommended_breeds，"
            "属于后续 DogKnowledgeAgent pipeline 收敛目标。"
        )

    if case.expected_is_fallback and not answer.fallback_reason:
        warnings.append(
            f"{case.name}: fallback 场景暂未产出 fallback_reason，"
            "属于后续 fallback 决策收敛目标。"
        )

    if answer.confidence < case.min_confidence:
        warnings.append(
            f"{case.name}: confidence 低于当前严格 smoke 预期，"
            f"min={case.min_confidence}, actual={answer.confidence}"
        )

    return warnings


def assert_agent_public_smoke_contract(
    public_answer: dict[str, Any],
    answer: DogKnowledgeAnswer,
    case: DogKnowledgeSmokeCase,
) -> None:
    """
    校验真实 agent 模式下的 public dict 基础契约。

    功能：
        只校验 public dict 是否具备基础字段、是否与 DogKnowledgeAnswer 保持一致，
        不强制要求 query_type / is_fallback 必须等于当前 case 的未来预期。

    参数：
        public_answer:
            DogKnowledgeAnswer.to_public_dict() 生成的对外字典。

        answer:
            标准答案对象。

        case:
            当前 smoke 测试用例。

    返回值：
        None。
    """

    required_keys = {
        "question",
        "query_type",
        "status",
        "answer",
        "recommended_breeds",
        "evidences",
        "confidence",
        "is_fallback",
        "metadata",
    }

    missing_keys = required_keys - set(public_answer.keys())

    assert not missing_keys, (
        f"{case.name} 失败：dog_knowledge_answer_public 缺少字段：{missing_keys}"
    )

    assert public_answer["question"] == answer.question, (
        f"{case.name} 失败：public question 必须与 answer.question 一致。"
    )

    assert public_answer["query_type"] == answer.query_type, (
        f"{case.name} 失败：public query_type 必须与 answer.query_type 一致。"
    )

    assert public_answer["status"] == answer.status, (
        f"{case.name} 失败：public status 必须与 answer.status 一致。"
    )

    assert public_answer["is_fallback"] is answer.is_fallback, (
        f"{case.name} 失败：public is_fallback 必须与 answer.is_fallback 一致。"
    )

    assert isinstance(public_answer["answer"], str) and public_answer["answer"].strip(), (
        f"{case.name} 失败：public answer 必须是非空字符串。"
    )

    assert "debug" not in public_answer, (
        f"{case.name} 失败：public dict 默认不应该暴露 debug 字段。"
    )


def run_contract_smoke_case(
    case: DogKnowledgeSmokeCase,
    output_dir: Path,
) -> dict[str, Any]:
    """
    运行单个 contract smoke case。

    功能：
        使用模拟 pipeline_result 测试 Response Adapter 的最终输出协议。

    参数：
        case:
            当前 smoke 测试用例。

        output_dir:
            smoke 输出目录，用于保存 public answer JSON。

    返回值：
        dict[str, Any]:
            当前 case 的简要结果。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    pipeline_result = build_contract_pipeline_result(case)

    answer = adapter.finalize(
        pipeline_result=pipeline_result,
        question=case.question,
        as_public_dict=False,
    )

    assert isinstance(answer, DogKnowledgeAnswer)

    public_answer = adapter.finalize(
        pipeline_result=pipeline_result,
        question=case.question,
        include_debug=False,
        as_public_dict=True,
    )

    assert isinstance(public_answer, dict)

    assert_answer_contract(
        answer=answer,
        case=case,
    )
    assert_public_contract(
        public_answer=public_answer,
        case=case,
    )

    save_public_answer(
        output_dir=output_dir,
        case_name=case.name,
        public_answer=public_answer,
    )

    return {
        "name": case.name,
        "question": case.question,
        "query_type": answer.query_type,
        "status": answer.status,
        "confidence": answer.confidence,
        "is_fallback": answer.is_fallback,
        "final_answer": answer.answer,
        "warnings": [],
    }


def build_initial_agent_state(
    question: str,
) -> dict[str, Any]:
    """
    构建真实 DogKnowledgeAgent graph 的初始 state。

    功能：
        尽量兼容不同版本的 DogKnowledgeAgent state。
        如果你的 v1.7.2/v1.7.3 graph 只接受固定字段，
        可以只保留你项目 DogState 中存在的字段。

    参数：
        question:
            用户原始问题。

    返回值：
        dict[str, Any]:
            初始 graph state。
    """

    return {
        "question": question,
        "user_question": question,
        "input": question,
        "user_input": question,
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
    }


async def build_real_dog_knowledge_agent() -> Any:
    """
    构建真实 DogKnowledgeAgent。

    功能：
        从当前项目中导入 build_dog_knowledge_agent。
        如果你的函数名或路径不同，只需要改这里。

    参数：
        无。

    返回值：
        Any:
            已编译或可调用的 DogKnowledgeAgent graph / runnable。
    """

    from src.agents.dog_knowledge_agent.agent import (
        build_dog_knowledge_agent,
    )
    from src.runtime.container.init import (
        container,
    )

    await container.startup()

    return build_dog_knowledge_agent(
        llm_provider=container.get("llm"),
        memory_provider=container.get("memory"),
        checkpoint_provider=container.get("checkpoint"),
        retriever_provider=container.get("retriever"),
        reranker_provider=container.get("reranker"),
    )


async def shutdown_runtime_container() -> None:
    """
    关闭 Runtime Container。

    功能：
        在 agent smoke 模式结束后，统一关闭 container 中的 provider / service。

    参数：
        无。

    返回值：
        None。
    """

    from src.runtime.container.init import (
        container,
    )

    await container.shutdown()


async def invoke_real_agent(
    agent: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    调用真实 DogKnowledgeAgent。

    功能：
        兼容 LangGraph 的 ainvoke / invoke，也兼容普通 async callable / sync callable。

    参数：
        agent:
            DogKnowledgeAgent graph、Runnable 或普通函数。

        state:
            初始输入 state。

    返回值：
        dict[str, Any]:
            DogKnowledgeAgent 最终输出 state。
    """

    if hasattr(agent, "ainvoke"):
        result = await agent.ainvoke(state)

    elif hasattr(agent, "invoke"):
        result = agent.invoke(state)

    elif callable(agent):
        result = agent(state)

        if asyncio.iscoroutine(result):
            result = await result

    else:
        raise TypeError("无法调用 DogKnowledgeAgent：agent 既不是 Runnable，也不是 callable。")

    if not isinstance(result, dict):
        raise TypeError(f"DogKnowledgeAgent 返回值必须是 dict，actual={type(result)}")

    return result


def extract_answer_from_agent_state(
    state: dict[str, Any],
    case: DogKnowledgeSmokeCase,
) -> tuple[DogKnowledgeAnswer, dict[str, Any], str]:
    """
    从真实 DogKnowledgeAgent 最终 state 中提取标准答案。

    参数：
        state:
            DogKnowledgeAgent 最终 state。

        case:
            当前 smoke 测试用例。

    返回值：
        tuple[DogKnowledgeAnswer, dict[str, Any], str]:
            DogKnowledgeAnswer 对象、public dict、final_answer 字符串。
    """

    answer = state.get("dog_knowledge_answer")
    public_answer = state.get("dog_knowledge_answer_public")
    final_answer = state.get("final_answer")

    assert isinstance(answer, DogKnowledgeAnswer), (
        f"{case.name} 失败：真实 agent 最终 state 缺少 dog_knowledge_answer，"
        "或者 dog_knowledge_answer 不是 DogKnowledgeAnswer。"
    )

    assert isinstance(public_answer, dict), (
        f"{case.name} 失败：真实 agent 最终 state 缺少 dog_knowledge_answer_public。"
    )

    assert isinstance(final_answer, str) and final_answer.strip(), (
        f"{case.name} 失败：真实 agent 最终 state 缺少 final_answer。"
    )

    return answer, public_answer, final_answer


async def run_agent_smoke_case(
    agent: Any,
    case: DogKnowledgeSmokeCase,
    output_dir: Path,
) -> dict[str, Any]:
    """
    运行单个真实 agent smoke case。

    功能：
        调用真实 DogKnowledgeAgent graph，验证最终输出协议。

    参数：
        agent:
            真实 DogKnowledgeAgent graph / runnable。

        case:
            当前 smoke 测试用例。

        output_dir:
            smoke 输出目录，用于保存 public answer JSON。

    返回值：
        dict[str, Any]:
            当前 case 的简要结果。
    """

    state = build_initial_agent_state(case.question)

    result_state = await invoke_real_agent(
        agent=agent,
        state=state,
    )

    answer, public_answer, final_answer = extract_answer_from_agent_state(
        state=result_state,
        case=case,
    )

    warnings = assert_agent_smoke_contract(
        answer=answer,
        public_answer=public_answer,
        final_answer=final_answer,
        case=case,
    )

    save_public_answer(
        output_dir=output_dir,
        case_name=case.name,
        public_answer=public_answer,
    )

    return {
        "name": case.name,
        "question": case.question,
        "query_type": answer.query_type,
        "status": answer.status,
        "confidence": answer.confidence,
        "is_fallback": answer.is_fallback,
        "final_answer": final_answer,
        "warnings": warnings,
    }


def save_public_answer(
    output_dir: Path,
    case_name: str,
    public_answer: dict[str, Any],
) -> None:
    """
    保存 public answer 到 JSON 文件。

    参数：
        output_dir:
            输出目录。

        case_name:
            smoke case 名称。

        public_answer:
            DogKnowledgeAnswer 对外字典。

    返回值：
        None。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{case_name}.json"

    output_path.write_text(
        json.dumps(
            public_answer,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_smoke_summary(
    mode: SmokeMode,
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    打印 smoke 汇总结果。

    参数：
        mode:
            smoke 模式。

        results:
            smoke case 执行结果列表。

        output_dir:
            输出目录。

    返回值：
        None。
    """

    print()
    print("=" * 80)
    print(f"DogKnowledgeAgent v1.7.3 Smoke Summary | mode={mode}")
    print("=" * 80)

    for item in results:
        print(f"[PASS] {item['name']}")
        print(f"  question     : {item['question']}")
        print(f"  query_type   : {item['query_type']}")
        print(f"  status       : {item['status']}")
        print(f"  confidence   : {item['confidence']}")
        print(f"  is_fallback  : {item['is_fallback']}")
        print(f"  final_answer : {item['final_answer']}")
        warnings = item.get("warnings") or []
        if warnings:
            print("  warnings     :")
            for warning in warnings:
                print(f"    - {warning}")
        else:
            print("  warnings     : none")
        print("-" * 80)

    print(f"Public answer JSON saved to: {output_dir.as_posix()}")
    print("=" * 80)


async def run_contract_smoke(
    output_dir: Path,
) -> None:
    """
    运行 contract 模式 smoke。

    参数：
        output_dir:
            输出目录。

    返回值：
        None。
    """

    results = []

    for case in build_smoke_cases():
        result = run_contract_smoke_case(
            case=case,
            output_dir=output_dir,
        )
        results.append(result)

    print_smoke_summary(
        mode="contract",
        results=results,
        output_dir=output_dir,
    )


async def run_agent_smoke(
    output_dir: Path,
) -> None:
    """
    运行真实 agent 模式 smoke。

    参数：
        output_dir:
            输出目录。

    返回值：
        None。
    """

    try:
        agent = await build_real_dog_knowledge_agent()

        results = []

        for case in build_smoke_cases():
            result = await run_agent_smoke_case(
                agent=agent,
                case=case,
                output_dir=output_dir,
            )
            results.append(result)

        print_smoke_summary(
            mode="agent",
            results=results,
            output_dir=output_dir,
        )

    finally:
        await shutdown_runtime_container()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace:
            命令行参数对象。
    """

    parser = argparse.ArgumentParser(
        description="DogKnowledgeAgent v1.7.3 输出协议 Smoke 脚本。"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "contract",
            "agent",
        ],
        default="contract",
        help="contract 表示只验证 Formatter/Adapter；agent 表示调用真实 DogKnowledgeAgent。",
    )

    parser.add_argument(
        "--output-dir",
        default="tmp/smoke_v173_dog_knowledge_answer_contract",
        help="public answer JSON 输出目录。",
    )

    return parser.parse_args()


async def main() -> None:
    """
    Smoke 脚本主入口。

    参数：
        无。

    返回值：
        None。
    """

    args = parse_args()

    output_dir = Path(args.output_dir)

    if args.mode == "contract":
        await run_contract_smoke(output_dir=output_dir)
        return

    await run_agent_smoke(output_dir=output_dir)


if __name__ == "__main__":
    asyncio.run(main())
