"""
DogKnowledgeAgent 入口编排骨架。

功能：
    为 DogKnowledgeAgent 提供 V1.7.2 阶段的标准 Pipeline Skeleton。

    Pipeline Skeleton 中文叫“管线骨架”，表示：
    先固定一个复杂流程的执行顺序和状态字段，
    但暂时不执行每一层的真实业务逻辑。

    当前模块主要负责：
    1. 读取 module_contracts.py 中定义的职责层。
    2. 构建标准 pipeline step 列表。
    3. 构建 pipeline trace。
    4. 返回可以写入 DogState 的结构化字段。
    5. 为后续真正接入 DogKnowledgeAgent entry node 做准备。

当前不负责：
    1. 不构建真实 RagQuery。
    2. 不执行真实 Retriever。
    3. 不执行真实 Reranker。
    4. 不执行真实质量检测。
    5. 不构建真实 RagContext。
    6. 不检索用户长期记忆。
    7. 不调用 LLM 生成答案。

专业名词：
    Pipeline：管线，表示按顺序执行的一组处理步骤。
    Skeleton：骨架，表示先搭好的流程结构。
    Trace：追踪记录，表示每一步执行状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from src.agents.dog_knowledge_agent.module_contracts import (
    DogKnowledgeModuleContract,
    get_dog_knowledge_module_contracts,
    get_expected_dog_knowledge_layers,
)


PipelineStepStatus = Literal[
    "planned",
    "skipped",
    "completed",
    "failed",
]


@dataclass(frozen=True)
class DogKnowledgePipelineStep:
    """
    DogKnowledgeAgent Pipeline 步骤定义。

    功能：
        描述 DogKnowledgeAgent 内部标准执行管线中的一个步骤。

    参数：
        index:
            当前步骤在 pipeline 中的顺序，从 1 开始。

        layer:
            模块层级名称，例如 query_builder、retrieval、rerank。

        module_name:
            模块英文名称。

        chinese_name:
            模块中文名称。

        responsibility:
            当前步骤职责说明。

        expected_input:
            当前步骤预期输入。

        expected_output:
            当前步骤预期输出。

        status:
            当前步骤状态。
            skeleton 阶段默认为 planned。

    返回值：
        DogKnowledgePipelineStep:
            一个不可变的 pipeline 步骤对象。
    """

    index: int
    layer: str
    module_name: str
    chinese_name: str
    responsibility: str
    expected_input: str
    expected_output: str
    status: PipelineStepStatus = "planned"

    def to_dict(self) -> dict[str, Any]:
        """
        将 pipeline step 转换为 dict。

        功能：
            把 dataclass 对象转换成普通 dict，
            方便写入 LangGraph state、checkpoint 和 debug report。

        参数：
            无。

        返回值：
            dict[str, Any]:
                当前 pipeline step 的字典表示。
        """

        return {
            "index": self.index,
            "layer": self.layer,
            "module_name": self.module_name,
            "chinese_name": self.chinese_name,
            "responsibility": self.responsibility,
            "expected_input": self.expected_input,
            "expected_output": self.expected_output,
            "status": self.status,
        }


@dataclass(frozen=True)
class DogKnowledgePipelineTraceItem:
    """
    DogKnowledgeAgent Pipeline Trace 记录项。

    功能：
        记录 pipeline skeleton 中某一个步骤的执行状态。

    参数：
        index:
            当前步骤顺序。

        layer:
            模块层级名称。

        status:
            当前步骤状态。

        message:
            当前步骤说明。

        created_at:
            记录创建时间。

    返回值：
        DogKnowledgePipelineTraceItem:
            一个不可变的 trace 记录对象。
    """

    index: int
    layer: str
    status: PipelineStepStatus
    message: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """
        将 trace item 转换为 dict。

        功能：
            把 trace 记录转换为普通 dict，
            方便写入 state 和 debug report。

        参数：
            无。

        返回值：
            dict[str, Any]:
                当前 trace item 的字典表示。
        """

        return {
            "index": self.index,
            "layer": self.layer,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
        }


def build_dog_knowledge_pipeline_steps() -> tuple[
    DogKnowledgePipelineStep,
    ...
]:
    """
    构建 DogKnowledgeAgent 标准 pipeline steps。

    功能：
        根据 module_contracts.py 中定义的模块职责契约，
        生成 DogKnowledgeAgent 的标准执行步骤列表。

    参数：
        无。

    返回值：
        tuple[DogKnowledgePipelineStep, ...]:
            DogKnowledgeAgent 标准 pipeline steps。
    """

    contracts = get_dog_knowledge_module_contracts()

    return tuple(
        build_pipeline_step_from_contract(
            index=index,
            contract=contract,
        )
        for index, contract in enumerate(
            contracts,
            start=1,
        )
    )


def build_pipeline_step_from_contract(
        index: int,
        contract: DogKnowledgeModuleContract,
) -> DogKnowledgePipelineStep:
    """
    根据模块契约构建 pipeline step。

    功能：
        将 DogKnowledgeModuleContract 转换为 DogKnowledgePipelineStep。

    参数：
        index:
            当前步骤顺序，从 1 开始。

        contract:
            DogKnowledgeAgent 内部模块职责契约。

    返回值：
        DogKnowledgePipelineStep:
            pipeline 步骤对象。
    """

    return DogKnowledgePipelineStep(
        index=index,
        layer=contract.layer,
        module_name=contract.module_name,
        chinese_name=contract.chinese_name,
        responsibility=contract.responsibility,
        expected_input=contract.expected_input,
        expected_output=contract.expected_output,
        status="planned",
    )


def build_dog_knowledge_pipeline_trace(
        steps: tuple[DogKnowledgePipelineStep, ...],
) -> tuple[
    DogKnowledgePipelineTraceItem,
    ...
]:
    """
    构建 DogKnowledgeAgent pipeline trace。

    功能：
        根据 pipeline steps 生成 skeleton 阶段的 trace 记录。

        当前阶段只生成 planned 状态，
        表示这些步骤已经被纳入标准流程，
        但还没有执行真实业务逻辑。

    参数：
        steps:
            DogKnowledgeAgent pipeline steps。

    返回值：
        tuple[DogKnowledgePipelineTraceItem, ...]:
            pipeline trace 记录列表。
    """

    created_at = datetime.now(
        timezone.utc,
    ).isoformat()

    return tuple(
        DogKnowledgePipelineTraceItem(
            index=step.index,
            layer=step.layer,
            status=step.status,
            message=(
                f"{step.chinese_name} 已纳入 DogKnowledgeAgent 标准管线，"
                "当前为 skeleton 阶段，尚未执行真实业务逻辑。"
            ),
            created_at=created_at,
        )
        for step in steps
    )


def build_dog_knowledge_pipeline_skeleton_state_update(
        state: dict[str, Any],
) -> dict[str, Any]:
    """
    构建 DogKnowledgeAgent pipeline skeleton 的 state 更新字段。

    功能：
        根据当前 DogState 构建 pipeline skeleton 输出。
        该函数不修改传入的 state，只返回新的 state update dict。

    参数：
        state:
            当前 LangGraph 状态。
            通常包含 question、user_id、session_id、trace_id 等字段。

    返回值：
        dict[str, Any]:
            可以写入 DogState 的更新字段。
            包含 dog_knowledge_pipeline_status、
            dog_knowledge_pipeline_steps、
            dog_knowledge_pipeline_trace 等。
    """

    steps = build_dog_knowledge_pipeline_steps()

    trace = build_dog_knowledge_pipeline_trace(
        steps=steps,
    )

    return {
        "current_agent": "dog_knowledge_agent",
        "dog_knowledge_pipeline_status": "skeleton_ready",
        "dog_knowledge_pipeline_version": "v1.7.2-step3",
        "dog_knowledge_pipeline_question": state.get(
            "question",
            "",
        ),
        "dog_knowledge_pipeline_steps": [
            step.to_dict()
            for step in steps
        ],
        "dog_knowledge_pipeline_trace": [
            item.to_dict()
            for item in trace
        ],
    }


def get_dog_knowledge_pipeline_layer_order() -> tuple[str, ...]:
    """
    获取 DogKnowledgeAgent pipeline 层级顺序。

    功能：
        返回 DogKnowledgeAgent 标准 pipeline 的 layer 顺序。
        该顺序应该与 module_contracts.py 中定义的职责顺序保持一致。

    参数：
        无。

    返回值：
        tuple[str, ...]:
            pipeline 层级顺序。
    """

    return tuple(
        get_expected_dog_knowledge_layers()
    )


def render_dog_knowledge_pipeline_skeleton_markdown() -> str:
    """
    渲染 DogKnowledgeAgent pipeline skeleton Markdown。

    功能：
        将当前标准 pipeline skeleton 渲染成 Markdown 文本，
        方便写入文档、Debug Report 或学习笔记。

    参数：
        无。

    返回值：
        str:
            Markdown 格式的 pipeline skeleton 文档片段。
    """

    steps = build_dog_knowledge_pipeline_steps()

    lines = [
        "# DogKnowledgeAgent Pipeline Skeleton",
        "",
        "该文档描述 V1.7.2 Step 3 阶段的 DogKnowledgeAgent 标准入口编排骨架。",
        "",
        "## 标准流程",
        "",
        " -> ".join(
            step.layer
            for step in steps
        ),
        "",
        "## 步骤说明",
        "",
    ]

    for step in steps:
        lines.extend(
            [
                f"### {step.index}. {step.chinese_name} / {step.module_name}",
                "",
                f"- layer: `{step.layer}`",
                f"- status: `{step.status}`",
                f"- responsibility: {step.responsibility}",
                f"- expected_input: {step.expected_input}",
                f"- expected_output: {step.expected_output}",
                "",
            ]
        )

    return "\n".join(
        lines,
    )