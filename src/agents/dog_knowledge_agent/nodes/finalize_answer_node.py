from typing import Any

from src.agents.dog_knowledge_agent.adapters.response_adapter import (
    DogKnowledgeAgentResponseAdapter,
)


def build_finalize_dog_knowledge_answer_node(
    response_adapter: DogKnowledgeAgentResponseAdapter | None = None,
    include_debug: bool = False,
):
    """
    构建 DogKnowledgeAgent 最终答案节点。

    功能：
        返回一个可被 LangGraph 使用的 node function。
        该节点负责把 DogKnowledgeAgent 内部 state 转换成统一的 DogKnowledgeAnswer。

    参数：
        response_adapter:
            DogKnowledgeAgentResponseAdapter 实例。
            如果不传，则内部自动创建默认 adapter。

        include_debug:
            是否在 dog_knowledge_answer_public 中包含 debug 信息。
            默认 False，避免对外暴露内部调试信息。

    返回值：
        callable:
            一个 LangGraph 节点函数。
            输入 state，返回 state update dict。
    """

    adapter = response_adapter or DogKnowledgeAgentResponseAdapter()

    def finalize_dog_knowledge_answer_node(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        DogKnowledgeAgent 最终答案节点。

        功能：
            从 state 中读取内部 pipeline 结果，
            生成统一的 dog_knowledge_answer、dog_knowledge_answer_public 和 final_answer。

        参数：
            state:
                LangGraph 当前状态。
                通常包含 question、pipeline_result、retrieved_chunks、recommendations 等字段。

        返回值：
            dict[str, Any]:
                LangGraph state update。
        """

        return adapter.finalize_state(
            state=state,
            include_debug=include_debug,
        )

    return finalize_dog_knowledge_answer_node


finalize_dog_knowledge_answer_node = build_finalize_dog_knowledge_answer_node()
