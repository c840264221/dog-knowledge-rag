from collections.abc import Mapping
from typing import Any

from src.agents.dog_knowledge_agent.formatters.answer_formatter import (
    DogKnowledgeAnswerFormatter,
)
from src.agents.dog_knowledge_agent.contracts.schemas import (
    DogKnowledgeAnswer,
)


class DogKnowledgeAgentResponseAdapter:
    """
    DogKnowledgeAgent 最终响应适配器。

    功能：
        将 DogKnowledgeAgent 内部 pipeline 的最终结果统一转换成 DogKnowledgeAnswer，
        并且可以进一步转换成对外返回的 public dict。

    适用场景：
        1. DogKnowledgeAgent 普通函数调用后的最终返回。
        2. LangGraph 最后一个 finalize 节点。
        3. Root Agent 调用 DogKnowledgeAgent 后的结果标准化。
        4. WebUI / API 对外返回前的统一格式转换。

    专业名词：
        Adapter：
            适配器。用于连接两个接口或数据结构不一致的模块。

        Response Adapter：
            响应适配器。负责把内部响应转换成外部稳定响应。

        Public Dict：
            对外字典。可以返回给 API、WebUI 或 Root Agent 的安全字段结构。
    """

    QUESTION_KEYS: tuple[str, ...] = (
        "question",
        "user_question",
        "query",
        "input",
        "user_input",
    )

    PIPELINE_RESULT_KEYS: tuple[str, ...] = (
        "dog_knowledge_pipeline_result",
        "pipeline_result",
        "result",
        "dog_knowledge_result",
        "answer_result",
        "retrieval_result",
        "recommendation_result",
    )

    def __init__(
        self,
        formatter: DogKnowledgeAnswerFormatter | None = None,
    ) -> None:
        """
        初始化 DogKnowledgeAgentResponseAdapter。

        参数：
            formatter:
                DogKnowledgeAnswerFormatter 实例。
                如果不传，则内部自动创建一个默认 formatter。

        返回值：
            None。
        """

        self.formatter = formatter or DogKnowledgeAnswerFormatter()

    def finalize(
        self,
        pipeline_result: Mapping[str, Any] | Any,
        question: str | None = None,
        include_debug: bool = False,
        as_public_dict: bool = False,
    ) -> DogKnowledgeAnswer | dict[str, Any]:
        """
        将 DogKnowledgeAgent pipeline_result 转换成最终响应。

        参数：
            pipeline_result:
                DogKnowledgeAgent 内部 pipeline 的最终结果。
                可以是 dict、Pydantic Model、普通对象等。

            question:
                用户原始问题。
                如果传入，会优先使用该 question。

            include_debug:
                是否在 public dict 中包含 debug 调试信息。
                仅当 as_public_dict=True 时影响输出。

            as_public_dict:
                是否返回对外 dict。
                True 表示返回 dict；
                False 表示返回 DogKnowledgeAnswer 对象。

        返回值：
            DogKnowledgeAnswer | dict[str, Any]:
                当 as_public_dict=False 时，返回 DogKnowledgeAnswer；
                当 as_public_dict=True 时，返回可对外使用的 dict。
        """

        if isinstance(pipeline_result, DogKnowledgeAnswer):
            answer = pipeline_result
        else:
            answer = self.formatter.format(
                pipeline_result=pipeline_result,
                question=question,
            )

        if as_public_dict:
            return answer.to_public_dict(include_debug=include_debug)

        return answer

    def finalize_state(
        self,
        state: Mapping[str, Any] | Any,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """
        将 LangGraph state 或普通状态对象转换成最终输出更新。

        功能：
            从 state 中自动提取 question 和 pipeline_result，
            然后格式化为 DogKnowledgeAnswer。
            返回值是一个 dict，适合直接作为 LangGraph 节点的 state update。

        参数：
            state:
                LangGraph state、dict、Pydantic Model 或普通对象。
                其中通常包含 question、pipeline_result、recommendation_result 等字段。

            include_debug:
                public dict 是否包含 debug 信息。
                默认 False，避免对外暴露内部调试信息。

        返回值：
            dict[str, Any]:
                LangGraph 节点可以返回的更新字段。
                包含：
                    dog_knowledge_answer:
                        DogKnowledgeAnswer 的普通 dict 表示。

                    dog_knowledge_answer_public:
                        对外安全 dict。

                    final_answer:
                        自然语言答案文本，兼容旧的上层调用。
        """

        state_data = self._to_dict(state)

        question = self._extract_question(state_data)
        pipeline_result = self._extract_pipeline_result(state_data)

        answer = self.finalize(
            pipeline_result=pipeline_result,
            question=question,
            include_debug=include_debug,
            as_public_dict=False,
        )

        if not isinstance(answer, DogKnowledgeAnswer):
            raise TypeError(
                "DogKnowledgeAgentResponseAdapter.finalize_state 期望得到 DogKnowledgeAnswer 对象。"
            )

        public_answer = answer.to_public_dict(include_debug=include_debug)
        state_answer = self._answer_to_state_dict(
            answer=answer,
        )

        return {
            "dog_knowledge_answer": state_answer,
            "dog_knowledge_answer_public": public_answer,
            "final_answer": answer.answer,
        }

    def _answer_to_state_dict(
        self,
        answer: DogKnowledgeAnswer,
    ) -> dict[str, Any]:
        """
        将 DogKnowledgeAnswer 转换成适合写入 LangGraph state 的 dict。

        功能：
            避免把 DogKnowledgeAnswer 这种自定义 Pydantic 对象直接写入 checkpoint。
            LangGraph checkpoint 更适合保存 dict、list、str、int、float、bool、None
            这类稳定可序列化数据。

        参数：
            answer:
                已经格式化完成的 DogKnowledgeAnswer 对象。

        返回值：
            dict[str, Any]:
                可以安全写入 LangGraph state / checkpoint 的普通字典。
        """

        return answer.model_dump(
            mode="json",
            exclude_none=True,
        )

    def _extract_question(
        self,
        state_data: dict[str, Any],
    ) -> str | None:
        """
        从 state 中提取用户问题。

        参数：
            state_data:
                状态字典。

        返回值：
            str | None:
                用户问题；如果没有找到则返回 None。
        """

        for key in self.QUESTION_KEYS:
            value = state_data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        messages = state_data.get("messages")

        if isinstance(messages, list) and messages:
            latest_message = messages[-1]

            message_text = self._extract_message_text(latest_message)

            if message_text:
                return message_text

        return None

    def _extract_pipeline_result(
        self,
        state_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        从 state 中提取 DogKnowledgeAgent 内部 pipeline 结果。

        参数：
            state_data:
                状态字典。

        返回值：
            dict[str, Any]:
                pipeline_result 字典。
                如果找不到专门的 pipeline_result 字段，则返回整个 state_data，
                让 formatter 尝试从 state 中直接解析字段。
        """

        for key in self.PIPELINE_RESULT_KEYS:
            value = state_data.get(key)

            if value is None:
                continue

            result_data = self._to_dict(value)

            if result_data:
                return result_data

        return state_data

    def _extract_message_text(
        self,
        message: Any,
    ) -> str | None:
        """
        从 message 对象中提取文本。

        功能：
            兼容 LangChain Message、dict message 和普通字符串。

        参数：
            message:
                消息对象，可能是 HumanMessage、AIMessage、dict 或 str。

        返回值：
            str | None:
                消息文本；无法提取时返回 None。
        """

        if isinstance(message, str):
            cleaned = message.strip()
            return cleaned if cleaned else None

        if isinstance(message, Mapping):
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        return None

    def _to_dict(
        self,
        value: Any,
    ) -> dict[str, Any]:
        """
        将任意对象转换成 dict。

        参数：
            value:
                原始对象。

        返回值：
            dict[str, Any]:
                转换后的字典。
        """

        if value is None:
            return {}

        if isinstance(value, Mapping):
            return dict(value)

        if hasattr(value, "model_dump"):
            dumped = value.model_dump()

            if isinstance(dumped, Mapping):
                return dict(dumped)

        if hasattr(value, "dict"):
            dumped = value.dict()

            if isinstance(dumped, Mapping):
                return dict(dumped)

        if hasattr(value, "__dict__"):
            return dict(value.__dict__)

        return {}


def finalize_dog_knowledge_response(
    pipeline_result: Mapping[str, Any] | Any,
    question: str | None = None,
    include_debug: bool = False,
    as_public_dict: bool = False,
) -> DogKnowledgeAnswer | dict[str, Any]:
    """
    DogKnowledgeAgent 最终响应适配便捷函数。

    功能：
        不需要手动实例化 DogKnowledgeAgentResponseAdapter，
        直接把 pipeline_result 转换成 DogKnowledgeAnswer 或 public dict。

    参数：
        pipeline_result:
            DogKnowledgeAgent 内部 pipeline 的最终结果。

        question:
            用户原始问题。

        include_debug:
            是否包含 debug 调试信息。

        as_public_dict:
            是否返回对外 dict。

    返回值：
        DogKnowledgeAnswer | dict[str, Any]:
            标准化后的最终响应。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    return adapter.finalize(
        pipeline_result=pipeline_result,
        question=question,
        include_debug=include_debug,
        as_public_dict=as_public_dict,
    )


def finalize_dog_knowledge_state(
    state: Mapping[str, Any] | Any,
    include_debug: bool = False,
) -> dict[str, Any]:
    """
    DogKnowledgeAgent LangGraph state 最终响应适配便捷函数。

    功能：
        将 state 转换成 LangGraph 节点可返回的更新 dict。

    参数：
        state:
            LangGraph state 或普通状态对象。

        include_debug:
            是否在 public dict 中包含 debug 信息。

    返回值：
        dict[str, Any]:
            LangGraph 节点 update 字典。
    """

    adapter = DogKnowledgeAgentResponseAdapter()

    return adapter.finalize_state(
        state=state,
        include_debug=include_debug,
    )

