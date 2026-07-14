from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.tool_agent.graph import build_tool_agent_graph
from src.evaluation.schemas import AgentEvaluationCase
from src.graph.tools.schemas.tool_metadata import ToolMetadata
from src.graph.tools.schemas.tool_result_schema import ToolResult


class EvaluationToolParser:
    """
    为 ToolAgent 行为评估提供确定性的工具解析结果。

    功能：
        记录 ToolAgent 传入的解析输入，并返回黄金用例预先声明的解析结果，
        避免真实 LLM 波动影响工具编排评估。

    参数含义：
        result:
            黄金用例指定的 need_tool 和 tool_calls 解析结果。

    返回值含义：
        EvaluationToolParser:
            提供 ainvoke 方法的确定性异步解析器。
    """

    def __init__(self, result: dict[str, Any]) -> None:
        """
        初始化评估工具解析器。

        参数含义：
            result:
                每次解析调用需要返回的固定结果。

        返回值含义：
            None。
        """

        self.result = dict(result)
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        parser_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        记录解析输入并返回固定结果。

        参数含义：
            parser_input:
                ToolAgent 工具解析节点构造的输入字典。

        返回值含义：
            dict[str, Any]:
                黄金用例预设的工具解析结果副本。
        """

        self.inputs.append(dict(parser_input))
        return dict(self.result)


class EvaluationTool:
    """
    保存 ToolAgent 行为评估使用的工具元数据。

    参数含义：
        metadata:
            工具名称、确认策略和输入参数 Schema（数据模型）。

    返回值含义：
        EvaluationTool:
            可由评估工具注册表返回的工具对象。
    """

    def __init__(self, metadata: ToolMetadata) -> None:
        """
        初始化评估工具对象。

        参数含义：
            metadata:
                当前工具的标准 ToolMetadata。

        返回值含义：
            None。
        """

        self.metadata = metadata


class EvaluationToolRegistry:
    """
    为 ToolAgent 行为评估提供确定性的工具目录。

    参数含义：
        tools:
            工具名称到 EvaluationTool 的映射。

    返回值含义：
        EvaluationToolRegistry:
            支持 get_tool 查询的评估工具注册表。
    """

    def __init__(
        self,
        tools: dict[str, EvaluationTool],
    ) -> None:
        """
        初始化评估工具注册表。

        参数含义：
            tools:
                工具名称到评估工具对象的映射。

        返回值含义：
            None。
        """

        self.tools = dict(tools)

    def get_tool(self, name: str) -> EvaluationTool | None:
        """
        根据工具名称返回评估工具。

        参数含义：
            name:
                ToolAgent 需要查询的工具名称。

        返回值含义：
            EvaluationTool | None:
                找到时返回工具对象，不存在时返回 None。
        """

        return self.tools.get(name)


class EvaluationToolExecutor:
    """
    为 ToolAgent 行为评估提供确定性的工具执行结果。

    功能：
        记录实际执行的工具名称和参数，并返回固定 ToolResult，
        防止天气 API、系统日期和数据库状态影响离线评估。

    参数含义：
        无。

    返回值含义：
        EvaluationToolExecutor:
            支持异步 execute 方法并记录调用轨迹的执行器。
    """

    def __init__(self) -> None:
        """
        初始化评估工具执行器。

        参数含义：
            无。

        返回值含义：
            None。
        """

        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> ToolResult:
        """
        记录工具调用并返回固定成功结果。

        参数含义：
            tool_name:
                ToolAgent 实际请求执行的工具名称。
            args:
                ToolAgent 传给工具的参数字典。

        返回值含义：
            ToolResult:
                可被真实 ToolAgent Runtime Adapter（运行时适配器）消费的结果。
        """

        self.calls.append(
            {
                "tool_name": tool_name,
                "args": dict(args),
            }
        )
        content_by_tool = {
            "date": "2026-07-08",
            "weather": "成都天气晴，30°C",
        }
        return ToolResult(
            success=True,
            tool_name=tool_name,
            content=content_by_tool.get(tool_name, "执行成功"),
            metadata={
                "source": "tool_agent_evaluation_executor",
            },
        )


@dataclass
class ToolAgentScenarioRuntime:
    """
    保存一条 ToolAgent 评估场景的可执行依赖和运行轨迹。

    参数含义：
        graph:
            注入确定性依赖后编译完成的真实 ToolAgent 子图。
        initial_state:
            已剥离 evaluation 专用字段的子图输入 state。
        parser:
            本场景使用的确定性解析器。
        executor:
            本场景使用的确定性工具执行器。
        confirmation_prompts:
            工具确认节点实际生成的提示词列表。

    返回值含义：
        ToolAgentScenarioRuntime:
            可供行为评估器执行和读取轨迹的场景运行环境。
    """

    graph: Any
    initial_state: dict[str, Any]
    parser: EvaluationToolParser
    executor: EvaluationToolExecutor
    confirmation_prompts: list[str]


def build_evaluation_tool_registry() -> EvaluationToolRegistry:
    """
    构建包含 weather 和 date 的确定性评估工具注册表。

    参数含义：
        无。

    返回值含义：
        EvaluationToolRegistry:
            weather 需要确认且 city 必填，date 不需要确认的注册表。
    """

    return EvaluationToolRegistry(
        tools={
            "weather": EvaluationTool(
                metadata=ToolMetadata(
                    name="weather",
                    description="查询指定城市天气",
                    require_confirm=True,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "需要查询天气的城市。",
                            }
                        },
                        "required": ["city"],
                    },
                )
            ),
            "date": EvaluationTool(
                metadata=ToolMetadata(
                    name="date",
                    description="获取当前日期",
                    require_confirm=False,
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                )
            ),
        }
    )


def build_tool_agent_scenario_runtime(
    eval_case: AgentEvaluationCase,
) -> ToolAgentScenarioRuntime:
    """
    根据黄金用例构建可重复的真实 ToolAgent 子图运行环境。

    参数含义：
        eval_case:
            包含固定 Parser 输出和可选确认回答的统一评估用例。

    返回值含义：
        ToolAgentScenarioRuntime:
            已编译子图、输入 state 和可观察调用轨迹。
    """

    raw_parser_result = eval_case.input_state.get(
        "evaluation_parser_result",
        {
            "need_tool": False,
            "tool_calls": [],
        },
    )
    if not isinstance(raw_parser_result, dict):
        raise ValueError("evaluation_parser_result 必须是字典")

    parser = EvaluationToolParser(result=raw_parser_result)
    executor = EvaluationToolExecutor()
    confirmation_prompts: list[str] = []
    confirmation_response = eval_case.input_state.get(
        "evaluation_confirmation_response"
    )

    interrupt_func = None
    if confirmation_response is not None:
        def evaluation_interrupt(prompt: str) -> Any:
            """
            记录确认提示并返回黄金用例指定的用户回答。

            参数含义：
                prompt:
                    ToolAgent 确认节点生成的原始提示词。

            返回值含义：
                Any:
                    黄金用例中的 evaluation_confirmation_response。
            """

            confirmation_prompts.append(prompt)
            return confirmation_response

        interrupt_func = evaluation_interrupt

    # evaluation_ 前缀字段只驱动评估环境，不允许写入真实 ToolAgent state。
    initial_state = {
        key: value
        for key, value in eval_case.input_state.items()
        if not key.startswith("evaluation_")
    }
    initial_state["question"] = eval_case.question

    graph = build_tool_agent_graph(
        parser=parser,
        tool_registry=build_evaluation_tool_registry(),
        executor=executor,
        runtime_context_getter=lambda: None,
        interrupt_func=interrupt_func,
    )
    return ToolAgentScenarioRuntime(
        graph=graph,
        initial_state=initial_state,
        parser=parser,
        executor=executor,
        confirmation_prompts=confirmation_prompts,
    )
