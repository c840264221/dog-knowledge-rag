"""将已确认的记账动作接入项目ToolAgent子图，不在通用模型解析阶段生成身份或幂等键。"""

from copy import deepcopy

from src.mcp.expenses.schemas import ExpenseBatchInput, ExpenseBatchWriteResult, ExpenseInput, ExpenseWriteResult


class ExpenseToolService:
    """项目级记账执行入口；复用目录、参数校验、确认、运行时分流和结果适配。"""

    def __init__(self, mcp_provider):
        """注入已启动的提供者；mcp_provider为真实记账MCP能力；无返回值，不直接创建连接。"""
        self.mcp_provider = mcp_provider

    async def execute_confirmed(self, tool_name: str, arguments: dict, *, confirmed: bool):
        """执行可信调用方已确认的单笔或批量动作。

        参数：tool_name由程序选定；arguments含程序分配且持久化的原动作键；
            confirmed必须来自调用方本次确认，不接收模型输出作为授权。
        返回值：业务回执与项目工具执行轨迹；失败抛异常，由调用方保留失败或未知状态。
            当前是受控应用入口，不自动路由任意问题，也不是生产身份认证系统。
        """
        if confirmed is not True:
            raise PermissionError("没有取得用户确认，禁止执行记账。")
        inputs = {"record_expense": ExpenseInput, "record_expenses": ExpenseBatchInput}
        outputs = {"record_expense": ExpenseWriteResult, "record_expenses": ExpenseBatchWriteResult}
        if tool_name not in inputs:
            raise ValueError("此入口只执行程序准备好的单笔或批量记账动作。")
        checked = inputs[tool_name].model_validate(arguments).model_dump(mode="json")
        calls = [{"name": tool_name, "args": checked}]

        from src.agents.tool_agent.graph import build_tool_agent_graph
        from src.agents.collaboration.workers.tool_agent_worker_adapter import build_tool_execution_guard

        def reject_replanning(_):
            """拒绝重新生成已确认动作；参数为解析输入；始终抛错，避免改变已确认参数与原键。"""
            raise RuntimeError("已确认动作不得重新交给模型规划。")

        graph = build_tool_agent_graph(
            parser=reject_replanning, llm_provider=None,
            mcp_tool_definitions=self.mcp_provider.tool_definitions,
            mcp_client=self.mcp_provider, checkpoint_manager=None, interrupt_func=None,
            before_execute=build_tool_execution_guard(deepcopy(calls)),
        )
        state = await graph.ainvoke({
            "question": "执行调用方已确认的记账清单，不重新规划，不改写参数。",
            "need_tool": True,
            "tool_calls": calls,
            "tool_results": [],
            "tool_round": 0,
            "tool_confirmed": "confirmed",
        })
        results = state.get("tool_results", [])
        if len(results) != 1 or results[0].get("success") is not True:
            raise RuntimeError(f"项目工具链执行失败：{results}")
        receipt = outputs[tool_name].model_validate(results[0]["content"])
        trace = {
            "entry": "ExpenseToolService -> existing ToolAgent graph",
            "tool_catalog": state.get("tool_agent_tool_catalog", []),
            "execution_records": state.get("tool_agent_runtime_execution_records", []),
            "tool_result": results[0],
        }
        return receipt, trace
