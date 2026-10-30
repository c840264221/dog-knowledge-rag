"""项目工具层的真实记账MCP服务提供者；默认关闭，仅供固定本机教学身份使用。"""

import asyncio
import json
from urllib.parse import urlsplit
from copy import deepcopy

from src.mcp.expenses.schemas import (
    ExpenseBatchInput, ExpenseBatchWriteResult, ExpenseInput, ExpenseQueryInput,
    ExpenseQueryResult, ExpenseWriteResult,
)
from src.mcp.schemas import MockMcpToolDefinition
from src.runtime.context import runtime_ctx


class ExpenseMcpProvider:
    """统一管理本机记账服务目录与调用；不宣称支持生产认证或多用户账本。"""

    def __init__(self, *, enabled: bool = False, client_factory=None,
                 url: str = "http://127.0.0.1:8766/mcp"):
        """配置提供者；enabled显式启用，client_factory为可注入连接工厂；无返回值。

        url由可信宿主配置，只允许本机HTTP的/mcp地址；不接收模型参数，不自动重试写入。
        复用现有MockMcpToolDefinition作为目录数据契约，实际通信不是Mock（模拟调用）。
        """
        self.enabled = enabled
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.path != "/mcp" or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("教学记账服务只允许可信宿主配置的本机HTTP /mcp地址。")
        self.url = url
        self._client_factory = client_factory
        self._definitions = []
        self._ready = False

    def _connection(self):
        """创建短连接上下文；无参数；返回SDK异步上下文，生命周期由本提供者的调用管理。"""
        if self._client_factory is not None:
            return self._client_factory(self.url)
        from mcp import Client
        return Client(self.url)

    @property
    def tool_client(self):
        """提供统一客户端接口；无参数；返回已实现call_tool的自身，不创建额外连接。"""
        return self

    @property
    def tool_definitions(self) -> list[MockMcpToolDefinition]:
        """读取已发现目录；无参数；返回独立副本，禁用时为空，启用未启动时拒绝读取。"""
        if self.enabled and not self._ready:
            raise RuntimeError("记账MCP Provider尚未启动。")
        return [item.model_copy(deep=True) for item in self._definitions]

    async def startup(self) -> None:
        """发现并核对三个工具；无参数、无返回值；默认关闭时不联网，不影响原服务。"""
        if not self.enabled or self._ready:
            return
        async with asyncio.timeout(10):
            async with self._connection() as client:
                tools = (await client.list_tools()).tools
        names = {tool.name for tool in tools}
        required = {"record_expense", "record_expenses", "query_expenses"}
        if not required.issubset(names):
            raise ValueError("记账服务缺少所需工具，请重启最新版MCP服务。")
        definitions = []
        for tool in tools:
            if tool.name not in required:
                continue
            if tool.name == "record_expense" and "expense_date" not in tool.input_schema.get("required", []):
                raise ValueError("服务端单笔工具未要求账单日期，拒绝接入旧接口。")
            # 远端发现确认工具存在；本服务适配器以现有业务契约补齐宿主侧的格式约束。
            contracts = {"record_expense": ExpenseInput, "record_expenses": ExpenseBatchInput,
                         "query_expenses": ExpenseQueryInput}
            input_schema = deepcopy(contracts[tool.name].model_json_schema())
            if tool.name in {"record_expense", "record_expenses"}:
                # 这是工具运行参数策略，不是Skill授予的能力；模型不负责生成动作键。
                input_schema["properties"]["idempotency_key"]["x-runtime-generated"] = "uuid"
                input_schema["x-planning-skill"] = "expense-recording"
            definitions.append(MockMcpToolDefinition(
                name=tool.name, description=tool.description or "", input_schema=input_schema,
                # 确认策略由项目确定，不信任远端描述替写工具放宽权限。
                annotations={"require_confirm": tool.name != "query_expenses", "timeout": 10, "retries": 0},
            ))
        self._definitions = definitions
        self._ready = True

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """校验后执行一次MCP业务调用，返回普通业务字典，交既有适配器封装ToolResult。

        参数：name为已发现工具名；arguments为上游程序已准备并确认的参数。
        返回值：通过输出契约校验的业务内容；MCP错误、网络异常或身份不符均抛出，
            不把协议返回的错误当成成功。这里不授予确认、不生成动作键、不自动重试。
        """
        if not self.enabled or not self._ready:
            raise RuntimeError("记账MCP Provider未启用或未启动。")
        context = runtime_ctx.get()
        if context is None or context.user_id != "expense_learning_user":
            raise PermissionError("当前服务固定绑定教学账本，只允许expense_learning_user，不能访问其他用户账本。")
        inputs = {"record_expense": ExpenseInput, "record_expenses": ExpenseBatchInput, "query_expenses": ExpenseQueryInput}
        outputs = {"record_expense": ExpenseWriteResult, "record_expenses": ExpenseBatchWriteResult, "query_expenses": ExpenseQueryResult}
        if name not in {tool.name for tool in self._definitions}:
            raise ValueError("工具不在当前记账目录中。")
        checked = inputs[name].model_validate(arguments).model_dump(mode="json")
        async with asyncio.timeout(10):
            async with self._connection() as client:
                result = await client.call_tool(name, checked)
        if result.is_error:
            detail = "\n".join(block.text for block in result.content if block.type == "text")
            raise RuntimeError(f"记账MCP执行错误：{detail}")
        payload = result.structured_content
        if payload is None:
            texts = [block.text for block in result.content if block.type == "text"]
            if len(texts) != 1:
                raise ValueError("MCP未提供唯一可解析的业务结果。")
            payload = json.loads(texts[0])
        receipt = outputs[name].model_validate(payload)
        if name == "record_expense":
            if receipt.record.model_dump(mode="json", include=set(checked)) != checked:
                raise ValueError("记账回执与已确认参数不一致，结果需核查，不得换键重试。")
        elif name == "record_expenses":
            items = checked["items"]
            actual = [row.model_dump(mode="json", include=set(expected))
                      for row, expected in zip(receipt.records, items)]
            if (receipt.idempotency_key != checked["idempotency_key"]
                    or receipt.record_count != len(items) or len(receipt.records) != len(items)
                    or len({row.expense_id for row in receipt.records}) != len(items) or actual != items):
                raise ValueError("批量回执与已确认清单不一致，结果需核查，不得换键重试。")
        return receipt.model_dump(mode="json")

    async def shutdown(self) -> None:
        """停用目录与就绪状态；无参数、无返回值；各SDK连接在调用结束时已关闭。"""
        self._definitions = []
        self._ready = False
