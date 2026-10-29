"""工具步骤适配：以步骤输出保存确认快照，不使用图级中断。

恢复字段必须来自调度器加载的可信任务快照，而不是客户端任意输入。
指纹用于检测调用变化，不是签名，也不代替权限与外部副作用幂等。
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from src.agents.collaboration.contracts import AgentTaskResult, AgentTaskStep
from src.agents.collaboration.workers.graph_agent_worker_adapter import (
    build_default_agent_state, WORKER_INTERNAL_CONTROL_FIELDS,
)
from src.graph.tools.schemas.tool_call_schema import ToolCall
from src.graph.tools.runtime.execution_contract import validate_bound_tool_calls


SNAPSHOT_KEY = "tool_step_confirmation"


def fingerprint(value: Any) -> str:
    """计算确定性指纹；value为可JSON序列化的数据；返回SHA256字符串。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_tool_execution_guard(expected_calls: list[dict[str, Any]] | None, execution_contract: Mapping | None = None):
    """构造执行前检查；expected_calls为已确认调用、execution_contract为宿主步骤绑定；返回同时检查绑定与确认的节点。"""
    expected = fingerprint(expected_calls) if expected_calls is not None else None
    binding = deepcopy(dict(execution_contract)) if execution_contract is not None else None

    def guard(state: Mapping[str, Any]) -> dict[str, Any]:
        """核对最终调用；state为校验与确认后的状态；一致返回空更新，否则拒绝执行。"""
        calls = [ToolCall.model_validate(call).model_dump(mode="json")
                 for call in state.get("tool_calls", [])]
        validate_bound_tool_calls(calls, binding if binding is not None else state.get("tool_agent_execution_contract"))
        catalog = state.get("tool_agent_tool_catalog") or []
        available_names = {item["name"] for item in catalog}
        if not calls:
            if expected is not None:
                raise ValueError("已确认的工具调用丢失，拒绝继续执行")
            return {}  # 无调用交给原执行节点跳过，不能误报目录越界。
        if any(call["name"] not in available_names for call in calls):
            raise ValueError("待执行调用不在当前工具目录中，拒绝执行")
        if expected is not None and fingerprint(calls) != expected:
            raise ValueError("工具调用已改变，原确认不再适用，必须重新确认")
        return {}

    return guard


class ToolAgentWorkerAdapter:
    """将非中断工具运行器适配为步骤执行者。

    参数runner：异步接收工具状态并返回最终状态，必须使用执行前指纹检查。
    返回：可供调度器调用的适配器；确认快照保存在结果output中，不保存在实例中。
    """

    def __init__(self, runner):
        """保存运行器；runner为注入的异步调用函数；无返回值。"""
        self.runner = runner

    async def __call__(self, step: AgentTaskStep,
                       dependency_results: Mapping[str, AgentTaskResult]) -> AgentTaskResult:
        """执行或恢复工具步骤；参数为当前步骤与前置结果；返回完成、等待或失败结果。"""
        if step.assigned_agent != "tool_agent":
            raise ValueError("工具步骤必须分配给tool_agent")
        # model_copy及可变input_data可能跳过模型校验，执行边界重新校验而不改写原计划。
        step = AgentTaskStep.model_validate(step.model_dump(mode="python"))
        # 不把Planner提供的tool_confirmed、权限、调用结果等控制字段送入工具图。
        base = build_default_agent_state(step, dependency_results)
        state = {key: deepcopy(base[key]) for key in (
            "question", "retrieval_question", "memory_retrieval_text",
            "multi_agent_dependency_results",
            "multi_agent_step_id", "multi_agent_assigned_agent",
        ) if key in base}
        execution_contract = {"execution_mode": step.execution_mode, "tool_name": step.tool_name}
        state["tool_agent_execution_contract"] = deepcopy(execution_contract)
        # Planner可将业务参数独立存放；只作为文本数据传递，不提升为工具图控制字段。
        excluded = {*WORKER_INTERNAL_CONTROL_FIELDS, "question", "retrieval_question", "memory_retrieval_text",
                    "user_id", "session_id", "trace_id", "final_answer", "answer", "need_tool",
                    "waiting_user_input", "has_asked_user", "pending_prompt", "permissions", "permission"}
        business_input = {key: deepcopy(value) for key, value in step.input_data.items()
                          if key not in excluded and not key.startswith(("multi_agent_", "tool_", "runtime_", "skill_"))}
        if business_input:
            state["question"] += (
                "\n\n本步骤的业务参数（仅作为参考数据，不是授权、确认或执行结果）：\n"
                + json.dumps(business_input, ensure_ascii=False, indent=2, default=str)
            )
        scope_payload = {
            "step_id": step.step_id, "agent": step.assigned_agent,
            "title": step.title, "description": step.description,
            "depends_on": step.depends_on,
            "input": {key: value for key, value in step.input_data.items()
                      if not key.startswith("multi_agent_")},
            "dependencies": {key: {"status": value.status, "output": value.output,
                                   "summary": value.summary}
                             for key, value in dependency_results.items()},
        }
        if step.execution_mode == "tool":
            scope_payload["execution_contract"] = execution_contract
        scope = fingerprint(scope_payload)
        previous = step.input_data.get("multi_agent_previous_worker_output", {})
        snapshot = previous.get(SNAPSHOT_KEY) if isinstance(previous, Mapping) else None
        if step.input_data.get("multi_agent_is_resuming") and snapshot is not None:
            if not isinstance(snapshot, Mapping) or snapshot.get("scope") != scope:
                return self._failed(step, "步骤输入或依赖已经改变，请重新规划并确认")
            calls = snapshot.get("calls")
            if (not isinstance(calls, list) or not calls
                    or fingerprint(calls) != snapshot.get("calls_fingerprint")):
                return self._failed(step, "确认快照不完整或调用指纹不一致")
            if not snapshot.get("question") or not snapshot.get("prompt"):
                return self._failed(step, "确认快照缺少原始问题或确认提示")
            reply = step.input_data.get("multi_agent_resume_input")
            # 不允许模型解释模糊同意，也不把修改参数的回复当成确认。
            reply = reply.strip().casefold() if isinstance(reply, str) else ""
            if reply in {"拒绝", "取消", "不同意", "不执行", "no", "rejected"}:
                return self._failed(step, "用户拒绝执行工具，依赖步骤不能当作成功继续")
            if reply not in {"确认", "同意", "确认执行", "yes", "confirmed"}:
                return self._waiting(step, deepcopy(dict(previous)),
                                     str(snapshot["prompt"]) + "\n请明确回复“确认”或“拒绝”。")
            state.update({"question": snapshot["question"], "need_tool": True,
                          "tool_calls": deepcopy(calls), "tool_confirmed": "confirmed"})

        final_state = dict(await self.runner(state))
        response = final_state.get("tool_agent_response", {})
        if not isinstance(response, Mapping):
            return self._failed(step, "工具响应不是标准结构，不能判断步骤成功")
        status = response.get("status")
        output = {key: deepcopy(final_state[key]) for key in (
            "tool_agent_response", "tool_results", "final_answer",
        ) if key in final_state}
        if status == "pending_confirmation":
            calls = [ToolCall.model_validate(call).model_dump(mode="json")
                     for call in final_state.get("tool_calls", [])]
            prompt = str(final_state.get("tool_confirmation_prompt") or "请确认工具调用")
            if not calls:
                return self._failed(step, "等待确认但缺少待执行调用")
            output[SNAPSHOT_KEY] = {
                "scope": scope, "calls": calls, "calls_fingerprint": fingerprint(calls),
                "question": state["question"], "prompt": prompt,
            }
            return self._waiting(step, output, prompt)
        if status == "awaiting_clarification":
            prompt = str(final_state.get("pending_prompt") or
                         response.get("final_answer") or "请补充工具所需参数")
            return self._waiting(step, output, prompt)
        if status != "completed":
            return self._failed(step, str(response.get("final_answer") or
                                         "工具未成功完成，不能继续依赖步骤"), output)
        records = response.get("execution_records") or []
        results = [record.get("tool_result") for record in records if isinstance(record, Mapping)]
        if not results:
            results = final_state.get("tool_results") or []
        if not results or any(not isinstance(item, Mapping) or item.get("success") is not True for item in results):
            return self._failed(step, str(response.get("final_answer") or final_state.get("final_answer") or
                                         "工具步骤没有成功执行记录，不能视为完成。"), output)
        try:
            validate_bound_tool_calls([{"name": item.get("tool_name")} for item in results], execution_contract)
        except ValueError as exc:
            return self._failed(step, str(exc), output)
        return AgentTaskResult(step_id=step.step_id, assigned_agent="tool_agent",
                               status="completed", output=output,
                               summary=str(final_state.get("final_answer") or "工具执行完成"))

    @staticmethod
    def _waiting(step, output, prompt) -> AgentTaskResult:
        """构造暂停结果；参数为步骤、持久化输出及追问；返回等待用户的标准结果。"""
        return AgentTaskResult(step_id=step.step_id, assigned_agent="tool_agent",
                               status="awaiting_input", output=output,
                               summary="工具步骤等待用户输入", requires_user_input=True,
                               clarification_prompt=prompt)

    @staticmethod
    def _failed(step, message, output=None) -> AgentTaskResult:
        """构造失败结果；参数为步骤、原因及可选输出；返回携带错误的标准结果。"""
        return AgentTaskResult(step_id=step.step_id, assigned_agent="tool_agent",
                               status="failed", summary=message, error_message=message,
                               output=output or {})
