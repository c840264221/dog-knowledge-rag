"""真实多智能体案例：支持固定计划与模型生成计划。

手动运行：python -m tests.learning.plus_real_weather_collaboration
模型规划：python -m tests.learning.plus_real_weather_collaboration --model-plan
会使用项目真实模型、天气接口及知识库，可能产生费用和模型下载。
模型模式在执行前人工审阅计划；不验证主图跨轮入口或跨进程恢复。
不要与服务进程共用此入口。
"""

import argparse
import asyncio
import json
from typing import Any
from uuid import uuid4

from src.agents.collaboration.aggregator import ResultAggregator
from src.agents.collaboration.contracts import AgentTaskPlan
from src.agents.collaboration.planner import PlannerAgent
from src.agents.collaboration.scheduler import MultiAgentTaskScheduler
from src.runtime.context import RuntimeContext, runtime_ctx


DEFAULT_OBJECTIVE = (
    "查询成都当前天气，结合知识库中的金毛养护资料，给出今天遛狗的注意事项。"
    "请区分实时天气、知识库依据和资料不足，不要推断天气接口未提供的信息，"
    "不要编造狗狗的年龄、健康状态或具体安全阈值。"
)


async def build_model_advice_plan(llm_provider: Any, objective: str) -> AgentTaskPlan:
    """使用现有规划器生成计划，不执行计划中的步骤。

    参数：llm_provider为容器提供的模型服务；objective为用户的完整需求。
    返回：通过现有结构和依赖校验的任务计划，仍需人工审阅业务合理性。
    """
    planner = PlannerAgent(
        llm_provider=llm_provider,
        available_agents={
            "tool_agent": (
                "执行工具任务。本演示只允许使用 weather 查询城市当前天气；"
                "不能编造天气结果，不得安排其他工具操作。"
            ),
            "dog_knowledge_agent": (
                "检索犬种知识库，提供有来源的养护资料或建议。"
                "如果该步骤需要前置工具结果，必须声明 depends_on；"
                "如果只是查询通用资料，可以独立执行。"
            ),
        },
        maximum_steps=4,
        maximum_plan_attempts=2,
    )
    return await planner.create_plan(
        objective=objective,
        context={
            "step_input_guidance": (
                "每个步骤的 input_data 应包含明确的 question；"
                "知识检索步骤可提供 retrieval_question。"
                "不预填执行结果、确认状态或内部控制字段。"
            ),
            "aggregation_guidance": (
                "最终聚合器会结合各步骤结果生成回答，不需额外创建最终汇总步骤。"
                "不要为凑顺序增加依赖；需要根据天气定向检索时才依赖天气步骤。"
            ),
        },
    )


def build_weather_advice_plan() -> AgentTaskPlan:
    """构建固定业务计划。

    参数：无，使用成都与金毛作为本次固定案例。
    返回：两个具有依赖关系的步骤；不预填工具调用或用户确认状态。
    """
    return AgentTaskPlan(
        plan_id=f"weather-advice-{uuid4().hex}",
        objective=(
            "查询成都当前天气，结合金毛养护资料给出今天遛狗的注意事项。"
            "区分实时天气、知识库依据与资料不足；不得推断接口未提供的降雨情况。"
        ),
        steps=[
            {
                "step_id": "weather",
                "title": "查询成都当前天气",
                "assigned_agent": "tool_agent",
                "input_data": {
                    "question": "使用weather工具查询成都当前天气，只查询，不执行其他操作。",
                },
                "expected_output": "天气工具实际返回的地点、温度、风速；查询失败时明确失败。",
            },
            {
                "step_id": "walking_advice",
                "title": "结合天气检索金毛遛狗注意事项",
                "assigned_agent": "dog_knowledge_agent",
                "depends_on": ["weather"],
                "input_data": {
                    "question": (
                        "请检索知识库中的金毛运动、遛狗和户外养护资料，"
                        "结合前置步骤真实天气结果给出建议并说明资料来源。"
                        "缺少依据就说明限制，不要编造来源、健康状态或具体安全阈值；"
                        "天气仅提供温度和风速时，不推断降雨情况。"
                    ),
                    "retrieval_question": "金毛运动需求 遛狗 户外养护 注意事项",
                },
                "expected_output": "有资料依据的建议，以及未被资料覆盖的问题说明。",
            },
        ],
    )


def show_stage(stage: str, result) -> None:
    """打印阶段证据。

    参数：stage为阶段名称，result为可序列化的项目数据模型。
    返回：无；向终端输出完整JSON，便于区分步骤状态、工具结果和最终答案。
    """
    print(f"\n===== {stage} =====", flush=True)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), flush=True)


async def run_real_weather_advice(
    *, model_plan: bool = False, objective: str = DEFAULT_OBJECTIVE,
) -> None:
    """运行真实服务案例，并在终端处理有限轮用户回复。

    参数：model_plan是否使用模型规划；objective为模型模式的用户需求。
    模型、知识库及服务连接读取项目现有配置；固定模式保持原有需求。
    返回：无；打印计划、每轮调度结果和最终聚合结果，异常保留调用栈。
    说明：等待结果仅保留在本次进程中，不声明已经验证检查点恢复。
    """
    # 延迟导入便于在真实服务初始化前给出清晰提示；不另建一套Provider。
    print("即将启动项目真实服务，可能加载重排模型并产生模型调用费用。", flush=True)
    from src.runtime.container.init import container

    run_id = uuid4().hex
    context = RuntimeContext(
        trace_id=run_id,
        user_id="plus_real_case_user",
        session_id=f"weather-advice-{run_id}",
        component="plus_real_weather_collaboration",
    )
    previous_context = runtime_ctx.get()
    runtime_ctx.set(context)
    try:
        await context.startup()
        await container.startup()
        runtime = container.get("graph_runtime")
        workers = runtime.collaboration_workers
        scheduler = MultiAgentTaskScheduler(
            workers={name: workers[name] for name in ("tool_agent", "dog_knowledge_agent")},
            maximum_parallel_steps=1,
            maximum_step_attempts=1,
        )
        if model_plan:
            plan = await build_model_advice_plan(container.get("llm"), objective)
            show_stage("模型生成的计划：结构通过不代表业务正确", plan)
            if plan.requires_user_input or plan.status != "planned":
                print(f"计划尚不能执行：{plan.clarification_prompt or plan.status}")
                print("本入口不处理规划阶段恢复；请补全需求后用 --question 重新运行。")
                return
            print("\n请核对每一步的执行者、业务输入、依赖和预期输出。")
            print("只允许本案例的天气查询和知识检索；不合理则退出，不执行计划。")
            print("本次审阅不代替工具自身的权限校验和执行确认。")
            approval = await asyncio.to_thread(input, "输入 执行 继续，其他输入均退出：")
            if approval.strip() != "执行":
                print("未批准执行计划，演示结束。")
                return
        else:
            plan = build_weather_advice_plan()
            show_stage("固定计划：不是模型生成的计划", plan)
        result = await scheduler.execute(plan)
        show_stage("首次调度结果", result)

        for round_number in range(1, 4):
            if result.status != "awaiting_input":
                break
            replies = {}
            for step_result in result.task_results:
                if step_result.status != "awaiting_input":
                    continue
                print(f"\n步骤 {step_result.step_id}：{step_result.clarification_prompt}", flush=True)
                reply = await asyncio.to_thread(input, "请输入回复（确认/拒绝；输入 exit 结束演示）：")
                if reply.strip().casefold() == "exit":
                    print("已结束演示，不再继续调度；本脚本不支持退出后恢复。")
                    return
                replies[step_result.step_id] = reply
            result = await scheduler.resume(result, user_inputs=replies)
            show_stage(f"第{round_number}轮恢复结果", result)

        if result.status not in {"running", "partial"}:
            print(f"当前状态为 {result.status}，不生成成功答案，请先检查上述步骤结果。")
            return
        aggregator = ResultAggregator(llm_provider=container.get("llm"))
        result = await aggregator.aggregate(result)
        show_stage("聚合结果：仍需人工检查事实和引用是否正确", result)
    finally:
        try:
            await container.shutdown()
        finally:
            try:
                await context.shutdown()
            finally:
                runtime_ctx.set(previous_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实多智能体固定计划/模型规划演示")
    parser.add_argument("--model-plan", action="store_true", help="由真实模型生成计划，人工审阅后执行")
    parser.add_argument("--question", default=None, help="模型规划模式下的完整用户需求")
    arguments = parser.parse_args()
    if arguments.question is not None and not arguments.model_plan:
        parser.error("--question 必须与 --model-plan 一起使用，固定模式不接受自定义需求")
    asyncio.run(run_real_weather_advice(
        model_plan=arguments.model_plan,
        objective=arguments.question if arguments.question is not None else DEFAULT_OBJECTIVE,
    ))
