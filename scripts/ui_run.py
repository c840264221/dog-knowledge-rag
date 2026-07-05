import gradio as gr
import src.runtime.events.setup  # noqa: F401
from src.graph.graph_run import run_main_graph_with_result
from src.logger import logger

from src.runtime.hooks.tool_counter_hook import ToolCounterHook


from src.runtime.trace.init import trace_manager

import uuid

from src.runtime.container.init import (
    container
)
from src.runtime.resume.contracts import (
    GraphFinalResult,
    GraphInterruptResult,
)

from src.runtime.timeline.timeline_reporter import TimelineReporter

from src.settings import settings



def ensure_trace_exists(
        trace_id: str,
) -> None:
    """
    确保 trace_manager 中存在当前 trace。

    功能：
        在新请求或 resume（恢复运行）前检查 trace_manager。
        如果当前 trace_id 对应的 Trace 不存在，则重新创建一个。
        这样可以避免 UI 恢复时 RuntimeContext 已恢复，但内存中的 Trace 丢失，
        导致工具 TraceMiddleware 创建 span 失败。

    参数：
        trace_id：
            当前请求链路追踪 ID。

    返回值：
        None：
            无业务返回值，只保证 trace_manager 内部状态存在。
    """

    if not trace_id:
        return

    trace_manager.ensure_trace(
        trace_id
    )


def build_resume_runtime_context(
        trace_id: str,
        session_id: str,
        state: dict,
        checkpoint_manager,
):
    """
    构建恢复阶段使用的 RuntimeContext。

    功能：
        优先从 checkpoint（检查点）恢复 RuntimeContext。
        如果检查点缺失，则创建一个最小可用 RuntimeContext 作为兜底。
        同时恢复 trace_id、session_id、user_id、component 和 tool.before hook。

    参数：
        trace_id：
            当前请求链路追踪 ID。
        session_id：
            Gradio 当前会话 ID，同时作为 LangGraph thread_id 使用。
        state：
            UI session_state，用于读取 user_id 等恢复信息。
        checkpoint_manager：
            CheckpointManager（检查点管理器），用于恢复 RuntimeContext 快照。

    返回值：
        RuntimeContext：
            可用于 resume 阶段的运行时上下文。
    """

    from src.runtime.context import RuntimeContext

    ctx = checkpoint_manager.restore_checkpoint(
        trace_id
    )

    if ctx is None:
        logger.warning(
            "未从 checkpoint 恢复到 RuntimeContext，"
            "将创建最小恢复上下文继续执行。"
        )

        ctx = RuntimeContext(
            trace_id=trace_id,
            session_id=session_id,
            user_id=state.get(
                "user_id",
                "unknown"
            ),
            component="resume_handler",
        )

    ctx.trace_id = trace_id
    ctx.session_id = session_id
    ctx.user_id = state.get(
        "user_id",
        ctx.user_id or "unknown"
    )
    ctx.component = "resume_handler"

    ctx.hooks().register(
        "tool.before",
        ToolCounterHook()
    )

    return ctx


def print_runtime_reports_if_enabled(
        runtime,
) -> None:
    """
    按配置打印 Runtime 可观测报告。

    功能：
        根据 settings.observability 控制 Timeline Report 和 Runtime Report
        是否打印到控制台。

        当前阶段：
            1. 控制台默认不打印完整 Timeline。
            2. 控制台默认不打印完整 Runtime Report。
            3. 后续 RAG Debug Report 会单独输出到文件。

    参数：
        runtime:
            当前 RuntimeContext。

    返回值：
        None。

    专业名词：
        Console：
            控制台。开发运行时直接看到的终端输出。

        Report：
            报告。结构化整理后的运行信息。
    """

    if settings.observability.ENABLE_CONSOLE_TIMELINE_REPORT:
        TimelineReporter.report()

    if settings.observability.ENABLE_CONSOLE_RUNTIME_REPORT:
        from src.runtime.observability.report_builder import (
            ReportBuilder
        )

        from src.runtime.observability.report_printer import (
            ReportPrinter
        )

        report = ReportBuilder.build(
            runtime
        )

        ReportPrinter.print(
            report
        )


async def respond_and_process(
    question: str,
    history: list,
    state: dict,
    request: gr.Request
):

    session_id = request.session_hash

    # ===== 创建 trace =====

    trace_id = str(uuid.uuid4())

    ensure_trace_exists(
        trace_id
    )

    # ===== user_id =====

    user_id = state.get("user_id") if state else None


    # if user_id:
    #
    #     trace_ctx.set_user_id(user_id)
    #
    # else:
    #
    #     trace_ctx.set_user_id("unknown")

    # ===== 写入 contextvars =====

    from src.runtime.context import (
        runtime_ctx,
        RuntimeContext
    )

    ctx = RuntimeContext(

        trace_id=trace_id,

        session_id=session_id,

        user_id=user_id or "unknown",

        component="gradio_handler"
    )

    ctx.hooks().register(
        "tool.before",
        ToolCounterHook()
    )

    await runtime_ctx.create(ctx)

    # 保存runtime state
    runtime_ctx.get().state().set_agent(
        "graph agent"
    )


    # trace_ctx.set_trace_id(trace_id)
    #
    # trace_ctx.set_session_id(session_id)
    #
    # trace_ctx.set_component("gradio_handler")

    # ===== 初始化 state =====

    if not state:

        state = {

            "config": {
                "configurable": {
                    "thread_id": session_id
                }
            },

            "pending": False,

            "pending_prompt": "",

            # ⭐ 新增
            "trace_id": trace_id
        }

    else:

        state["config"]["configurable"]["thread_id"] = session_id

        # ⭐ 保证恢复时 trace 不丢
        state["trace_id"] = trace_id

    # ===== 添加用户消息 =====

    history.append({

        "role": "user",

        "content": question
    })

    # ===== 调用图 =====

    result = await run_main_graph_with_result(

        question,

        thread_id=session_id,

        trace_id=trace_id
    )

    # ===== interrupt =====

    if isinstance(result, GraphInterruptResult):

        prompt = result.prompt

        state["pending"] = True

        state["pending_prompt"] = prompt

        history.append({

            "role": "assistant",

            "content": f"⚠️ 需要确认：{prompt}"
        })

        return (

            history,

            state,

            gr.update(visible=True),

            prompt
        )

    # ===== normal =====

    else:

        state["pending"] = False

        answer = (
            result.answer
            if isinstance(result, GraphFinalResult)
            else str(result)
        )

        history.append({

            "role": "assistant",

            "content": answer
        })

        from src.runtime.scopes.metrics_scope import MetricsScope
        metrics_scope = runtime_ctx.get().service(MetricsScope).get_metrics()
        logger.info(f'运行结束，metrics为：{metrics_scope}')

        #==============时间线打印==============
        # TimelineReporter.report()
        #
        #
        # # ============可观测性完整日志==============
        #
        # from src.runtime.observability.report_builder import (
        #     ReportBuilder
        # )
        #
        # from src.runtime.observability.report_printer import (
        #     ReportPrinter
        # )
        #
        # report = ReportBuilder.build(
        #     runtime_ctx.get()
        # )
        #
        # ReportPrinter.print(
        #     report
        # )

        # 打印日志 这里是经过处理的  不在打印完全版的timeline和report了
        print_runtime_reports_if_enabled(
            runtime=runtime_ctx.get()
        )

        # 销毁所有作用域scope
        await runtime_ctx.destroy()

        # 销毁清空checkpoint  否则回无限增长

        checkpoint = container.get(
            "checkpoint").manager
        checkpoint.clear_checkpoint(
            trace_id
        )

        return (

            history,

            state,

            gr.update(visible=False),

            ""
        )

async def resume_agent(
    confirm_value: str,
    history: list,
    state: dict,
    request: gr.Request
):

    session_id = request.session_hash

    if not state.get("pending"):

        history.append({
            "role": "assistant",
            "content": "没有待确认的操作。"
        })

        return (
            history,
            state,
            gr.update(visible=False),
            ""
        )

    # =========================
    # 恢复 trace_id
    # =========================

    trace_id = state["trace_id"]

    # trace_ctx.set_trace_id(trace_id)

    # trace_ctx.set_session_id(session_id)

    # trace_ctx.set_user_id(
    #     state.get("user_id", "unknown")
    # )

    # trace_ctx.set_component("resume_handler")


    # =========================
    # RuntimeContext
    # =========================

    # 恢复runtime_ctx
    from src.runtime.context import (
        runtime_ctx,
    )
    #
    # ctx = RuntimeContext(
    #
    #     trace_id=trace_id,
    #
    #     session_id=session_id,
    #
    #     user_id=state.get(
    #         "user_id",
    #         "unknown"
    #     ),
    #
    #     component="resume_handler"
    # )


    # 用persistence来回复runtime_ctx
    checkpoint = container.get(
        "checkpoint"
    ).manager

    ensure_trace_exists(
        trace_id
    )

    ctx = build_resume_runtime_context(
        trace_id=trace_id,
        session_id=session_id,
        state=state,
        checkpoint_manager=checkpoint,
    )

    await runtime_ctx.create(ctx)

    #todo: 这里手动写死恢复current_agent 因为我现在的项目只有走general_agent时才会出现interrupt的情况
    # 所以直接写死没问题  后续如果有新的分支走interrupt的话就将current_agent存到dogstate里用于恢复
    runtime_ctx.get().state().set_agent(
        "general_agent"
    )
    # =========================
    # 添加用户确认
    # =========================

    history.append({
        "role": "user",
        "content": f"确认选择：{confirm_value}"
    })

    # =========================
    # 恢复 graph
    # =========================

    result = await run_main_graph_with_result(
        confirm_value,
        thread_id=session_id,
        trace_id=trace_id,
        resume_value=confirm_value,
    )

    # =========================
    # 再次中断
    # =========================

    if isinstance(result, GraphInterruptResult):
        logger.warning("再次中断了!...")

        prompt = result.prompt

        state["pending_prompt"] = prompt

        history.append({
            "role": "assistant",
            "content": f"⚠️ 再次需要确认：{prompt}"
        })

        # 用persistence来回复runtime_ctx
        checkpoint = container.get(
            "checkpoint"
        ).manager

        ensure_trace_exists(
            trace_id
        )

        ctx = build_resume_runtime_context(
            trace_id=trace_id,
            session_id=session_id,
            state=state,
            checkpoint_manager=checkpoint,
        )

        await runtime_ctx.create(ctx)

        # todo: 这里手动写死恢复current_agent 因为我现在的项目只有走general_agent时才会出现interrupt的情况
        # 所以直接写死没问题  后续如果有新的分支走interrupt的话就将current_agent存到dogstate里用于恢复
        runtime_ctx.get().state().set_agent(
            "general_agent"
        )

        return (
            history,
            state,
            gr.update(visible=True),
            prompt
        )

    # =========================
    # 正常结束
    # =========================

    state["pending"] = False

    answer = (
        result.answer
        if isinstance(result, GraphFinalResult)
        else str(result)
    )

    history.append({
        "role": "assistant",
        "content": answer
    })

    # ==============时间线打印==============
    # TimelineReporter.report()
    #
    # # ==============可观测性完整日志==============
    # from src.runtime.observability.report_builder import (
    #     ReportBuilder
    # )
    #
    # from src.runtime.observability.report_printer import (
    #     ReportPrinter
    # )
    #
    # report = ReportBuilder.build(
    #     runtime_ctx.get()
    # )
    #
    # ReportPrinter.print(
    #     report
    # )

    # 打印日志 这里是打印经过处理的日志 而不是打印全部内容
    print_runtime_reports_if_enabled(
        runtime=runtime_ctx.get()
    )

    from src.runtime.scopes.metrics_scope import MetricsScope
    metrics_scope = runtime_ctx.get().service(MetricsScope).get_metrics()
    logger.info(
        "运行结束，"
        f"tools={metrics_scope.get('tool_count', 0)}, "
        f"llm={metrics_scope.get('llm_count', 0)}, "
        f"errors={metrics_scope.get('error_count', 0)}, "
        f"tool_latency={metrics_scope.get('tool_latency', 0)}"
    )

    checkpoint = container.get("checkpoint").manager
    checkpoint.clear_checkpoint(trace_id)

    await runtime_ctx.destroy()
    
    return (
        history,
        state,
        gr.update(visible=False),
        ""
    )


# 清空对话
def clear_all():
    return [], {}, gr.update(visible=False), ""

# Gradio 界面
with gr.Blocks(title="狗狗百科 AI Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐕 狗狗百科 AI Agent")
    gr.Markdown("基于 AKC 知识库，支持推荐、精确查询、通用问答、工具调用和长期记忆。")

    chatbot = gr.Chatbot(label="对话记录")
    msg = gr.Textbox(label="输入你的问题", placeholder="例如：推荐三种不爱叫的大型犬", scale=9)
    clear = gr.Button("清空对话")

    # 确认面板（初始隐藏）
    with gr.Row(visible=False) as confirm_row:
        confirm_prompt = gr.Textbox(label="确认信息", interactive=False, scale=6)
        confirm_input = gr.Textbox(label="请输入 1/2/3 或 y/n", scale=2)
        confirm_btn = gr.Button("提交确认", scale=1)

    # 状态存储
    session_state = gr.State({})

    # 事件绑定
    msg.submit(
        respond_and_process,
        [msg, chatbot, session_state],
        [chatbot, session_state, confirm_row, confirm_prompt]
    ).then(
        lambda: "", None, msg   # 清空输入框
    )

    confirm_btn.click(
        resume_agent,
        [confirm_input, chatbot, session_state],
        [chatbot, session_state, confirm_row, confirm_prompt]
    ).then(
        lambda: "", None, confirm_input   # 清空确认输入框
    )

    clear.click(
        clear_all,
        None,
        [chatbot, session_state, confirm_row, confirm_prompt]
    )

    # 示例问题
    gr.Examples(
        examples=["推荐三种不爱叫的大型犬", "金毛的性格是什么", "今天成都的天气怎么样", "我最喜欢金毛"],
        inputs=[msg]
    )

import asyncio


async def main():

    # 启动 Container
    await container.startup()

    # 启动 Gradio
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    # demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    try:

        asyncio.run(main())

    finally:

        asyncio.run(
            container.shutdown()
        )
