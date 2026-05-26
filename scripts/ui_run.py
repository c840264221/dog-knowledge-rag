import gradio as gr
from src.graph.graph_run import run_main_graph_with_stream
from src.logger import logger

import src.runtime.events.setup

from src.runtime.trace import trace_ctx

from src.runtime.trace.init import trace_manager

import uuid


# 用于存储每个会话的配置信息和中断状态（使用 gr.State 更安全）
# 但 State 需要绑定到界面，我们直接在函数中使用 gr.State 对象
# async def respond_and_process(question: str, history: list, state: dict, request: gr.Request):
#     """
#     核心处理函数：
#     - 调用 Agent
#     - 处理中断/恢复
#     - 返回 (新历史, 新状态, 确认面板可见性, 确认提示文本)
#     """
#     session_id = request.session_hash
#
#     # 生成 trace_id（也可以使用 uuid 或 trace_manager 的 create_trace）
#     trace_id, _ = trace_manager.create_trace()
#
#     # 设置上下文变量
#     trace_ctx.set_trace_id(trace_id)
#     trace_ctx.set_session_id(session_id)
#
#     user_id = state.get("user_id") if state else None
#     if user_id:
#         trace_ctx.set_user_id(user_id)
#     else:
#         trace_ctx.set_user_id("unknown")
#
#     trace_ctx.set_component("gradio_handler")
#
#     # 初始化或获取状态
#     if not state:
#         state = {"config": {"configurable": {"thread_id": session_id}}, "pending": False, "pending_prompt": ""}
#     else:
#         # 确保 config 中的 thread_id 与当前会话一致
#         state["config"]["configurable"]["thread_id"] = session_id
#
#     # 将用户问题也添加到聊天框中
#     history.append({"role": "user", "content": question})
#
#     # 调用 Agent
#     result = await run_main_graph_with_stream(question, thread_id=session_id)
#
#     # 判断是否中断
#     if result.startswith("__INTERRUPT__:"):
#         prompt = result[len("__INTERRUPT__:"):].strip()
#         # 更新状态：标记等待确认
#         state["pending"] = True
#         state["pending_prompt"] = prompt
#         # 在聊天记录中添加一条系统提示（可选）
#         history.append({"role": "assistant", "content": f"⚠️ 需要确认：{prompt}"})
#         # 返回历史、状态、显示确认面板、设置确认框的提示文本
#         return history, state, gr.update(visible=True), prompt
#     else:
#         # 正常答案，且没有等待确认
#         if state.get("pending"):
#             # 可能恢复后已经解决了，但确保清除标志
#             state["pending"] = False
#         history.append({"role": "assistant", "content": result})
#         return history, state, gr.update(visible=False), ""

async def respond_and_process(
    question: str,
    history: list,
    state: dict,
    request: gr.Request
):

    session_id = request.session_hash

    # ===== 创建 trace =====

    trace_id = str(uuid.uuid4())

    trace_manager.create_trace(trace_id)

    # ===== 写入 contextvars =====

    trace_ctx.set_trace_id(trace_id)

    trace_ctx.set_session_id(session_id)

    trace_ctx.set_component("gradio_handler")

    # ===== user_id =====

    user_id = state.get("user_id") if state else None

    if user_id:

        trace_ctx.set_user_id(user_id)

    else:

        trace_ctx.set_user_id("unknown")

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

    result = await run_main_graph_with_stream(

        question,

        thread_id=session_id,

        trace_id=trace_id
    )

    # ===== interrupt =====

    if result.startswith("__INTERRUPT__:"):

        prompt = result[len("__INTERRUPT__:"):].strip()

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

        history.append({

            "role": "assistant",

            "content": result
        })

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

    trace_ctx.set_trace_id(trace_id)

    trace_ctx.set_session_id(session_id)

    trace_ctx.set_user_id(
        state.get("user_id", "unknown")
    )

    trace_ctx.set_component("resume_handler")

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

    resume_msg = f"RESUME:{confirm_value}"

    result = await run_main_graph_with_stream(
        resume_msg,
        thread_id=session_id,
        trace_id=trace_id
    )

    # =========================
    # 再次中断
    # =========================

    if result.startswith("__INTERRUPT__:"):

        prompt = result[len("__INTERRUPT__:"):].strip()

        state["pending_prompt"] = prompt

        history.append({
            "role": "assistant",
            "content": f"⚠️ 再次需要确认：{prompt}"
        })

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

    history.append({
        "role": "assistant",
        "content": result
    })

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

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)