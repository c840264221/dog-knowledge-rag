import json

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.logger import logger
from src.memory.memory_retrieve import retrieve_user_memory
from src.runtime.context import runtime_ctx
from src.runtime.scopes.retrieval_scope import RetrievalScope


def build_context(
        docs
):
    """
    构建 Prompt 上下文数据。

    功能：
    - 将检索到的 Document 列表转换成 JSON 字符串
    - 提取狗狗名称、结构化字段和正文内容
    - 控制正文长度，避免 Prompt 过长

    参数：
    - docs:
      LangChain Document 列表。
      中文释义：检索阶段返回的文档数据。

    返回值：
    - str
      JSON 字符串格式的上下文数据。
    """

    context = []

    for doc in docs:

        item = {
            "name": doc.metadata.get(
                "name"
            ),
            "structured": {
                "barking": doc.metadata.get(
                    "barking"
                ),
                "trainability": doc.metadata.get(
                    "trainability"
                ),
                "shedding": doc.metadata.get(
                    "shedding"
                ),
            },
            "text": doc.page_content[:300]
        }

        context.append(
            item
        )

    return json.dumps(
        context,
        ensure_ascii=False,
        indent=2
    )


def build_generate_node(
        llm_provider,
        memory_provider=None,
        checkpoint_provider=None
):
    """
    构建 generate_node 节点函数。

    功能：
    - 使用闭包方式注入 LLMProvider、MemoryProvider、CheckpointProvider
    - 避免 generate_node 内部直接 import container
    - 让节点符合 v1.2.1 container/provider 架构规范

    技术名词：
    - Closure：闭包，指内部函数可以使用外部函数传入的变量
    - Provider：提供者，负责统一创建和管理服务对象
    - Node：节点，LangGraph 中执行某个业务步骤的函数

    参数：
    - llm_provider:
      LLMProvider 实例。
      中文释义：用于获取主模型 main_llm，并执行安全 LLM 调用。

    - memory_provider:
      MemoryProvider 实例。
      中文释义：用于召回用户长期记忆。

    - checkpoint_provider:
      CheckpointProvider 实例。
      中文释义：用于保存运行时 checkpoint 检查点。

    返回值：
    - callable
      返回一个 async generate_node 函数，供 LangGraph 注册使用。
    """

    async def generate_node(
            state
    ):
        """
        生成最终回答。

        功能：
        - 从 runtime_ctx 获取检索作用域中的 docs
        - 召回用户长期记忆
        - 构建 Prompt
        - 调用 LLM 生成回答
        - 保存 checkpoint
        - 返回 answer 和 messages

        参数：
        - state:
          LangGraph 当前状态。
          中文释义：包含 question、intent、docs、messages、user_id 等字段。

        返回值：
        - dict
          返回 answer 和 messages，用于合并回 LangGraph state。
        """

        runtime = runtime_ctx.get()

        runtime.state().set_node(
            "generate_node"
        )

        runtime.timeline().add_event(
            event_type="node",
            name="generate_node"
        )

        retrieval_scope = runtime.service(
            RetrievalScope
        )

        docs = retrieval_scope.get_docs()

        logger.debug(
            f"获取 runtime_ctx 中 retrieval 作用域 docs，数量为：{len(docs)}"
        )

        main_llm = llm_provider.main_llm

        logger.info(
            "进入 generate_node 节点，state 为："
            f"question:{state['question']}, "
            f"intent:{state['intent']}, "
            f"strategy:{state['strategy']}, "
            f"filters:{state['filters']}, "
            f"tags:{state['tags']}, "
            f"dog_name:{state['dog_name']}, "
            f"docs len:{len(state['docs'])}, "
            f"user_id:{state['user_id']}"
        )

        if memory_provider is not None:

            memory_text = await retrieve_user_memory(
                user_id=state["user_id"],
                question=state["question"],
                memory_provider=memory_provider,
                limit=10
            )

        else:

            memory_text = "暂无用户记忆"

        context = build_context(
            state["docs"]
        )

        simplified_logger_context = [
            {
                "text": len(
                    doc.page_content
                ),
                "metadata": doc.metadata
            }
            for doc in state["docs"]
        ]

        logger.debug(
            f"context数量: {simplified_logger_context}"
        )

        prompt = ChatPromptTemplate.from_template(
            """
你是一个严谨的狗狗百科助手。

# 用户长期记忆（Memory）
{memory_text}

【任务】
根据 intent 决定行为：
- recommendation → 推荐狗狗
- 其他 → 回答问题

【严格要求】
1. 必须使用数据中的 "name" 字段作为狗狗名称
2. 严禁使用“品种一/二”等编号
3. 只能基于提供数据，不得编造
4. 每条推荐必须包含名称 + 原因
5. 至少3条数据，但最多不超过5条
6. 如果 intent 不是 general，只回答该 intent 相关内容

intent: {intent}

数据（JSON格式）：
{context}

历史信息：
{history_text}

问题：
{question}

输出规则：
- 如果是推荐：最多5个，名称+原因
- 如果是问答：直接回答，不要推荐
"""
        )

        history_text = "\n".join(
            [
                f"用户: {message.content}"
                if isinstance(
                    message,
                    HumanMessage
                )
                else f"助手: {message.content}"
                for message in state.get(
                    "messages",
                    []
                )
            ]
        )

        logger.debug(
            f"history_text:{history_text}"
        )

        async def create_async_safe_llm_ainvoke(
                prompt_value
        ):
            """
            安全调用 LLM。

            功能：
            - 通过 LLMProvider 调用 safe_ainvoke
            - 支持失败兜底
            - 避免节点直接操作底层模型调用细节

            参数：
            - prompt_value:
              Prompt 模板渲染后的输入。

            返回值：
            - str
              LLM 返回的文本结果。
            """

            return await llm_provider.safe_ainvoke(
                llm=main_llm,
                prompt=prompt_value,
                fallback_response="调用LLM失败"
            )

        safe_llm = RunnableLambda(
            create_async_safe_llm_ainvoke
        )

        answer = await (
                prompt
                | safe_llm
                | StrOutputParser()
        ).ainvoke(
            {
                "memory_text": memory_text,
                "intent": state["intent"],
                "context": context,
                "question": state["question"],
                "history_text": history_text
            }
        )

        messages = state.get(
            "messages",
            []
        )

        messages.append(
            AIMessage(
                content=answer
            )
        )

        logger.info(
            f"generate_node 节点完成，结果 answer 为：{answer}"
        )

        logger.debug(
            f"Runtime State:{runtime.state().get_state()}"
        )

        if checkpoint_provider is not None:

            checkpoint_provider.manager.save_checkpoint()

        return {
            "answer": answer,
            "messages": messages
        }

    return generate_node