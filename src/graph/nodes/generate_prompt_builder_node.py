from __future__ import annotations

from typing import Any
from typing import Mapping

from src.graph.schemas.answer_strategy import (
    AnswerStrategy,
)


def build_generation_prompt(
        state: Mapping[str, Any],
        answer_strategy: AnswerStrategy,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
) -> str:
    """
    构建 generate_node 使用的 Prompt。

    功能：
        根据 answer_strategy 选择不同回答模板。

    技术名词：
        Prompt Builder：
            提示词构建器。负责把 question、context、memory、history 等信息拼成最终 Prompt。

        Context：
            上下文。这里主要指 RAG 检索上下文。

        Memory：
            用户长期记忆。用于个性化回答。

        Answer Strategy：
            回答策略。决定当前 Prompt 使用什么回答格式。

    参数：
        state:
            当前 DogState。

        answer_strategy:
            回答策略对象。

        context:
            生成答案使用的上下文文本。

        context_source:
            上下文来源，例如 rag_context / docs / empty。

        memory_text:
            用户长期记忆文本。

        history_text:
            历史对话文本。

    返回值：
        str:
            最终传给 LLM 的 Prompt 字符串。
    """

    question = str(
        state.get(
            "question",
            "",
        )
        or ""
    ).strip()

    if answer_strategy.task_type == "recommendation":
        return build_recommendation_prompt(
            question=question,
            context=context,
            context_source=context_source,
            memory_text=memory_text,
            history_text=history_text,
            answer_strategy=answer_strategy,
        )

    if answer_strategy.task_type == "comparison":
        return build_comparison_prompt(
            question=question,
            context=context,
            context_source=context_source,
            memory_text=memory_text,
            history_text=history_text,
            answer_strategy=answer_strategy,
        )

    if answer_strategy.task_type == "care_advice":
        return build_care_advice_prompt(
            question=question,
            context=context,
            context_source=context_source,
            memory_text=memory_text,
            history_text=history_text,
            answer_strategy=answer_strategy,
        )

    if answer_strategy.task_type == "exact_info":
        return build_exact_info_prompt(
            question=question,
            context=context,
            context_source=context_source,
            memory_text=memory_text,
            history_text=history_text,
            answer_strategy=answer_strategy,
        )

    return build_general_dog_qa_prompt(
        question=question,
        context=context,
        context_source=context_source,
        memory_text=memory_text,
        history_text=history_text,
        answer_strategy=answer_strategy,
    )


def build_exact_info_prompt(
        question: str,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
        answer_strategy: AnswerStrategy,
) -> str:
    """
    构建精确信息问答 Prompt。

    功能：
        用于回答某个犬种的事实信息，例如寿命、体型、性格、掉毛、吠叫等。

    参数：
        question:
            用户问题。

        context:
            RAG 检索上下文。

        context_source:
            上下文来源。

        memory_text:
            用户长期记忆。

        history_text:
            历史对话文本。

        answer_strategy:
            回答策略对象。

    返回值：
        str:
            Prompt 字符串。
    """

    return f"""
你是 Dog Agent Framework 的犬种知识助手。

你的任务是基于【检索上下文】回答用户的问题。

# 当前回答策略

task_type: {answer_strategy.task_type}
answer_style: {answer_strategy.answer_style}
strategy_reason: {answer_strategy.reason}
context_source: {context_source}

# 回答要求

1. 先直接回答用户问题。
2. 必须优先基于“检索上下文”回答。
3. 不要编造检索上下文中没有的信息。
4. 如果上下文不足，请明确说明“当前资料中没有足够信息”。
5. 如果问题涉及适合新手、公寓、小孩、训练、掉毛、吠叫等，请给出简短注意事项。
6. 回答使用中文。
7. 犬种英文名需要保留，必要时补充中文解释。
8. 不要输出和问题无关的大段百科内容。

# 回答格式

直接答案：
- ...

依据：
- ...

补充说明：
- ...

# 用户长期记忆

{memory_text or "无"}

# 历史对话

{history_text or "无"}

# 检索上下文

{context or "无可用上下文"}

# 用户问题

{question}
"""


def build_recommendation_prompt(
        question: str,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
        answer_strategy: AnswerStrategy,
) -> str:
    """
    构建推荐类 Prompt。

    功能：
        用于根据用户条件推荐犬种。

    参数：
        question:
            用户问题。

        context:
            RAG 检索上下文。

        context_source:
            上下文来源。

        memory_text:
            用户长期记忆。

        history_text:
            历史对话文本。

        answer_strategy:
            回答策略对象。

    返回值：
        str:
            Prompt 字符串。
    """

    return f"""
你是 Dog Agent Framework 的犬种推荐助手。

你的任务是基于【检索上下文】给用户推荐合适的犬种。

# 当前回答策略

task_type: {answer_strategy.task_type}
answer_style: {answer_strategy.answer_style}
strategy_reason: {answer_strategy.reason}
context_source: {context_source}

# 回答要求

1. 只能基于检索上下文推荐，不要编造不存在的犬种资料。
2. 优先推荐最符合用户条件的犬种。
3. 每个推荐都要说明推荐理由。
4. 如果某个犬种有明显不适合点，也要提醒。
5. 如果上下文不足，请明确说明推荐依据有限。
6. 回答使用中文。
7. 犬种英文名需要保留，必要时补充中文名。
8. 不要推荐检索上下文中没有出现的犬种。

# 回答格式

推荐结论：
- 最推荐：...

推荐列表：
1. 犬种英文名
   - 推荐理由：
   - 需要注意：
   - 适合人群：

2. 犬种英文名
   - 推荐理由：
   - 需要注意：
   - 适合人群：

选择建议：
- ...

# 用户长期记忆

{memory_text or "无"}

# 历史对话

{history_text or "无"}

# 检索上下文

{context or "无可用上下文"}

# 用户问题

{question}
"""


def build_comparison_prompt(
        question: str,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
        answer_strategy: AnswerStrategy,
) -> str:
    """
    构建对比类 Prompt。

    功能：
        用于比较两个或多个犬种。

    参数：
        question:
            用户问题。

        context:
            RAG 检索上下文。

        context_source:
            上下文来源。

        memory_text:
            用户长期记忆。

        history_text:
            历史对话文本。

        answer_strategy:
            回答策略对象。

    返回值：
        str:
            Prompt 字符串。
    """

    return f"""
你是 Dog Agent Framework 的犬种对比助手。

你的任务是基于【检索上下文】比较犬种差异。

# 当前回答策略

task_type: {answer_strategy.task_type}
answer_style: {answer_strategy.answer_style}
strategy_reason: {answer_strategy.reason}
context_source: {context_source}

# 回答要求

1. 只基于检索上下文回答。
2. 按清晰维度比较，例如体型、精力、训练难度、掉毛、吠叫、适合人群。
3. 如果某个维度上下文不足，请明确说明。
4. 最后给出选择建议。
5. 回答使用中文。
6. 不要强行比较上下文中没有提供的信息。

# 回答格式

对比结论：
- ...

维度对比：
- 体型：
- 精力：
- 训练难度：
- 掉毛：
- 吠叫：
- 适合人群：

选择建议：
- 如果你更在意 ...，可以选 ...
- 如果你更在意 ...，可以选 ...

# 用户长期记忆

{memory_text or "无"}

# 历史对话

{history_text or "无"}

# 检索上下文

{context or "无可用上下文"}

# 用户问题

{question}
"""


def build_care_advice_prompt(
        question: str,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
        answer_strategy: AnswerStrategy,
) -> str:
    """
    构建护理 / 训练 / 饲养建议 Prompt。

    功能：
        用于回答狗狗护理、训练、喂养相关问题。

    参数：
        question:
            用户问题。

        context:
            RAG 检索上下文。

        context_source:
            上下文来源。

        memory_text:
            用户长期记忆。

        history_text:
            历史对话文本。

        answer_strategy:
            回答策略对象。

    返回值：
        str:
            Prompt 字符串。
    """

    return f"""
你是 Dog Agent Framework 的狗狗护理建议助手。

你的任务是基于【检索上下文】给出护理、训练或饲养建议。

# 当前回答策略

task_type: {answer_strategy.task_type}
answer_style: {answer_strategy.answer_style}
strategy_reason: {answer_strategy.reason}
context_source: {context_source}

# 回答要求

1. 优先基于检索上下文回答。
2. 给出可执行步骤。
3. 如果涉及健康、疾病、异常行为，不要替代兽医诊断。
4. 对风险点要提醒。
5. 回答使用中文。
6. 如果上下文不足，请说明建议依据有限。

# 回答格式

建议结论：
- ...

具体做法：
1. ...
2. ...
3. ...

注意事项：
- ...

什么时候需要找专业人士：
- ...

# 用户长期记忆

{memory_text or "无"}

# 历史对话

{history_text or "无"}

# 检索上下文

{context or "无可用上下文"}

# 用户问题

{question}
"""


def build_general_dog_qa_prompt(
        question: str,
        context: str,
        context_source: str,
        memory_text: str,
        history_text: str,
        answer_strategy: AnswerStrategy,
) -> str:
    """
    构建普通狗狗知识问答 Prompt。

    功能：
        用于兜底回答普通狗狗知识问题。

    参数：
        question:
            用户问题。

        context:
            RAG 检索上下文。

        context_source:
            上下文来源。

        memory_text:
            用户长期记忆。

        history_text:
            历史对话文本。

        answer_strategy:
            回答策略对象。

    返回值：
        str:
            Prompt 字符串。
    """

    return f"""
你是 Dog Agent Framework 的狗狗知识助手。

你的任务是基于【检索上下文】回答用户问题。

# 当前回答策略

task_type: {answer_strategy.task_type}
answer_style: {answer_strategy.answer_style}
strategy_reason: {answer_strategy.reason}
context_source: {context_source}

# 回答要求

1. 优先使用检索上下文。
2. 不要编造上下文中没有的事实。
3. 如果上下文不足，请说明资料有限。
4. 回答要通俗易懂。
5. 回答使用中文。

# 回答格式

回答：
- ...

补充说明：
- ...

# 用户长期记忆

{memory_text or "无"}

# 历史对话

{history_text or "无"}

# 检索上下文

{context or "无可用上下文"}

# 用户问题

{question}
"""