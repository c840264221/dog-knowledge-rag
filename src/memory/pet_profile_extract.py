"""从用户自然语言中批量抽取结构化宠物档案候选事实。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.logger import logger
from src.memory.memory_schema import (
    PetProfileExtractionResult,
    PetProfileFactCandidate,
)
from src.runtime.observability.llm_call_records import (
    LLMCallPurpose,
    build_llm_call_metadata,
)


MAX_PET_PROFILE_CANDIDATES = 20


PET_PROFILE_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
你是企业宠物服务系统的结构化档案提取器。

目标：
从用户本轮输入中提取用户明确陈述的宠物档案事实。一次输入可能包含多条
事实，必须全部拆开输出。不要推测用户没有明确提供的信息。

attribute 只能使用以下值：
- breed：品种
- birth_date：出生日期
- age_years：年龄（年）
- weight_kg：体重（公斤）
- sex：性别
- neutered：是否绝育
- health_condition：健康状况或疾病
- allergy：过敏信息
- diet_pattern：长期饮食模式
- activity_level：日常活动水平
- training_goal：持续性的训练目标

字段规则：
1. subject_reference 保留用户原文中的宠物称呼，例如“豆豆”“它”“我家狗”。
2. 不要自行生成 pet_id、pet_key 或用户编号。
3. value 使用简洁值；年龄只写数字，体重只写公斤数值。
4. evidence_text 复制支持该事实的最短原文片段，不要写推理过程。
5. 普通知识问题、假设案例和助手生成的建议不能成为档案事实。
6. 同一句包含多个属性时，每个属性生成一条 facts 记录。
7. 无法确定对象归属时仍保留原始称呼，后续由实体解析器处理。

严格输出 JSON，不要输出 Markdown：
{{
  "facts": [
    {{
      "subject_reference": "豆豆",
      "attribute": "breed",
      "value": "金毛",
      "confidence": 0.98,
      "evidence_text": "豆豆是一只金毛"
    }}
  ],
  "reason": "用户明确提供了宠物档案信息"
}}

没有明确档案事实时输出：
{{
  "facts": [],
  "reason": "当前输入没有明确的宠物档案事实"
}}

用户输入：
{user_text}
"""
)


def normalize_pet_profile_extraction_result(
    raw_result: Mapping[str, Any] | None,
) -> PetProfileExtractionResult:
    """
    逐条校验并整理宠物档案批量抽取结果。

    功能：
        遍历模型返回的 facts，只保留符合 PetProfileFactCandidate 契约的
        候选；某一条非法时记录拒绝数量，但继续处理同批其他候选。

    参数含义：
        raw_result：JsonOutputParser 解析出的原始模型结果。

    返回值含义：
        PetProfileExtractionResult：合法候选、拒绝数量和抽取原因。
    """

    source = raw_result if isinstance(raw_result, Mapping) else {}
    raw_facts = source.get("facts", [])
    if not isinstance(raw_facts, list):
        raw_facts = []

    accepted_facts: list[PetProfileFactCandidate] = []
    rejected_candidate_count = max(
        0,
        len(raw_facts) - MAX_PET_PROFILE_CANDIDATES,
    )
    for raw_fact in raw_facts[:MAX_PET_PROFILE_CANDIDATES]:
        if not isinstance(raw_fact, Mapping):
            rejected_candidate_count += 1
            continue
        try:
            accepted_facts.append(
                PetProfileFactCandidate.model_validate(dict(raw_fact))
            )
        except ValidationError as exc:
            rejected_candidate_count += 1
            logger.warning(
                "宠物档案候选未通过契约校验，已拒绝单条候选: "
                f"{exc.errors(include_url=False)}"
            )

    return PetProfileExtractionResult(
        facts=accepted_facts,
        rejected_candidate_count=rejected_candidate_count,
        reason=str(source.get("reason") or "").strip(),
    )


def default_pet_profile_extraction_result(
    reason: str,
) -> PetProfileExtractionResult:
    """
    创建不包含候选事实的安全降级结果。

    参数含义：
        reason：未执行或抽取失败的具体原因。

    返回值含义：
        PetProfileExtractionResult：facts 为空的稳定结果。
    """

    return PetProfileExtractionResult(facts=[], reason=reason)


async def extract_pet_profile_facts(
    *,
    llm_provider: Any,
    user_text: str,
) -> PetProfileExtractionResult:
    """
    使用 LLM 批量抽取宠物档案候选事实。

    功能：
        调用统一 LLMProvider，解析 JSON，并逐条执行 Pydantic 契约校验；
        模型调用、JSON 解析或整体流程失败时返回空候选，不抛出到主图。

    参数含义：
        llm_provider：项目统一的大语言模型服务提供者。
        user_text：用户本轮提供的原始自然语言。

    返回值含义：
        PetProfileExtractionResult：批量候选事实和局部拒绝统计。
    """

    clean_user_text = str(user_text or "").strip()
    if not clean_user_text:
        return default_pet_profile_extraction_result("用户输入为空")

    fallback_response = (
        '{"facts": [], "reason": "宠物档案模型调用失败"}'
    )

    async def safe_llm_call(prompt_value: Any) -> str:
        """
        通过统一 Provider 调用模型并返回文本内容。

        参数含义：
            prompt_value：模板渲染后的 LangChain PromptValue。

        返回值含义：
            str：供 JsonOutputParser 解析的模型响应文本。
        """

        response = await llm_provider.safe_ainvoke(
            llm=llm_provider.chinese_llm,
            prompt=prompt_value.to_string(),
            fallback_response=fallback_response,
            call_metadata=build_llm_call_metadata(
                purpose=LLMCallPurpose.PET_PROFILE_EXTRACTION,
                component="pet_profile_extract",
            ),
        )
        return str(getattr(response, "content", response))

    chain = (
        PET_PROFILE_EXTRACTION_PROMPT
        | RunnableLambda(safe_llm_call)
        | JsonOutputParser()
    )
    try:
        raw_result = await chain.ainvoke({"user_text": clean_user_text})
        result = normalize_pet_profile_extraction_result(raw_result)
        logger.info(
            "宠物档案候选抽取完成: "
            f"accepted={len(result.facts)}, "
            f"rejected={result.rejected_candidate_count}"
        )
        return result
    except Exception as exc:
        logger.warning(f"宠物档案候选抽取失败，已安全降级: {exc}")
        return default_pet_profile_extraction_result("宠物档案抽取失败")
