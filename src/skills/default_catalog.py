"""Dog Agent Framework 默认 Skill（技能）目录。"""

from __future__ import annotations

from src.skills.extractors import extract_dog_training_plan_inputs
from src.skills.input_checker import SkillInputChecker
from src.skills.input_extractor import SkillInputExtractor
from src.skills.loader import SkillLoader
from src.skills.registry import SkillRegistry
from src.skills.runtime import SkillRuntime
from src.skills.schemas import SkillDefinition, SkillInputRequirement
from src.skills.selector import SkillSelector


DOG_TRAINING_PLAN_SKILL = SkillDefinition(
    skill_id="dog-training-plan",
    name="狗狗训练计划",
    description="根据狗狗档案和训练目标制定分阶段、可执行的训练计划。",
    activation_hints=[
        "狗狗训练计划",
        "制定训练计划",
        "定制训练计划",
        "训练计划",
        "训练方案",
        "行为训练计划",
    ],
    required_inputs=[
        SkillInputRequirement(
            input_id="breed",
            name="犬种",
            description="狗狗的犬种或混血情况。",
        ),
        SkillInputRequirement(
            input_id="age",
            name="年龄",
            description="狗狗当前年龄或年龄阶段。",
        ),
        SkillInputRequirement(
            input_id="current_behavior",
            name="当前行为基础",
            description="已经掌握的指令和当前存在的行为问题。",
        ),
        SkillInputRequirement(
            input_id="training_goal",
            name="训练目标",
            description="本次最希望改善或建立的行为。",
        ),
    ],
    instructions=[
        "检查狗狗档案和训练目标是否完整，缺失关键信息时先提出澄清问题。",
        "根据犬种、年龄和当前行为基础识别训练难度与注意事项。",
        "把训练目标拆分成循序渐进的阶段，并为每个阶段设置可观察目标。",
        "生成每日训练频率、单次时长、奖励方式和调整条件。",
        "检查计划是否符合正向强化原则，并明确需要专业人士介入的情况。",
    ],
    allowed_tools=[],
    output_contract=(
        "输出档案摘要、阶段目标、每日安排、奖励方式、进度判断和注意事项。"
    ),
    guardrails=[
        "不得使用体罚、恐吓或可能伤害狗狗的训练方式。",
        "涉及攻击、严重焦虑或健康异常时，应建议咨询专业训犬师或兽医。",
        "资料不足时必须说明限制，不得编造狗狗档案。",
    ],
    version="1.0.0",
)


def build_default_skill_registry() -> SkillRegistry:
    """
    构建项目默认技能注册表。

    功能：
        每次调用都创建独立注册表，并注册当前项目内置的启用技能，避免测试
        和不同运行实例共享可变注册表状态。

    参数含义：
        无。

    返回值含义：
        SkillRegistry:
            已注册默认技能的新注册表实例。
    """

    return SkillRegistry([DOG_TRAINING_PLAN_SKILL])


def build_default_skill_input_extractor(
    registry: SkillRegistry | None = None,
) -> SkillInputExtractor:
    """
    构建项目默认技能输入提取器。

    功能：
        为 dog-training-plan 注册确定性自然语言提取规则，并允许调用方复用
        已有技能注册表，保证 Selector、Loader 和 Extractor 使用同一份技能定义。

    参数含义：
        registry:
            可选的已有技能注册表；为空时创建默认注册表。

    返回值含义：
        SkillInputExtractor:
            已注册训练计划提取规则的输入提取器。
    """

    resolved_registry = registry or build_default_skill_registry()
    return SkillInputExtractor(
        loader=SkillLoader(resolved_registry),
        rules={
            "dog-training-plan": extract_dog_training_plan_inputs,
        },
    )


def build_default_skill_runtime(
    registry: SkillRegistry | None = None,
) -> SkillRuntime:
    """
    构建使用项目默认技能目录的 Skill 运行器。

    功能：
        使用同一份注册表组装 Selector、Extractor、Checker 和 Loader，形成
        可以直接执行技能选择与输入准备的完整调用链。

    参数含义：
        registry:
            可选的已有技能注册表；为空时创建默认注册表。

    返回值含义：
        SkillRuntime:
            已装配默认训练技能和确定性提取规则的技能运行器。
    """

    # 调用方传入的注册表优先；没有传入时才创建项目默认技能目录。
    resolved_registry = registry or build_default_skill_registry()

    # Loader 负责读取同一份注册表，Extractor、Checker 和 Runtime 都复用它。
    loader = SkillLoader(resolved_registry)
    return SkillRuntime(
        selector=SkillSelector(resolved_registry),
        extractor=SkillInputExtractor(
            loader=loader,
            rules={
                "dog-training-plan": extract_dog_training_plan_inputs,
            },
        ),
        checker=SkillInputChecker(loader),
        loader=loader,
    )
