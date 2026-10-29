"""记账文件式Skill的薄适配：固定资源读取与现有技能结构复用，不实现通用发现框架。"""

import hashlib
import json
from pathlib import Path


SKILL_DIRECTORY = Path(__file__).resolve().parents[3] / "skills/expense-recording"


def load_expense_category_reference() -> dict:
    """读取分类参考；无参数；返回合法目录与说明，读取失败或目录结构错误时抛出异常。

    参考路径由程序固定，不允许模型提供文件路径。当前一级分类仍受工具契约约束，
    增加一级分类需要同步调整契约；本函数不代表任意分类都能动态接入。
    """
    reference = json.loads((SKILL_DIRECTORY / "references/categories.json").read_text(encoding="utf-8"))
    categories = reference.get("categories")
    if not isinstance(categories, dict) or set(categories) != {"生活", "出行", "娱乐", "宠物", "其他"}:
        raise ValueError("记账分类目录与现有工具契约不一致。")
    for category, values in categories.items():
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"分类{category}的二级目录无效。")
        if len(values) != len(set(values)):
            raise ValueError(f"分类{category}的二级目录重复。")
    return reference


def load_expense_categories() -> dict[str, tuple[str, ...]]:
    """为现有参数校验读取分类；无参数；返回与原EXPENSE_CATEGORIES兼容的分类到元组映射。"""
    return {key: tuple(values) for key, values in load_expense_category_reference()["categories"].items()}


def build_expense_skill_context() -> tuple[str, dict]:
    """将文件式技能适配为项目技能上下文，不调用模型或执行工具。

    参数：无，加载固定记账技能和分类参考。
    返回值：渲染上下文及加载元信息。复用SkillDefinition、SkillRegistry、SkillLoader，
        但本步显式选定技能，不改默认选择器，也不声称已接通完整主图或SkillRuntime。
    """
    import yaml
    from src.skills.loader import SkillLoader
    from src.skills.registry import SkillRegistry
    from src.skills.schemas import SkillDefinition, SkillInputRequirement

    skill_path = SKILL_DIRECTORY / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("Skill缺少名称与说明的YAML头部。")
    header, separator, body = text[4:].partition("\n---\n")
    if not separator or not body.strip():
        raise ValueError("Skill头部或正文不完整。")
    manifest = yaml.safe_load(header)
    if not isinstance(manifest, dict) or manifest.get("name") != "expense-recording":
        raise ValueError("加载的不是预期记账技能。")
    skill = SkillDefinition(
        skill_id=manifest["name"], name="自然语言支出记账", description=manifest["description"],
        version=manifest["metadata"]["version"],
        instructions=[body.strip()],
        required_inputs=[
            SkillInputRequirement(input_id=key, name=name)
            for key, name in [
                ("amount", "支出金额"), ("expense_date", "账单日期"),
                ("category", "一级分类"), ("subcategory", "二级分类"), ("description", "用途"),
            ]
        ],
        allowed_tools=["record_expense", "record_expenses"],
        output_contract="提取阶段只输出宿主提供的单笔或批量提议结构；执行阶段由程序确认并校验回执。",
    )
    context = SkillLoader(SkillRegistry([skill])).render_context(skill.skill_id)
    reference = load_expense_category_reference()
    context += "\n\n以下分类参考已由程序读取，无须额外发起文件读取：\n" + json.dumps(reference, ensure_ascii=False)
    metadata = {
        "skill_id": skill.skill_id,
        "version": skill.version,
        "skill_path": str(skill_path),
        "reference_path": str(SKILL_DIRECTORY / "references/categories.json"),
        "loaded_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "loading_mode": "explicit_skill_with_preloaded_reference",
    }
    return context, metadata
