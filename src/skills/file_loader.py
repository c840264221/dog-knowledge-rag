"""通用文件式技能加载、Agent内筛选及上下文构建，不要求技能使用工具。"""

import hashlib
from pathlib import Path

import yaml

from src.skills.loader import SkillLoader
from src.skills.registry import SkillRegistry
from src.skills.schemas import SkillDefinition
from src.skills.selector import SkillSelector


def load_file_skill_registry(skill_files) -> tuple[SkillRegistry, dict]:
    """读取宿主指定的技能文件。

    参数：skill_files为可信文件路径列表，相对路径以项目根目录为基准。
    返回值：统一技能注册表及文件信息映射；不按Agent或工具过滤，也不调用模型。
    未声明allowed_tools的技能照常注册；输出要求由文件或宿主决定，不固定为工具调用结构。
    """
    root = Path(__file__).resolve().parents[2]
    registry = SkillRegistry()
    locations = {}
    for filename in skill_files:
        path = (root / filename).resolve()
        text = path.read_text(encoding="utf-8-sig")
        if not text.startswith("---\n"):
            raise ValueError(f"技能缺少YAML头部：{path}")
        header, separator, body = text[4:].partition("\n---\n")
        manifest = yaml.safe_load(header)
        if not separator or not body.strip() or not isinstance(manifest, dict):
            raise ValueError(f"技能头部或正文无效：{path}")
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("技能metadata必须是字典。")
        target_agents = metadata.get("target_agents", [])
        if not isinstance(target_agents, list) or any(not isinstance(name, str) or not name.strip() for name in target_agents):
            raise ValueError("target_agents必须是非空Agent名称组成的列表。")
        skill = SkillDefinition(
            skill_id=manifest["name"], name=metadata.get("display_name", manifest["name"]),
            description=manifest["description"], version=metadata.get("version", "1.0.0"),
            activation_hints=metadata.get("activation_hints", []), instructions=[body.strip()],
            allowed_tools=metadata.get("allowed_tools", []), required_inputs=metadata.get("required_inputs", []),
            output_contract=metadata.get("output_contract", "遵守当前宿主提供的输出契约。"),
            enabled=metadata.get("enabled", True),
        )
        registry.register(skill)
        locations[skill.skill_id] = {
            "path": path, "references": metadata.get("preload_references", []),
            "target_agents": target_agents,
        }
    return registry, locations


def select_agent_skill(question: str, agent_name: str, registry: SkillRegistry,
                       locations: dict, tool_names: set[str]) -> SkillDefinition | None:
    """在已确定的Agent内部选择一次技能。

    参数：question为问题，agent_name为粗路由已确定的Agent，registry及locations为已加载目录，
        tool_names为该Agent可用工具集合；无工具能力时传空集合。
    返回值：选中的SkillDefinition或None；先按Agent和声明的工具依赖过滤，再复用已有选择器。
    空工具列表表示没有工具依赖，不会因此排除技能；不检查或授予用户权限。
    """
    candidates = []
    for skill in registry.list_enabled():
        targets = locations[skill.skill_id]["target_agents"]
        if targets and agent_name not in targets:
            continue
        if not set(skill.allowed_tools).issubset(tool_names):
            continue
        candidates.append(skill)
    filtered = SkillRegistry(candidates)
    selected = SkillSelector(filtered).select(question).selected_skill_id
    return SkillLoader(filtered).load(selected) if selected else None


def build_file_skill_context(skill: SkillDefinition | None, registry: SkillRegistry,
                             locations: dict) -> tuple[str, dict]:
    """根据已选技能构建上下文，不再次选择。

    参数：skill为上一步选择结果，registry及locations必须来自同次目录加载。
    返回值：规则和参考资料文本、加载元信息；skill为None时返回空文本及空字典。
    参考文件必须位于技能目录内部；读取文本不执行脚本，也不调用模型。
    """
    if skill is None:
        return "", {}
    context = SkillLoader(registry).render_context(skill.skill_id)
    location = locations[skill.skill_id]
    path, references = location["path"], location["references"]
    if not isinstance(references, list) or any(not isinstance(name, str) for name in references):
        raise ValueError("preload_references必须是宿主配置的相对路径列表。")
    reference_paths = []
    for filename in references:
        reference = (path.parent / filename).resolve()
        if not reference.is_relative_to(path.parent):
            raise ValueError("参考文件不能越出当前技能目录。")
        context += f"\n\n宿主已读取参考资料 {filename}：\n" + reference.read_text(encoding="utf-8-sig")
        reference_paths.append(str(reference))
    return context, {
        "skill_id": skill.skill_id, "version": skill.version, "skill_path": str(path),
        "reference_paths": reference_paths, "allowed_tools": skill.allowed_tools,
        "loaded_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "loading_mode": "selected_skill_with_preloaded_references",
    }
