"""ToolAgent的技能接入点；复用通用文件加载能力，不复制加载、选择或渲染实现。"""

from src.skills.file_loader import load_file_skill_registry, select_agent_skill, build_file_skill_context
from src.skills.registry import SkillRegistry
from src.skills.schemas import SkillDefinition


def load_tool_planning_registry(skill_files=None) -> tuple[SkillRegistry, dict]:
    """读取工具Agent配置的技能文件；skill_files为可选可信路径列表；返回注册表和文件信息。"""
    if skill_files is None:
        from src.settings import settings
        skill_files = settings.tool.planning_skill_files
    return load_file_skill_registry(skill_files)


def select_tool_planning_skill(question: str, registry: SkillRegistry, locations: dict,
                               tool_names: set[str]) -> SkillDefinition | None:
    """在ToolAgent内选择技能；参数为问题、已加载目录及可用工具；返回技能或None，不重读文件。"""
    return select_agent_skill(question, "tool_agent", registry, locations, tool_names)


def build_tool_planning_context(skill: SkillDefinition | None, registry: SkillRegistry,
                                locations: dict) -> tuple[str, dict]:
    """构建已选技能的工具规划上下文；参数为选择结果及原目录；返回文本和元信息，不再次选择。"""
    return build_file_skill_context(skill, registry, locations)


def load_required_tool_skill(skill_id: str, registry: SkillRegistry, locations: dict,
                             tool_names: set[str]) -> SkillDefinition | None:
    """加载目录声明的技能依赖；参数为技能编号、可信注册表及位置、可用工具名；合法返回定义，否则返回None，不授予执行权限。"""
    skill = registry.get(skill_id)
    location = locations.get(skill_id)
    if skill is None or not skill.enabled or location is None:
        return None
    targets = location.get("target_agents") or []
    if targets and "tool_agent" not in targets:
        return None
    if not set(skill.allowed_tools).issubset(tool_names):
        return None
    return skill
