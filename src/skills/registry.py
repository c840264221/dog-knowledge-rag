"""Skill（技能）注册表。"""

from __future__ import annotations

from collections.abc import Iterable

from src.skills.schemas import SkillCatalogItem, SkillDefinition


class SkillNotFoundError(LookupError):
    """表示注册表中不存在请求的技能。"""


class DuplicateSkillError(ValueError):
    """表示同一个技能编号被重复注册。"""


class SkillRegistry:
    """
    保存并查询项目可用的 Skill 定义。

    功能：
        使用 skill_id 作为唯一键管理技能，提供注册、精确查询和稳定排序列表。

    参数含义：
        skills:
            可选的初始技能集合；未提供时创建空注册表。

    返回值含义：
        SkillRegistry:
            独立的技能注册表实例。
    """

    def __init__(
        self,
        skills: Iterable[SkillDefinition] | None = None,
    ) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills or []:
            self.register(skill)

    def register(self, skill: SkillDefinition) -> None:
        """
        注册一个技能定义。

        功能：
            按 skill_id 保存技能；如果编号已经存在则明确报错，避免后注册的
            技能悄悄覆盖旧定义。

        参数含义：
            skill:
                已经过 Schema 校验的技能定义。

        返回值含义：
            None:
                注册成功后不返回额外数据。
        """

        if skill.skill_id in self._skills:
            raise DuplicateSkillError(
                f"技能已经注册: {skill.skill_id}"
            )
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> SkillDefinition | None:
        """
        根据编号查询技能。

        功能：
            在允许“技能可能不存在”的调用场景中执行安全查询。

        参数含义：
            skill_id:
                需要查询的技能编号。

        返回值含义：
            SkillDefinition | None:
                找到时返回技能定义，找不到时返回 None。
        """

        return self._skills.get(skill_id)

    def require(self, skill_id: str) -> SkillDefinition:
        """
        根据编号读取必须存在的技能。

        功能：
            为加载和执行阶段提供严格查询；技能不存在时抛出明确异常。

        参数含义：
            skill_id:
                必须存在的技能编号。

        返回值含义：
            SkillDefinition:
                对应的技能定义。
        """

        skill = self.get(skill_id)
        if skill is None:
            raise SkillNotFoundError(f"技能不存在: {skill_id}")
        return skill

    def list_enabled(self) -> list[SkillDefinition]:
        """
        列出全部启用技能。

        功能：
            过滤 enabled=False 的技能，并按 skill_id 排序，保证 Prompt、日志
            和测试输出稳定。

        参数含义：
            无。

        返回值含义：
            list[SkillDefinition]:
                按技能编号升序排列的启用技能列表。
        """

        return sorted(
            (
                skill
                for skill in self._skills.values()
                if skill.enabled
            ),
            key=lambda skill: skill.skill_id,
        )

    def list_catalog(self) -> list[SkillCatalogItem]:
        """
        构建启用技能的精简目录。

        功能：
            从每个启用技能中只提取发现阶段需要的元数据，不返回完整执行步骤，
            为 Skill Selector 和未来 LLM Router 提供低噪音候选目录。

        参数含义：
            无。

        返回值含义：
            list[SkillCatalogItem]:
                按技能编号排序的精简目录项列表。
        """

        return [
            SkillCatalogItem.from_definition(skill)
            for skill in self.list_enabled()
        ]
