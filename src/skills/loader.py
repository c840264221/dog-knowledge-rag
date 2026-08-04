"""Skill（技能）加载与上下文渲染。"""

from __future__ import annotations

from src.skills.registry import SkillRegistry
from src.skills.schemas import SkillDefinition


class DisabledSkillError(ValueError):
    """表示请求加载的技能当前处于停用状态。"""


class SkillLoader:
    """
    从注册表加载技能并构建 Agent 可使用的上下文。

    功能：
        隔离“保存技能”和“使用技能”两个职责。注册表只管理定义，加载器负责
        检查启用状态，并把结构化定义渲染成确定性的文本上下文。

    参数含义：
        registry:
            保存项目技能定义的 SkillRegistry。

    返回值含义：
        SkillLoader:
            可以加载和渲染指定技能的对象。
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def load(self, skill_id: str) -> SkillDefinition:
        """
        加载一个启用的技能定义。

        功能：
            先要求注册表中存在该技能，再检查 enabled 状态，防止停用技能
            被误注入 Agent。

        参数含义：
            skill_id:
                需要加载的技能编号。

        返回值含义：
            SkillDefinition:
                存在且已启用的技能定义。
        """

        skill = self.registry.require(skill_id)
        if not skill.enabled:
            raise DisabledSkillError(f"技能已停用: {skill_id}")
        return skill

    def render_context(self, skill_id: str) -> str:
        """
        将技能定义渲染成可注入 Agent Prompt 的文本。

        功能：
            把结构化字段按照稳定顺序组织成人类和模型都容易阅读的上下文。
            本方法只准备说明，不调用 LLM，也不执行工具。

        参数含义：
            skill_id:
                需要渲染的技能编号。

        返回值含义：
            str:
                包含技能身份、输入、步骤、工具、输出和边界的文本。
        """

        skill = self.load(skill_id)
        sections = [
            f"技能：{skill.name}（{skill.skill_id}@{skill.version}）",
            f"职责：{skill.description}",
            self._render_list(
                "必需输入",
                [
                    f"{item.name}（{item.input_id}）"
                    for item in skill.required_inputs
                ],
            ),
            self._render_numbered_list("执行步骤", skill.instructions),
            self._render_list("允许工具", skill.allowed_tools),
            f"输出要求：{skill.output_contract}",
            self._render_list("执行边界", skill.guardrails),
        ]
        return "\n".join(sections)

    def render_catalog(self) -> str:
        """
        渲染启用技能的精简目录。

        功能：
            只输出技能编号、名称、职责和触发提示，不加载完整执行步骤。
            未来需要 LLM 选择技能时，可以只把这份短目录放入上下文。

        参数含义：
            无。

        返回值含义：
            str:
                每个技能占一行的精简目录；没有启用技能时返回“无可用技能”。
        """

        catalog = self.registry.list_catalog()
        if not catalog:
            return "无可用技能"

        return "\n".join(
            (
                f"- {item.skill_id}@{item.version} | {item.name} | "
                f"{item.description} | 触发提示："
                f"{'、'.join(item.activation_hints) if item.activation_hints else '无'}"
            )
            for item in catalog
        )

    @staticmethod
    def _render_list(title: str, values: list[str]) -> str:
        """
        渲染普通列表字段。

        功能：
            将多个值连接为一行；空列表使用“无”明确表示没有声明内容。

        参数含义：
            title:
                当前列表的中文标题。
            values:
                需要渲染的字符串列表。

        返回值含义：
            str:
                单行列表文本。
        """

        return f"{title}：{'、'.join(values) if values else '无'}"

    @staticmethod
    def _render_numbered_list(title: str, values: list[str]) -> str:
        """
        渲染带顺序编号的步骤列表。

        功能：
            保留技能步骤的执行顺序，让 Agent 明确先做什么、后做什么。

        参数含义：
            title:
                步骤部分的中文标题。
            values:
                已按执行顺序排列的步骤列表。

        返回值含义：
            str:
                标题和逐行编号步骤组成的文本。
        """

        rendered_steps = "\n".join(
            f"{index}. {value}"
            for index, value in enumerate(values, start=1)
        )
        return f"{title}：\n{rendered_steps}"
