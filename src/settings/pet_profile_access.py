"""宠物档案字段访问业务策略配置。"""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from src.memory.memory_schema import PetProfileAttribute


class AgentPetProfileAccessPolicy(BaseModel):
    """
    定义一个 Agent 可以读取和处理哪些宠物档案字段。

    功能：
        把数据库读取权限与业务处理权限分开配置。读取权限决定系统能否主动
        从数据库取得字段；处理权限决定该字段能否进入 Agent 上下文。

    参数含义：
        database_read_attributes:
            允许该 Agent 主动从宠物档案数据库读取的字段。
        processing_attributes:
            允许该 Agent 在业务执行和 Prompt 中使用的字段。

    返回值含义：
        AgentPetProfileAccessPolicy:
            经过 Pydantic 校验的单 Agent 字段访问策略。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    database_read_attributes: frozenset[PetProfileAttribute] = Field(
        default_factory=frozenset,
    )
    processing_attributes: frozenset[PetProfileAttribute] = Field(
        default_factory=frozenset,
    )


class PetProfileAccessSettings(BaseModel):
    """
    集中保存所有 Agent 的宠物档案字段访问策略。

    功能：
        作为随代码版本管理的业务配置，为字段访问决策提供统一策略来源。
        未登记的 Agent 默认没有读取和处理权限，遵循 fail closed（默认拒绝）
        原则。本配置不从环境变量读取，避免把业务权限散落在部署参数中。

    参数含义：
        agent_policies:
            Agent 名称到字段访问策略的映射。

    返回值含义：
        PetProfileAccessSettings:
            可挂载到项目统一 settings 的宠物档案访问配置。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    agent_policies: dict[str, AgentPetProfileAccessPolicy] = Field(
        default_factory=lambda: {
            "dog_knowledge_agent": AgentPetProfileAccessPolicy(
                database_read_attributes=frozenset(
                    get_args(PetProfileAttribute)
                ),
                processing_attributes=frozenset(
                    get_args(PetProfileAttribute)
                ),
            ),
        }
    )

    def get_agent_policy(
        self,
        agent_name: str,
    ) -> AgentPetProfileAccessPolicy:
        """
        获取指定 Agent 的字段访问策略。

        功能：
            根据标准化 Agent 名称查找配置。Agent 未登记时返回空权限策略，
            不会因为配置遗漏而意外获得全部档案权限。

        参数含义：
            agent_name:
                申请访问宠物档案的 Agent 名称。

        返回值含义：
            AgentPetProfileAccessPolicy:
                已登记策略或默认的空权限策略。
        """

        normalized_agent_name = str(agent_name or "").strip()
        return self.agent_policies.get(
            normalized_agent_name,
            AgentPetProfileAccessPolicy(),
        )
