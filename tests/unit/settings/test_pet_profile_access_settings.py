"""宠物档案字段访问业务配置测试。"""

from src.settings.pet_profile_access import PetProfileAccessSettings


def test_unknown_agent_should_receive_empty_permissions() -> None:
    """未登记 Agent 必须默认获得空权限，不能自动继承全部字段。"""

    settings = PetProfileAccessSettings()

    policy = settings.get_agent_policy("unknown_agent")

    assert policy.database_read_attributes == frozenset()
    assert policy.processing_attributes == frozenset()


def test_default_dog_agent_policy_should_be_business_configuration() -> None:
    """默认狗知识 Agent 权限应由业务配置集中提供。"""

    settings = PetProfileAccessSettings()

    policy = settings.get_agent_policy("dog_knowledge_agent")

    assert "breed" in policy.database_read_attributes
    assert "training_goal" in policy.processing_attributes
