"""MemoryProvider（记忆服务提供者）的宠物档案服务管理测试。"""

from src.memory.pet_profile_service import PetProfileService
from src.runtime.container.providers.memory_provider import MemoryProvider


class FakeMemoryStore:
    """用于验证依赖复用的 SQLite 存储测试替身。"""


def test_memory_provider_should_cache_pet_profile_service() -> None:
    """
    验证 MemoryProvider 会创建并复用同一个宠物档案服务。

    参数含义：
        无。

    返回值含义：
        None，断言失败时由 pytest 报错。
    """

    provider = MemoryProvider(vectorstore_provider=object())
    memory_store = FakeMemoryStore()
    provider._store = memory_store

    first_service = provider.pet_profile_service
    second_service = provider.pet_profile_service

    assert isinstance(first_service, PetProfileService)
    assert first_service is second_service
    assert first_service.store is memory_store
