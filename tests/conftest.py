"""
pytest 公共测试配置文件。

本文件用于放置测试过程中会复用的 fixture（测试夹具）。
fixture（测试夹具）：pytest 中用于准备测试依赖、测试数据、测试环境的函数。
"""

import pytest


@pytest.fixture
def sample_user_id() -> str:
    """
    提供测试用 user_id。

    参数：
        无。

    返回值：
        str：测试用户 ID，格式为普通字符串。
    """
    return "test_user"


@pytest.fixture
def sample_session_id() -> str:
    """
    提供测试用 session_id。

    参数：
        无。

    返回值：
        str：测试会话 ID，格式为普通字符串。
    """
    return "test_session"


@pytest.fixture
def sample_question() -> str:
    """
    提供测试用用户问题。

    参数：
        无。

    返回值：
        str：测试问题文本。
    """
    return "金毛适合新手养吗？"