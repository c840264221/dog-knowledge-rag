"""
runtime hook manager import 回归测试。

regression test（回归测试）：
用于防止已经修复过的问题在未来再次出现。

circular import（循环导入）：
两个或多个模块互相 import，导致 Python 在模块还没初始化完成时访问对象，从而报错。

本测试用于防止 RuntimeHookManager 再次因为顶层 import logger 等模块，
触发 hook_manager -> logger -> runtime_context -> hook_manager 的循环导入。
"""


def test_runtime_hook_manager_can_be_imported_directly():
    """
    测试 RuntimeHookManager 是否可以被直接导入。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.hooks.hook_manager import RuntimeHookManager

    assert RuntimeHookManager is not None


def test_runtime_hook_manager_module_can_be_imported_without_circular_import():
    """
    测试 hook_manager 模块是否可以被正常导入，且不会触发 circular import。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    import src.runtime.hooks.hook_manager as hook_manager_module

    assert hook_manager_module is not None
    assert hasattr(
        hook_manager_module,
        "RuntimeHookManager",
    )


def test_runtime_hook_manager_can_create_instance_after_import():
    """
    测试导入 RuntimeHookManager 后是否可以正常创建实例。

    参数：
        无。

    返回值：
        None：pytest 会根据 assert 判断测试是否通过。
    """

    from src.runtime.hooks.hook_manager import RuntimeHookManager

    hook_manager = RuntimeHookManager()

    assert hook_manager is not None
    assert hook_manager._hooks is not None