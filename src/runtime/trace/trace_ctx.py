import contextvars
from typing import Optional

# 定义上下文变量
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('trace_id', default=None)
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('session_id', default=None)
component_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('component', default=None)

def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)

def get_trace_id() -> Optional[str]:
    return trace_id_var.get()

def set_user_id(user_id: str) -> None:
    user_id_var.set(user_id)

def get_user_id() -> Optional[str]:
    return user_id_var.get()

def set_session_id(session_id: str) -> None:
    session_id_var.set(session_id)

def get_session_id() -> Optional[str]:
    return session_id_var.get()

def set_component(component: str) -> None:
    component_var.set(component)

def get_component() -> Optional[str]:
    return component_var.get()

def clear_context() -> None:
    """重置所有上下文变量（在请求结束时可选调用）"""
    trace_id_var.set(None)
    user_id_var.set(None)
    session_id_var.set(None)
    component_var.set(None)