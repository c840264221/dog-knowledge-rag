from enum import Enum


class CheckpointStatus(str, Enum):

    # 等待恢复
    ACTIVE = "active"

    # 已恢复
    RESUMED = "resumed"

    # 已经结束
    COMPLETED = "completed"

    # 超时
    EXPIRED = "expired"

    # 用户取消
    CANCELLED = "cancelled"