"""宿主生成的请求时间快照；不从用户正文或模型输出获取可信时间。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class RuntimeTimeContext:
    """保存不可变时间快照；observed_at为带时区采样时间、timezone_name为业务时区；实例供同一请求共享。"""

    observed_at: datetime
    timezone_name: str = "Asia/Hong_Kong"

    def __post_init__(self) -> None:
        """检查快照参数；无额外参数、无返回值；拒绝无时区时间及非法时区名称。"""
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("运行时间基准必须带时区。")
        ZoneInfo(self.timezone_name)

    @classmethod
    def capture(cls, *, now: datetime | None = None, timezone_name: str = "Asia/Hong_Kong") -> "RuntimeTimeContext":
        """采样宿主时间；now为可信入口可注入的带时区时间、timezone_name为业务时区；返回不可变快照。"""
        return cls(observed_at=now if now is not None else datetime.now(timezone.utc), timezone_name=timezone_name)

    def as_dict(self) -> dict[str, str]:
        """转换为模型可用的环境数据；无参数；返回参考时间、日期和时区的独立字典。"""
        local_time = self.observed_at.astimezone(ZoneInfo(self.timezone_name))
        return {"reference_time": local_time.isoformat(), "reference_date": local_time.date().isoformat(),
                "timezone": self.timezone_name}
