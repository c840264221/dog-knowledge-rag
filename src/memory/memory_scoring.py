from datetime import datetime
from math import exp

from src.logger import logger


class MemoryScorer:
    """
    MemoryScorer（记忆评分器）

    负责计算记忆最终召回分数。

    score = strength * decay

    strength:
        记忆强度，越高说明用户越频繁提到。

    decay:
        时间衰减，越久没出现，分数越低。
    """

    def __init__(
            self,
            decay_rate: float = 0.05
    ):
        self.decay_rate = decay_rate

    def score(
            self,
            strength: float,
            last_seen: str | None
    ) -> float:

        if not last_seen:
            return strength

        try:
            last_seen_dt = datetime.fromisoformat(
                last_seen
            )
        except ValueError:
            logger.error("last_seen_dt 转换成时间格式失败")
            return strength

        days = (
            datetime.now() - last_seen_dt
        ).days

        decay = exp(
            -self.decay_rate * days
        )
        logger.info(f"days={days}, decay={decay}")

        return strength * decay