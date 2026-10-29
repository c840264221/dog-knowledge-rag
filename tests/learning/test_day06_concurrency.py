import asyncio
from pydantic import BaseModel

from datetime import datetime
import pytest

from datetime import datetime, timedelta, timezone


class InMemoryClaimStore:
    def __init__(self):
        """
        初始化内存领取记录。

        参数：
            无。

        返回：
            None。
            初始化任务领取字典和异步锁。
        """
        self.claims: dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def try_claim(
        self,
        task_id: str,
        claim_id: str,
    ) -> bool:
        """
        尝试为指定任务保存领取令牌。

        参数：
            task_id:
                准备领取的任务编号。
            claim_id:
                当前 Worker 为本次领取生成的令牌。

        返回：
            True：
                当前任务原来没有被领取，本次领取成功。
            False：
                当前任务已经存在领取记录，本次领取失败。
        """

        async with self.lock:
            if task_id in self.claims:
                return False
            await asyncio.sleep(0)
            self.claims[task_id] = claim_id
            return True


@pytest.mark.asyncio
async def test_两个Worker同时领取同一任务时只有一个成功():
    """
    验证两个 Worker 同时领取同一个任务时，
    只有一个 Worker 可以成功保存领取令牌。

    参数：
        无。

    返回：
        None。
        通过断言验证一个领取结果为 True、另一个为 False，
        并且存储中只保留一个领取令牌。
    """
    claim_store = InMemoryClaimStore()
    claim_results = await asyncio.gather(
        claim_store.try_claim(task_id="task-1", claim_id="claim-A"),
        claim_store.try_claim(task_id="task-1", claim_id="claim-B"),
    )
    assert claim_results.count(True) == 1
    assert claim_results.count(False) == 1
    assert len(claim_store.claims) == 1
    assert claim_store.claims["task-1"] in {"claim-A", "claim-B"}



class ClaimLease(BaseModel):
    task_id: str
    claim_id: str
    worker_id: str
    acquired_at: datetime
    expires_at: datetime


def is_claim_lease_expired(
    claim_lease: ClaimLease,
    current_time: datetime,
) -> bool:
    """
    判断任务领取租约在指定时间是否已经过期。

    参数：
        claim_lease:
            Worker 当前持有的任务领取租约，
            包含领取时间和租约到期时间。
        current_time:
            准备用来判断租约状态的当前时间。

    返回：
        True：
            当前时间已经到达或超过租约到期时间。
        False：
            当前时间仍然早于租约到期时间。
    """
    if current_time < claim_lease.expires_at:
        return False
    if current_time == claim_lease.expires_at:
        return True
    if current_time > claim_lease.expires_at:
        return True
    return False

def can_commit_claim_result(
    current_claim_lease: ClaimLease | None,
    submitted_claim_id: str,
    current_time: datetime,
) -> bool:
    """
    判断 Worker 提交的执行结果是否仍有资格保存。

    参数：
        current_claim_lease:
            当前任务在权威存储中的有效领取租约。
            如果任务当前没有租约，则为 None。
        submitted_claim_id:
            Worker 提交执行结果时携带的领取令牌。
        current_time:
            接收执行结果时的当前时间。

    返回：
        True：
            提交令牌与当前租约一致，并且租约尚未过期，
            可以保存本次执行结果。
        False：
            当前没有租约、领取令牌不一致或租约已经过期，
            不能保存本次执行结果。
    """
    if current_claim_lease is None:
        return False

    if submitted_claim_id != current_claim_lease.claim_id:
        return False

    lease_expired = is_claim_lease_expired(
        current_claim_lease,
        current_time,
    )
    if lease_expired:
        return False

    return True

@pytest.mark.parametrize(
    "has_current_lease, submitted_claim_id, current_time, expected",
    [
        (True,"CLAIM-A",datetime(2026, 9, 8, 10, 5, 0,tzinfo=timezone.utc),False),
        (True,"CLAIM-B",datetime(2026, 9, 8, 10, 11, 0,tzinfo=timezone.utc),False),
        (True,"CLAIM-B",datetime(2026, 9, 8, 10, 10, 0,tzinfo=timezone.utc),False),
        (True,"CLAIM-B",datetime(2026, 9, 8, 10, 5, 0,tzinfo=timezone.utc),True),
        (False,"CLAIM-B",datetime(2026, 9, 8, 10, 5, 0,tzinfo=timezone.utc),False),
    ]
)
def test_只有持有当前有效租约的Worker才能提交结果(has_current_lease, submitted_claim_id, current_time, expected):
    """
    验证执行结果只有在领取令牌一致且租约未过期时才能保存。

    参数：

        has_current_lease:
            当前场景是否存在任务领取租约。
        submitted_claim_id:
            Worker提交结果时携带的领取令牌。
        current_time:
            模拟接收执行结果的时间。
        expected:
            当前场景预期是否允许保存结果。

    返回：
        None。
        通过断言验证无租约、旧令牌和过期租约都会被拒绝，
        只有持有当前有效租约的 Worker 可以提交结果。
    """
    acquired_at = datetime(
        2026, 9, 8, 10, 0, 0,
        tzinfo=timezone.utc,
    )

    expires_at = acquired_at + timedelta(minutes=10)
    if has_current_lease:
        current_claim_lease = ClaimLease(
            task_id="TASK-1",
            claim_id="CLAIM-B",
            worker_id="WORKER-B",
            acquired_at=acquired_at,
            expires_at=expires_at,
        )
    else:
        current_claim_lease = None
    result = can_commit_claim_result(
        current_claim_lease=current_claim_lease,
        submitted_claim_id=submitted_claim_id,
        current_time=current_time,
    )
    assert result == expected