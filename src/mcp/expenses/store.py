"""本机教学支出存储；记录本身充当持久化回执，不提供任意SQL或数据库路径工具。"""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from src.mcp.expenses.schemas import (
    ExpenseBatchInput, ExpenseBatchWriteResult,
    ExpenseInput, ExpenseQueryInput, ExpenseQueryResult, ExpenseRecord, ExpenseWriteResult,
)


class ExpenseStore:
    """按服务端绑定的用户写入SQLite；幂等键和记录在同一事务中保存。"""

    def __init__(self, database_path: Path, *, user_id: str):
        """初始化表；database_path为服务端配置的路径，user_id为绑定身份；无返回值。

        本模块不负责认证；教学服务使用固定身份，生产必须由可信认证上下文提供。
        用户身份不是MCP工具参数，客户端不能通过正文更换记账归属。
        """
        if not user_id.strip():
            raise ValueError("绑定的用户身份不能为空。")
        self.database_path = Path(database_path).resolve()
        self.user_id = user_id.strip()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path, timeout=5)) as connection:
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS expenses (
                        expense_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        UNIQUE(user_id, idempotency_key)
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS expense_batches (
                        batch_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        request_fingerprint TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        UNIQUE(user_id, idempotency_key)
                    )
                """)

    def record(self, request: ExpenseInput) -> ExpenseWriteResult:
        """提交一笔支出；request为通过校验的输入；返回新增或已有记录。

        同用户同键同内容返回原记录；同键不同内容抛出异常，不覆盖也不新增。
        账单日期包含在内容指纹中，不用系统创建时间代替；旧记录不自动补写日期或更改指纹。
        BEGIN IMMEDIATE先取得写事务资格，再检查和插入，避免并发请求检查后重复写入。
        SQLite保存失败会回滚；只有事务提交完成后才返回成功，不自动重试未知提交。
        """
        payload = request.model_dump(exclude={"idempotency_key"})
        fingerprint = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        with closing(sqlite3.connect(self.database_path, timeout=5)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT request_fingerprint, record_json FROM expenses WHERE user_id=? AND idempotency_key=?",
                    (self.user_id, request.idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing[0] != fingerprint:
                        raise ValueError("幂等键冲突：该键已用于不同记账内容，拒绝修改原记录。")
                    result = ExpenseWriteResult(
                        status="already_recorded", record=ExpenseRecord.model_validate_json(existing[1]),
                    )
                else:
                    record = ExpenseRecord(
                        expense_id="expense_" + uuid4().hex,
                        user_id=self.user_id,
                        **request.model_dump(),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    connection.execute(
                        "INSERT INTO expenses (expense_id,user_id,idempotency_key,request_fingerprint,record_json) VALUES (?,?,?,?,?)",
                        (record.expense_id, self.user_id, request.idempotency_key, fingerprint, record.model_dump_json()),
                    )
                    result = ExpenseWriteResult(status="recorded", record=record)
        return result

    def record_batch(self, request: ExpenseBatchInput) -> ExpenseBatchWriteResult:
        """在一个事务中写入整批账单及批次回执，不循环提交单笔事务。

        参数：request为逐笔校验后的完整批次；身份取服务端绑定用户。
        返回值：新增批次或同键同内容的原批次回执；任意写入失败整批回滚。
            指纹包含明细顺序；同键改内容或顺序报冲突。单项键由批次键和序号派生，
            只用于批内记录唯一性，不用于识别不同批次中的现实重复消费。
        """
        payload = request.model_dump(exclude={"idempotency_key"})
        fingerprint = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        with closing(sqlite3.connect(self.database_path, timeout=5)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT request_fingerprint, result_json FROM expense_batches WHERE user_id=? AND idempotency_key=?",
                    (self.user_id, request.idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing[0] != fingerprint:
                        raise ValueError("批次幂等键冲突：明细内容或顺序已改变，拒绝重复提交。")
                    return ExpenseBatchWriteResult.model_validate_json(existing[1]).model_copy(
                        update={"status": "already_recorded"},
                    )
                records = []
                created_at = datetime.now(timezone.utc).isoformat()
                for index, item in enumerate(request.items):
                    item_key = "batch-item:" + hashlib.sha256(
                        f"{request.idempotency_key}:{index}".encode("utf-8"),
                    ).hexdigest()
                    record = ExpenseRecord(
                        expense_id="expense_" + uuid4().hex, user_id=self.user_id,
                        **item.model_dump(), idempotency_key=item_key, created_at=created_at,
                    )
                    item_fingerprint = hashlib.sha256(json.dumps(
                        item.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                    ).encode("utf-8")).hexdigest()
                    connection.execute(
                        "INSERT INTO expenses (expense_id,user_id,idempotency_key,request_fingerprint,record_json) VALUES (?,?,?,?,?)",
                        (record.expense_id, self.user_id, item_key, item_fingerprint, record.model_dump_json()),
                    )
                    records.append(record)
                result = ExpenseBatchWriteResult(
                    status="recorded", batch_id="expense_batch_" + uuid4().hex,
                    idempotency_key=request.idempotency_key,
                    record_count=len(records), records=records,
                )
                connection.execute(
                    "INSERT INTO expense_batches (batch_id,user_id,idempotency_key,request_fingerprint,result_json) VALUES (?,?,?,?,?)",
                    (result.batch_id, self.user_id, request.idempotency_key, fingerprint, result.model_dump_json()),
                )
        return result

    def query(self, request: ExpenseQueryInput) -> ExpenseQueryResult:
        """按绑定用户及分类查询最近支出，保留原表结构，不修改任何记录。

        参数：request为校验后的分类与条数上限；用户身份始终取self.user_id。
        返回值：按新增顺序从近到远排列的记录；空结果为not_found，不视为查询失败。
            多读一条判断has_more，不返回全量金额统计，也不表示查询了全部历史。
            当前分类位于JSON正文中，通过SQLite JSON函数过滤；大规模场景应另评估索引字段。
        """
        with closing(sqlite3.connect(self.database_path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT record_json FROM expenses "
                "WHERE user_id=? AND json_extract(record_json, '$.category')=? "
                "ORDER BY rowid DESC LIMIT ?",
                (self.user_id, request.category, request.limit + 1),
            ).fetchall()
        records = [ExpenseRecord.model_validate_json(row[0]) for row in rows[:request.limit]]
        return ExpenseQueryResult(
            status="found" if records else "not_found",
            category=request.category,
            records=records,
            returned_count=len(records),
            limit=request.limit,
            has_more=len(rows) > request.limit,
        )
