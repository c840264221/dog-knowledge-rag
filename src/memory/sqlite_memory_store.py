import sqlite3
from src.config import MEMORY_DB_PATH, BASE_DIR
import os
from datetime import datetime



class SQLiteMemoryStore:
    """
       SQLiteMemoryStore 是 Memory 模块的 SQLite 存储服务。

       功能：
       - 管理 user_memory 表
       - 提供新增、查询、更新、失效、关闭数据库连接等能力
       - 作为 Memory 的主数据库使用

       技术名词：
       - SQLite：轻量级本地数据库，适合存储小型结构化数据
       - Store：存储层，负责数据读写
       - Connection：数据库连接，用于执行 SQL 语句
   """

    def __init__(self, db_path=MEMORY_DB_PATH):
        """
           初始化 SQLiteMemoryStore。

           功能：
           - 创建数据库目录
           - 建立 SQLite 连接
           - 设置 row_factory，让查询结果可以转成 dict
           - 自动创建 user_memory 表

           参数：
           - db_path:
             SQLite 数据库文件路径。

           返回值：
           - None
             初始化函数不返回业务数据。
       """
        os.makedirs(
            os.path.dirname(db_path),
            exist_ok=True
        )

        self.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False
        )

        self.conn.row_factory = sqlite3.Row

        self.create_tables()

    def create_tables(self):
        """
            创建 Memory 数据表。

            功能：
            - 如果 user_memory 表不存在，则自动创建
            - 保证 user_id、memory_type、content 组合唯一
            - 避免同一用户保存重复记忆

            参数：
            - 无

            返回值：
            - None
                只执行建表操作。
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id TEXT NOT NULL,

                memory_type TEXT NOT NULL,

                content TEXT NOT NULL,

                confidence REAL DEFAULT 0,

                strength REAL DEFAULT 1,

                status TEXT DEFAULT 'active',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(user_id, memory_type, content)
            )
            """
        )

        self.conn.commit()

    def add_memory(
            self,
            user_id: str,
            memory_type: str,
            content: str,
            confidence: float,
            strength: float = 1.0,
            status: str = "active"
    ) -> int | None:
        """
        新增一条用户记忆。

        功能：
        - 将用户记忆写入 SQLite 的 user_memory 表
        - 如果相同 user_id、memory_type、content 已存在，则不会重复插入
        - 返回新建或已存在的 memory_id

        参数：
        - user_id: str
          用户 ID，用于区分不同用户的记忆

        - memory_type: str
          记忆类型，例如 favorite_dog、preference、dislike

        - content: str
          记忆内容，例如“用户喜欢金毛”

        - confidence: float
          LLM 判断这条记忆可信度的分数

        - strength: float
          记忆强度，越高表示用户越频繁提到

        - status: str
          记忆状态，例如 active、inactive

        返回值：
        - int | None
          返回记忆 ID；如果写入失败则返回 None
        """

        cursor = self.conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR IGNORE INTO user_memory
            (
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                now,
                now
            )
        )

        self.conn.commit()

        existing = self.find_memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content
        )

        if not existing:
            return None

        memory_id = existing.get("id")

        if memory_id is None:
            return None

        return int(memory_id)

    def get_memories(
            self,
            user_id: str,
            memory_type: str | None = None,
            limit: int = 20
    ):

        cursor = self.conn.cursor()

        if memory_type:

            cursor.execute(
                """
                SELECT
                    memory_type,
                    content,
                    confidence,
                    strength,
                    last_seen
                FROM user_memory
                WHERE user_id = ?
                AND memory_type = ?
                AND status = 'active'
                ORDER BY strength DESC, last_seen DESC
                LIMIT ?
                """,
                (
                    user_id,
                    memory_type,
                    limit
                )
            )

        else:

            cursor.execute(
                """
                SELECT
                    memory_type,
                    content,
                    confidence,
                    strength,
                    last_seen
                FROM user_memory
                WHERE user_id = ?
                ORDER BY strength DESC, last_seen DESC
                LIMIT ?
                """,
                (
                    user_id,
                    limit
                )
            )

        return cursor.fetchall()

    def find_memory(
            self,
            user_id: str,
            memory_type: str,
            content: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            FROM user_memory
            WHERE user_id = ?
            AND memory_type = ?
            AND content = ?
            AND status = 'active'
            LIMIT 1
            """,
            (
                user_id,
                memory_type,
                content
            )
        )

        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def update_memory(
            self,
            memory_id: int,
            confidence: float | None = None,
            strength: float | None = None,
            last_seen: str | None = None,
            status: str | None = None
    ):

        fields = []

        values = []

        if confidence is not None:
            fields.append("confidence = ?")
            values.append(confidence)

        if strength is not None:
            fields.append("strength = ?")
            values.append(strength)

        if last_seen is not None:
            fields.append("last_seen = ?")
            values.append(last_seen)

        if status is not None:
            fields.append("status = ?")
            values.append(status)

        if not fields:
            return

        values.append(memory_id)

        sql = f"""
            UPDATE user_memory
            SET {", ".join(fields)}
            WHERE id = ?
        """

        cursor = self.conn.cursor()

        cursor.execute(
            sql,
            values
        )

        self.conn.commit()

    def get_all_memories(
            self,
            user_id: str
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            FROM user_memory
            WHERE user_id = ?
            AND status = 'active'
            """,
            (
                user_id,
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # 按strength和last_seen粗筛
    def get_candidate_memories(
            self,
            user_id: str,
            limit: int = 50
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            FROM user_memory
            WHERE user_id = ?
            AND status = 'active'
            ORDER BY strength DESC, last_seen DESC
            LIMIT ?
            """,
            (
                user_id,
                limit
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # 找到冲突的记忆
    def find_conflict_memories(
            self,
            user_id: str,
            conflict_types: list[str],
            content: str
    ):

        if not conflict_types:
            return []

        placeholders = ",".join(
            "?"
            for _ in conflict_types
        )

        sql = f"""
            SELECT
                id,
                user_id,
                memory_type,
                content,
                status
            FROM user_memory
            WHERE user_id = ?
            AND memory_type IN ({placeholders})
            AND content = ?
            AND status = 'active'
        """

        cursor = self.conn.cursor()

        cursor.execute(
            sql,
            [
                user_id,
                *conflict_types,
                content
            ]
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # 失效旧记忆
    def deactivate_memory(
            self,
            memory_id: int
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE user_memory
            SET status = 'inactive'
            WHERE id = ?
            """,
            (
                memory_id,
            )
        )

        self.conn.commit()

    def get_memory_by_id(
            self,
            memory_id: int
    ):
        """
        根据 memory_id 查询单条记忆。

        功能：
        - 从 SQLite 的 user_memory 表中查询指定 ID 的记忆
        - 返回完整字段，供后续同步到 Chroma 使用
        - 主要用于 save_memory / update_memory 之后获取最新数据

        参数：
        - memory_id: int
          SQLite 中 user_memory 表的主键 ID

        返回值：
        - dict | None
          如果找到记忆，返回 dict 格式的记忆数据；
          如果没有找到，返回 None
        """

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            FROM user_memory
            WHERE id = ?
            LIMIT 1
            """,
            (
                memory_id,
            )
        )

        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def get_memories_by_ids(
            self,
            memory_ids: list[int],
            only_active: bool = True
    ) -> list[dict]:
        """
        根据 memory_id 列表批量查询记忆。

        功能：
        - 从 SQLite 的 user_memory 表中批量查询多条记忆
        - 用于 Chroma 语义检索后，根据 memory_id 回查主数据库
        - 默认只返回 active 状态的记忆，避免召回失效记忆

        参数：
        - memory_ids: list[int]
          需要查询的记忆 ID 列表

        - only_active: bool
          是否只查询 active 状态的记忆。
          True 表示只返回有效记忆。
          False 表示不限制 status。

        返回值：
        - list[dict]
          返回多条记忆数据，每条记忆是 dict 格式。
          如果 memory_ids 为空，则返回空列表。
        """

        if not memory_ids:
            return []

        placeholders = ",".join(
            "?"
            for _ in memory_ids
        )

        sql = f"""
            SELECT
                id,
                user_id,
                memory_type,
                content,
                confidence,
                strength,
                status,
                created_at,
                last_seen
            FROM user_memory
            WHERE id IN ({placeholders})
        """

        values = list(
            memory_ids
        )

        if only_active:
            sql += """
                AND status = 'active'
            """

        cursor = self.conn.cursor()

        cursor.execute(
            sql,
            values
        )

        rows = cursor.fetchall()

        memories = [
            dict(row)
            for row in rows
        ]

        memory_order = {
            memory_id: index
            for index, memory_id in enumerate(memory_ids)
        }

        memories.sort(
            key=lambda memory: memory_order.get(
                int(memory["id"]),
                999999
            )
        )

        return memories

    def close(self):
        if self.conn:
            self.conn.close()