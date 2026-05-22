import sqlite3
from src.config import MEMORY_DB_PATH, BASE_DIR
import os
import atexit


_memory_store = None

def get_memory_store():

    global _memory_store

    if _memory_store is None:

        _memory_store = SQLiteMemoryStore()

        # 注册程序退出关闭
        atexit.register(
            _memory_store.close
        )

    return _memory_store


class SQLiteMemoryStore:

    def __init__(self, db_path=MEMORY_DB_PATH):
        os.makedirs(
            os.path.dirname(db_path),
            exist_ok=True
        )

        self.conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False
        )

        self.create_tables()

    def create_tables(self):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id TEXT,

            memory_type TEXT,

            content TEXT,

            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()

    def add_memory(
            self,
            user_id: str,
            memory_type: str,
            content: str,
            confidence: float
    ):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT 1
            FROM user_memory
            WHERE user_id = ?
            AND memory_type = ?
            AND content = ?
            """,
            (
                user_id,
                memory_type,
                content
            )
        )

        exists = cursor.fetchone()

        if exists:
            return

        cursor.execute(
            """
            INSERT INTO user_memory
            (
                user_id,
                memory_type,
                content,
                confidence
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                memory_type,
                content,
                confidence
            )
        )

        self.conn.commit()

    def get_memories(self, user_id: str, memory_type: str | None = None):

        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT memory_type, content
            FROM user_memory
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

        return rows

    def close(self):
        if self.conn:
            self.conn.close()