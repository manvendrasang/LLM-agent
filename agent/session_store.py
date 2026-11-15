from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from agent.db import open_connection, transaction, utc_now

ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Session:
    session_id: str
    title: str | None
    system_prompt: str
    summary: str
    summarised_until: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    message_id: int
    session_id: str
    role: str
    content: str
    created_at: str
    token_count: int | None


class SessionStore:
    """Chat sessions and their message history, persisted in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator:
        with transaction(open_connection(self.db_path)) as connection:
            yield connection

    def _initialize(self) -> None:
        with self._connection() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    summarised_until INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "summarised_until" not in columns:
                db.execute(
                    "ALTER TABLE sessions ADD COLUMN "
                    "summarised_until INTEGER NOT NULL DEFAULT 0"
                )
            db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL
                        REFERENCES sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL
                        CHECK (role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    token_count INTEGER
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, message_id)
            """)

    @staticmethod
    def _session(row) -> Session:
        return Session(
            session_id=row["session_id"],
            title=row["title"],
            system_prompt=row["system_prompt"],
            summary=row["summary"],
            summarised_until=row["summarised_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _message(row) -> Message:
        return Message(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            token_count=row["token_count"],
        )

    def create_session(
        self,
        system_prompt: str = "",
        *,
        title: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")

        session_id = session_id or uuid.uuid4().hex
        now = utc_now()
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO sessions (
                    session_id, title, system_prompt, summary,
                    created_at, updated_at
                ) VALUES (?, ?, ?, '', ?, ?)
                """,
                (session_id, title, system_prompt, now, now),
            )

        session = self.get_session(session_id)
        assert session is not None
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session(row) if row else None

    def list_sessions(self, limit: int | None = None) -> list[Session]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        query = "SELECT * FROM sessions ORDER BY updated_at DESC, session_id"
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self._connection() as db:
            rows = db.execute(query, params).fetchall()
        return [self._session(row) for row in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        token_count: int | None = None,
    ) -> Message:
        if role not in ROLES:
            raise ValueError(f"Invalid message role: {role}")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if token_count is not None and (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")

        now = utc_now()
        with self._connection() as db:
            exists = db.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"Session not found: {session_id}")

            cursor = db.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, created_at, token_count
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, token_count),
            )
            db.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            message_id = cursor.lastrowid

        return Message(
            message_id=int(message_id),
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
            token_count=token_count,
        )

    def messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        if self.get_session(session_id) is None:
            raise KeyError(f"Session not found: {session_id}")

        with self._connection() as db:
            if limit is None:
                rows = db.execute(
                    "SELECT * FROM messages WHERE session_id = ? "
                    "ORDER BY message_id",
                    (session_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY message_id DESC
                        LIMIT ?
                    ) ORDER BY message_id
                    """,
                    (session_id, limit),
                ).fetchall()

        return [self._message(row) for row in rows]

    def set_summary(
        self,
        session_id: str,
        summary: str,
        *,
        summarised_until: int | None = None,
    ) -> Session:
        """Store the summary and, optionally, the last message it covers."""
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        if summarised_until is not None and (
            isinstance(summarised_until, bool)
            or not isinstance(summarised_until, int)
            or summarised_until < 0
        ):
            raise ValueError("summarised_until must be a non-negative integer")

        now = utc_now()
        with self._connection() as db:
            if summarised_until is None:
                cursor = db.execute(
                    "UPDATE sessions SET summary = ?, updated_at = ? "
                    "WHERE session_id = ?",
                    (summary, now, session_id),
                )
            else:
                cursor = db.execute(
                    "UPDATE sessions SET summary = ?, summarised_until = ?, "
                    "updated_at = ? WHERE session_id = ?",
                    (summary, summarised_until, now, session_id),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")

        session = self.get_session(session_id)
        assert session is not None
        return session

    def delete_message(self, message_id: int) -> None:
        """Remove one message, used to roll back a turn that never completed."""
        if isinstance(message_id, bool) or not isinstance(message_id, int):
            raise ValueError("message_id must be an integer")

        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM messages WHERE message_id = ?", (message_id,)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Message not found: {message_id}")
