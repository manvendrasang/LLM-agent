from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.db import open_connection, transaction, utc_now


STATUSES = {"queued", "running", "completed", "failed"}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    description: str
    status: str
    created_at: str
    updated_at: str
    checkpoint: dict[str, Any]
    result: str | None
    error: str | None


class TaskStore:
    """Persistent task records and resumable checkpoints."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        # Single connection factory: tests subclass the store to assert every
        # handle is closed, so this stays a method even though it is a one-liner.
        return open_connection(self.db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        # transaction() scopes the commit and closes the handle afterwards;
        # sqlite3.Connection alone would only scope the transaction.
        with transaction(self._connect()) as connection:
            yield connection

    def _initialize(self) -> None:
        with self._connection() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'queued', 'running', 'completed', 'failed'
                        )),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    result TEXT,
                    error TEXT
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
            """)

    @staticmethod
    def _record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            description=row["description"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            checkpoint=json.loads(row["checkpoint_json"]),
            result=row["result"],
            error=row["error"],
        )

    def create_task(self, description: str) -> TaskRecord:
        description = description.strip()
        if not description:
            raise ValueError("Task description must not be empty")

        task_id = uuid.uuid4().hex
        now = utc_now()

        with self._connection() as db:
            db.execute(
                """
                INSERT INTO tasks (
                    task_id, description, status,
                    created_at, updated_at, checkpoint_json
                ) VALUES (?, ?, 'queued', ?, ?, '{}')
                """,
                (task_id, description, now, now),
            )

        task = self.get_task(task_id)
        if task is None:
            raise RuntimeError(f"task {task_id} vanished after insert")
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()

        return self._record(row) if row else None

    def list_tasks(self, status: str | None = None) -> list[TaskRecord]:
        if status is not None and status not in STATUSES:
            raise ValueError(f"Invalid task status: {status}")

        with self._connection() as db:
            if status is None:
                rows = db.execute(
                    "SELECT * FROM tasks ORDER BY created_at, task_id"
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT * FROM tasks
                    WHERE status = ?
                    ORDER BY created_at, task_id
                    """,
                    (status,),
                ).fetchall()

        return [self._record(row) for row in rows]

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> TaskRecord:
        if status not in STATUSES:
            raise ValueError(f"Invalid task status: {status}")

        now = utc_now()
        with self._connection() as db:
            cursor = db.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, result = ?, error = ?
                WHERE task_id = ?
                """,
                (status, now, result, error, task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Task not found: {task_id}")

        task = self.get_task(task_id)
        if task is None:
            raise RuntimeError(f"task {task_id} vanished after update")
        return task

    def save_checkpoint(
        self,
        task_id: str,
        checkpoint: dict[str, Any],
    ) -> TaskRecord:
        # Validate that the checkpoint can be serialized before writing.
        serialized = json.dumps(checkpoint, ensure_ascii=False)
        now = utc_now()

        with self._connection() as db:
            cursor = db.execute(
                """
                UPDATE tasks
                SET checkpoint_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (serialized, now, task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Task not found: {task_id}")

        task = self.get_task(task_id)
        if task is None:
            raise RuntimeError(f"task {task_id} vanished after checkpoint")
        return task
