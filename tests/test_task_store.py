import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.task_store import TaskStore


class RecordingStore(TaskStore):
    """TaskStore that keeps every connection it opens."""

    def __init__(self, db_path):
        self.connections = []
        super().__init__(db_path)

    def _connect(self):
        connection = super()._connect()
        self.connections.append(connection)
        return connection


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "tasks.sqlite3"
        self.store = TaskStore(self.db_path)

    def test_create_task_starts_queued(self):
        task = self.store.create_task("Build a test agent")

        self.assertTrue(task.task_id)
        self.assertEqual(task.description, "Build a test agent")
        self.assertEqual(task.status, "queued")
        self.assertEqual(task.checkpoint, {})
        self.assertIsNone(task.result)
        self.assertIsNone(task.error)
        self.assertTrue(task.created_at)
        self.assertTrue(task.updated_at)

    def test_rejects_empty_description(self):
        for description in ("", "   ", "\n\t"):
            with self.subTest(description=repr(description)):
                with self.assertRaises(ValueError):
                    self.store.create_task(description)

    def test_checkpoint_survives_new_store_instance(self):
        task = self.store.create_task("Resume a task")
        checkpoint = {
            "phase": "retrieval",
            "completed_steps": ["inspect", "index"],
            "metadata": {"attempt": 2},
        }

        self.store.save_checkpoint(task.task_id, checkpoint)

        reopened_store = TaskStore(self.db_path)
        recovered = reopened_store.get_task(task.task_id)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.checkpoint, checkpoint)

    def test_status_and_result_are_persisted(self):
        task = self.store.create_task("Generate a report")

        updated = self.store.update_status(
            task.task_id,
            "completed",
            result="Report generated",
        )

        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.result, "Report generated")
        self.assertIsNone(updated.error)

        reopened_store = TaskStore(self.db_path)
        recovered = reopened_store.get_task(task.task_id)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.result, "Report generated")

    def test_missing_task_raises_key_error_on_update(self):
        with self.assertRaises(KeyError):
            self.store.update_status("missing-task-id", "failed")

    def test_invalid_status_is_rejected(self):
        task = self.store.create_task("Reject an invalid status")

        with self.assertRaises(ValueError):
            self.store.update_status(task.task_id, "unknown")

        with self.assertRaises(ValueError):
            self.store.list_tasks(status="unknown")

    def test_connections_are_closed_after_each_operation(self):
        store = RecordingStore(self.db_path)
        task = store.create_task("Close my connections")
        store.save_checkpoint(task.task_id, {"step": 1})
        store.update_status(task.task_id, "completed", result="done")
        store.get_task(task.task_id)
        store.list_tasks(status="completed")

        self.assertGreater(len(store.connections), 1)
        for connection in store.connections:
            with self.subTest(connection=id(connection)):
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def test_failed_operation_still_closes_connection(self):
        store = RecordingStore(self.db_path)

        with self.assertRaises(KeyError):
            store.update_status("missing-task-id", "failed")

        self.assertTrue(store.connections)
        with self.assertRaises(sqlite3.ProgrammingError):
            store.connections[-1].execute("SELECT 1")

    def test_list_tasks_can_filter_by_status(self):
        queued = self.store.create_task("Queued task")
        completed = self.store.create_task("Completed task")
        self.store.update_status(completed.task_id, "completed")

        all_tasks = self.store.list_tasks()
        queued_tasks = self.store.list_tasks(status="queued")
        completed_tasks = self.store.list_tasks(status="completed")

        self.assertEqual(len(all_tasks), 2)
        self.assertEqual({t.task_id for t in queued_tasks}, {queued.task_id})
        self.assertEqual(
            {t.task_id for t in completed_tasks},
            {completed.task_id},
        )


if __name__ == "__main__":
    unittest.main()
