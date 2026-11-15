import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = temp_dir
        self.db_path = Path(temp_dir.name) / "sessions.sqlite3"
        self.store = SessionStore(self.db_path)

    def test_create_session_defaults(self):
        session = self.store.create_session("You are helpful.")

        self.assertTrue(session.session_id)
        self.assertEqual(session.system_prompt, "You are helpful.")
        self.assertEqual(session.summary, "")
        self.assertIsNone(session.title)
        self.assertTrue(session.created_at)
        self.assertTrue(session.updated_at)

    def test_missing_session_returns_none(self):
        self.assertIsNone(self.store.get_session("missing"))

    def test_append_and_read_messages_in_order(self):
        session = self.store.create_session()
        first = self.store.append_message(session.session_id, "user", "one")
        second = self.store.append_message(
            session.session_id, "assistant", "two", token_count=7
        )

        messages = self.store.messages(session.session_id)

        self.assertEqual(
            [message.content for message in messages], ["one", "two"]
        )
        self.assertEqual(messages[1].token_count, 7)
        self.assertGreater(second.message_id, first.message_id)

    def test_messages_limit_returns_newest_block(self):
        session = self.store.create_session()
        for index in range(5):
            self.store.append_message(
                session.session_id, "user", f"message {index}"
            )

        messages = self.store.messages(session.session_id, limit=2)

        self.assertEqual(
            [message.content for message in messages],
            ["message 3", "message 4"],
        )

    def test_invalid_role_is_rejected(self):
        session = self.store.create_session()
        with self.assertRaises(ValueError):
            self.store.append_message(session.session_id, "tool", "nope")

    def test_append_to_missing_session_raises(self):
        with self.assertRaises(KeyError):
            self.store.append_message("missing", "user", "hello")
        with self.assertRaises(KeyError):
            self.store.messages("missing")

    def test_summary_is_persisted(self):
        session = self.store.create_session()
        self.store.set_summary(session.session_id, "we chose sqlite")

        reopened = SessionStore(self.db_path)
        recovered = reopened.get_session(session.session_id)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.summary, "we chose sqlite")

    def test_set_summary_on_missing_session_raises(self):
        with self.assertRaises(KeyError):
            self.store.set_summary("missing", "summary")

    def test_sessions_list_most_recently_updated_first(self):
        older = self.store.create_session(title="older")
        newer = self.store.create_session(title="newer")
        self.store.append_message(older.session_id, "user", "updated now")

        sessions = self.store.list_sessions()

        self.assertEqual({item.session_id for item in sessions},
                         {older.session_id, newer.session_id})
        self.assertEqual(sessions[0].session_id, older.session_id)

    def test_legacy_database_gains_the_watermark_column(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        legacy = sqlite3.connect(legacy_path)
        legacy.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                system_prompt TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        legacy.execute("""
            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                token_count INTEGER
            )
        """)
        legacy.execute(
            "INSERT INTO sessions VALUES ('old', 't', 'sys', 'sum', 'a', 'b')"
        )
        legacy.commit()
        legacy.close()

        store = SessionStore(legacy_path)
        session = store.get_session("old")

        self.assertIsNotNone(session)
        self.assertEqual(session.summarised_until, 0)
        self.assertEqual(session.summary, "sum")
        store.append_message("old", "user", "still works")
        self.assertEqual(len(store.messages("old")), 1)

    def test_set_summary_stores_the_watermark(self):
        session = self.store.create_session()
        updated = self.store.set_summary(
            session.session_id, "summary", summarised_until=42
        )

        self.assertEqual(updated.summarised_until, 42)
        self.assertEqual(
            self.store.get_session(session.session_id).summarised_until, 42
        )

    def test_delete_message_removes_only_that_row(self):
        session = self.store.create_session()
        first = self.store.append_message(session.session_id, "user", "one")
        self.store.append_message(session.session_id, "assistant", "two")

        self.store.delete_message(first.message_id)

        remaining = self.store.messages(session.session_id)
        self.assertEqual([message.content for message in remaining], ["two"])
        with self.assertRaises(KeyError):
            self.store.delete_message(first.message_id)

    def test_messages_survive_reopen(self):
        session = self.store.create_session()
        self.store.append_message(session.session_id, "user", "persisted")

        reopened = SessionStore(self.db_path)
        messages = reopened.messages(session.session_id)

        self.assertEqual([message.content for message in messages],
                         ["persisted"])


if __name__ == "__main__":
    unittest.main()
