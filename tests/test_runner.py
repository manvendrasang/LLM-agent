import tempfile
import unittest
from pathlib import Path

from agent.llm_client import LLMConnectionError, LLMResponse
from agent.runner import Step, TaskRunner, run_task
from agent.task_store import TaskStore


class FakeClient:
    """Stands in for LocalLLMClient so tests never touch the network."""

    def __init__(self, replies=None, fail_on_call=None, observer=None):
        self.replies = list(replies or [])
        self.fail_on_call = fail_on_call
        self.observer = observer
        self.calls = []

    def chat(self, messages, *, temperature=0.0, max_tokens=256):
        self.calls.append(list(messages))
        if self.observer is not None:
            self.observer(len(self.calls), self)
        if self.fail_on_call == len(self.calls):
            raise LLMConnectionError("Could not connect to the inference server")

        index = len(self.calls) - 1
        text = self.replies[index] if index < len(self.replies) else f"reply {index}"
        return LLMResponse(
            text=text,
            model="fake-model",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_seconds=0.0,
        )


class TaskRunnerTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.store = TaskStore(Path(temp_dir.name) / "tasks.sqlite3")

    def test_default_step_sends_description_and_completes_task(self):
        task = self.store.create_task("Explain the task store")
        client = FakeClient(replies=["It stores tasks."])

        result = run_task(self.store, client, task.task_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            client.calls[0],
            [{"role": "user", "content": "Explain the task store"}],
        )
        self.assertEqual(result.outputs, ("It stores tasks.",))

        stored = self.store.get_task(task.task_id)
        self.assertEqual(stored.status, "completed")
        self.assertEqual(stored.result, "It stores tasks.")
        self.assertEqual(stored.checkpoint["step_index"], 1)
        self.assertEqual(stored.checkpoint["completed_steps"], ["answer"])
        self.assertEqual(
            stored.checkpoint["messages"][-1],
            {"role": "assistant", "content": "It stores tasks."},
        )

    def test_each_step_is_checkpointed_before_the_next_call(self):
        task = self.store.create_task("Two steps")
        snapshots = []

        def observer(call_number, _client):
            # Called after a request was sent but before the next one.
            snapshots.append(
                self.store.get_task(task.task_id).checkpoint.get("step_index", 0)
            )

        client = FakeClient(observer=observer)
        runner = TaskRunner(
            self.store,
            client,
            steps=[Step("draft", "Draft: {task}"), Step("review", "Review it")],
        )
        runner.run(task.task_id)

        self.assertEqual(snapshots, [0, 1])

    def test_later_steps_see_earlier_turns(self):
        task = self.store.create_task("Chain steps")
        client = FakeClient(replies=["first answer", "second answer"])
        runner = TaskRunner(
            self.store,
            client,
            steps=[Step("first", "Ask one"), Step("second", "Ask two")],
        )

        result = runner.run(task.task_id)

        second_call = client.calls[1]
        self.assertEqual(
            [message["role"] for message in second_call],
            ["user", "assistant", "user"],
        )
        self.assertEqual(second_call[1]["content"], "first answer")
        self.assertEqual(second_call[2]["content"], "Ask two")
        self.assertEqual(result.outputs, ("first answer", "second answer"))
        self.assertEqual(
            self.store.get_task(task.task_id).result,
            "second answer",
        )

    def test_run_resumes_from_saved_checkpoint(self):
        task = self.store.create_task("Resume me")
        self.store.save_checkpoint(
            task.task_id,
            {
                "step_index": 1,
                "completed_steps": ["first"],
                "messages": [
                    {"role": "user", "content": "Ask one"},
                    {"role": "assistant", "content": "already done"},
                ],
                "outputs": ["already done"],
            },
        )
        self.store.update_status(task.task_id, "failed", error="boom")
        client = FakeClient(replies=["fresh"])
        runner = TaskRunner(
            self.store,
            client,
            steps=[Step("first", "Ask one"), Step("second", "Ask two")],
        )

        result = runner.run(task.task_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][-1], {"role": "user", "content": "Ask two"})
        self.assertEqual(result.outputs, ("already done", "fresh"))
        self.assertEqual(self.store.get_task(task.task_id).status, "completed")

    def test_failure_marks_task_failed_and_keeps_resume_point(self):
        task = self.store.create_task("Fail midway")
        steps = [Step("first", "Ask one"), Step("second", "Ask two")]
        failing = FakeClient(replies=["ok"], fail_on_call=2)
        runner = TaskRunner(self.store, failing, steps=steps)

        with self.assertRaises(LLMConnectionError):
            runner.run(task.task_id)

        failed = self.store.get_task(task.task_id)
        self.assertEqual(failed.status, "failed")
        self.assertIn("LLMConnectionError", failed.error)
        self.assertEqual(failed.checkpoint["step_index"], 1)

        retry_client = FakeClient(replies=["recovered"])
        result = TaskRunner(self.store, retry_client, steps=steps).run(
            task.task_id
        )

        self.assertEqual(len(retry_client.calls), 1)
        self.assertEqual(client_content(retry_client.calls[0][-1]), "Ask two")
        self.assertEqual(result.outputs, ("ok", "recovered"))
        self.assertEqual(self.store.get_task(task.task_id).status, "completed")

    def test_completed_task_is_not_sent_again(self):
        task = self.store.create_task("Already done")
        self.store.save_checkpoint(
            task.task_id,
            {
                "step_index": 1,
                "completed_steps": ["answer"],
                "messages": [
                    {"role": "user", "content": "Already done"},
                    {"role": "assistant", "content": "finished earlier"},
                ],
                "outputs": ["finished earlier"],
            },
        )
        self.store.update_status(task.task_id, "completed", result="finished earlier")
        client = FakeClient()

        result = run_task(self.store, client, task.task_id)

        self.assertEqual(client.calls, [])
        self.assertEqual(result.outputs, ("finished earlier",))
        self.assertEqual(result.task.status, "completed")

    def test_unknown_task_raises_key_error(self):
        with self.assertRaises(KeyError):
            run_task(self.store, FakeClient(), "missing-task-id")

    def test_invalid_step_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskRunner(self.store, FakeClient(), steps=[])
        with self.assertRaises(ValueError):
            TaskRunner(
                self.store,
                FakeClient(),
                steps=[Step("same", "a"), Step("same", "b")],
            )

    def test_corrupt_checkpoint_restarts_the_task(self):
        task = self.store.create_task("Corrupt state")
        self.store.save_checkpoint(task.task_id, {"step_index": "nonsense"})
        client = FakeClient(replies=["started over"])

        result = run_task(self.store, client, task.task_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.outputs, ("started over",))


def client_content(message):
    return message["content"]


if __name__ == "__main__":
    unittest.main()
