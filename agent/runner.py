from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.llm_client import LocalLLMClient
from agent.task_store import TaskRecord, TaskStore


@dataclass(frozen=True)
class Step:
    """One request to the model, recorded in the task checkpoint."""

    name: str
    prompt: str

    def render(self, description: str) -> str:
        try:
            return self.prompt.format(task=description)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Step '{self.name}' has an invalid prompt template: {exc}"
            ) from exc


DEFAULT_STEPS = (Step(name="answer", prompt="{task}"),)


@dataclass(frozen=True)
class RunResult:
    task: TaskRecord
    outputs: tuple[str, ...]
    messages: tuple[dict[str, str], ...]


def _checkpoint_state(
    task: TaskRecord,
    step_count: int,
) -> tuple[int, list[dict[str, str]], list[str]]:
    """Read a checkpoint defensively; unusable state restarts the task."""
    checkpoint: Any = task.checkpoint or {}
    if not isinstance(checkpoint, dict):
        return 0, [], []

    index = checkpoint.get("step_index", 0)
    messages = checkpoint.get("messages", [])
    outputs = checkpoint.get("outputs", [])

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return 0, [], []
    if not isinstance(messages, list) or not isinstance(outputs, list):
        return 0, [], []
    for message in messages:
        if not (
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and isinstance(message.get("content"), str)
        ):
            return 0, [], []
    if not all(isinstance(output, str) for output in outputs):
        return 0, [], []

    # A changed step list must not resume past its new end.
    index = min(index, step_count)
    return index, list(messages), list(outputs)


class TaskRunner:
    """Drive a stored task through model calls, checkpointing every step.

    Each step sends the accumulated conversation to the model and saves the
    transcript to the task store afterwards, so an interrupted run resumes at
    the first step that never completed instead of starting over.
    """

    def __init__(
        self,
        store: TaskStore,
        client: LocalLLMClient,
        *,
        steps: tuple[Step, ...] | list[Step] = DEFAULT_STEPS,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        steps = tuple(steps)
        if not steps:
            raise ValueError("steps must not be empty")
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")
        if not all(isinstance(step, Step) for step in steps):
            raise ValueError("steps must contain Step instances")

        self.store = store
        self.client = client
        self.steps = steps
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, task_id: str) -> RunResult:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")

        index, messages, outputs = _checkpoint_state(task, len(self.steps))
        if task.status == "completed":
            return RunResult(
                task=task,
                outputs=tuple(outputs),
                messages=tuple(messages),
            )

        task = self.store.update_status(task.task_id, "running")

        try:
            for position in range(index, len(self.steps)):
                step = self.steps[position]
                conversation = messages + [
                    {"role": "user", "content": step.render(task.description)}
                ]
                response = self.client.chat(
                    conversation,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                messages = conversation + [
                    {"role": "assistant", "content": response.text}
                ]
                outputs = outputs + [response.text]
                self.store.save_checkpoint(
                    task.task_id,
                    {
                        "step_index": position + 1,
                        "completed_steps": [
                            item.name for item in self.steps[: position + 1]
                        ],
                        "messages": messages,
                        "outputs": outputs,
                    },
                )
        except Exception as exc:
            # Keep the last good checkpoint so a retry continues where this
            # attempt stopped.
            self.store.update_status(
                task.task_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        if not outputs:
            # Empty step list cannot happen (validated in __init__), but a task
            # already checkpointed past the end has no output of its own.
            result = ""
        else:
            result = outputs[-1]

        completed = self.store.update_status(
            task.task_id,
            "completed",
            result=result,
        )
        return RunResult(
            task=completed,
            outputs=tuple(outputs),
            messages=tuple(messages),
        )


def run_task(
    store: TaskStore,
    client: LocalLLMClient,
    task_id: str,
    **kwargs: Any,
) -> RunResult:
    """Create a runner for a single task and execute it."""
    return TaskRunner(store, client, **kwargs).run(task_id)
