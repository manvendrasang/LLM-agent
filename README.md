# LLM-agent

Working notes and a small, dependency-free Python stack for running a local
coding model: a client for an OpenAI-compatible inference server, plus
persistent task records with resumable checkpoints.

Status: the client, the task store and a step runner are implemented and tested.
A task in the store can be sent to the model as a sequence of steps, with a
checkpoint saved after each one so an interrupted run resumes where it stopped.

## Layout

| Path | What it is |
| --- | --- |
| `agent/llm_client.py` | `LocalLLMClient`: sync `/v1/chat/completions` over `urllib`, stdlib only |
| `agent/task_store.py` | `TaskStore`: SQLite task records (`queued/running/completed/failed`) + JSON checkpoints |
| `agent/runner.py` | `TaskRunner`: drives a stored task through model calls, checkpointing every step |
| `tests/` | `unittest` suite for both modules |
| `models/gguf/` | Local GGUF weights (gitignored) |
| `logs/` | Baseline captures: hardware, llama.cpp build, throughput |
| `first.ipynb` | PyTorch walkthrough on `wizard.txt` (char-level tokenizer, tensor basics, CPU/GPU matmul) |
| `docs/`, `knowledge/`, `data/` | Empty scaffolding, not used yet |

## Requirements

- Python 3.10+ (the test suite passes on 3.14; no third-party packages needed)
- `llama-server` from llama.cpp on `PATH`
- A GGUF model — currently `models/gguf/qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2.1 GB, Q4_K_M)

The `cuda/` directory is a separate Python 3.11 virtualenv with CUDA-enabled
PyTorch, used only for `first.ipynb`. It is not needed by `agent/` or the tests:

```bash
cuda/bin/python -m jupyter notebook first.ipynb
```

## Start the inference server

```bash
llama-server \
  --model models/gguf/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias qwen2.5-coder-3b \
  --n-gpu-layers 99 \
  --ctx-size 4096
```

- **`--port 8080` matters.** This llama.cpp build defaults to port 9931, while
  `LocalLLMClient` defaults to `http://127.0.0.1:8080`. Either pass `--port 8080`
  as above or point the client somewhere else with `LLM_BASE_URL`.
- `--n-gpu-layers 99` offloads everything to the GPU. On a 4 GB card use `0` for
  CPU-only, or a smaller number to partially offload.
- `--alias` sets the model ID reported by `/v1/models`; the client otherwise
  accepts whatever the server reports.

Check that it is up before running anything:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Leave the server running in its own terminal; stop it with `Ctrl+C`.

Measured on an RTX 3050 (4 GB) with this model, build 11429: ~770 prompt t/s and
**67.3 generation t/s**. See `logs/` for the full captures.

## Configure the client

Both settings are optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://127.0.0.1:8080` | Base URL of the inference server (no trailing `/v1`) |
| `LLM_MODEL` | *(autodiscovered)* | Model ID sent in requests; if unset, the client reads `/v1/models` once and caches the first entry |

```bash
# server on a non-default port
export LLM_BASE_URL=http://127.0.0.1:9931

# skip autodiscovery and pin the alias explicitly
export LLM_MODEL=qwen2.5-coder-3b
```

## Use the client

```python
from agent.llm_client import LocalLLMClient

client = LocalLLMClient()  # or LocalLLMClient(base_url=..., model=..., timeout_seconds=120)
response = client.chat(
    [{"role": "user", "content": "Write a Python function that reverses a list."}],
    temperature=0.0,
    max_tokens=256,
)

print(response.text)
print(response.model, response.total_tokens, response.latency_seconds)
```

Errors are split by cause, so callers can retry the useful ones:

- `LLMConnectionError` — server not reachable
- `LLMTimeoutError` — request exceeded `timeout_seconds` (default 60 s)
- `LLMHTTPError` — non-2xx status, with `.status_code`
- `LLMResponseError` — invalid JSON or unexpected response shape

`chat()` is synchronous and has no retry logic. The first request after a cold
`llama-server` start includes model load time, so raise `timeout_seconds` if
you see timeouts right after launching the server.

## Use the task store

```python
from agent.task_store import TaskStore

store = TaskStore("state/tasks.sqlite3")
task = store.create_task("Summarise each file in agent/")

store.update_status(task.task_id, "running")
store.save_checkpoint(task.task_id, {"step": "inspect", "done": ["agent/"]})
store.update_status(task.task_id, "completed", result="done")

recovered = TaskStore("state/tasks.sqlite3").get_task(task.task_id)
print(recovered.checkpoint)
```

The database file is created on first use; `data/*.sqlite3` is gitignored.
Connections are opened per operation and closed when it ends.

## Run a task

`TaskRunner` sends a stored task to the model, step by step:

```python
from agent.llm_client import LocalLLMClient
from agent.runner import Step, TaskRunner
from agent.task_store import TaskStore

store = TaskStore("state/tasks.sqlite3")
client = LocalLLMClient(timeout_seconds=120)

task = store.create_task("Write a Python function that reverses a list.")

runner = TaskRunner(
    store,
    client,
    steps=[
        Step(name="draft", prompt="Write the code:\n\n{task}"),
        Step(name="review", prompt="Review the code above for bugs."),
    ],
    max_tokens=512,
)
result = runner.run(task.task_id)
print(result.outputs)      # one entry per completed step
print(result.task.status)  # "completed"
```

- `{task}` in a step prompt is replaced with the task description; the default
  step list is a single `answer` step that sends the description unchanged.
- After every step the transcript is written to the task's checkpoint
  (`step_index`, `completed_steps`, `messages`, `outputs`), and the next step
  receives the conversation so far.
- The status moves `queued`/`failed` → `running` → `completed`. On an error the
  task is marked `failed` with the error text, the last good checkpoint is kept,
  and the exception is re-raised — calling `run()` again retries only the step
  that never finished.
- A task already `completed` is returned without contacting the model.
- A checkpoint that cannot be read (wrong types) restarts the task from step 0.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

25 tests, no network or model required — the client tests start a local mock
HTTP server on an ephemeral port and the runner tests use a fake client. To run
a single module:

```bash
python3 -m unittest tests.test_llm_client -v
python3 -m unittest tests.test_task_store -v
python3 -m unittest tests.test_runner -v
```
