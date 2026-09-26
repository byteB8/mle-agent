# mle-agent

An autonomous ML-engineering agent built from scratch: given a Kaggle-style task (data, metric,
compute budget), it inspects the data, writes and runs training code in an isolated Docker sandbox,
validates locally, and submits predictions that are graded against held-out labels.

No agent frameworks. The agent loop, tool protocol, sandbox, tracing and evaluation harness are
all in this repo (~600 lines of Python, stdlib + pandas/sklearn for grading). The model is served
locally with vLLM (Qwen3-Coder-30B-A3B) through its OpenAI-compatible API.

**Phase 1** – agent + evaluation harness on a subset of MLE-bench Lite tasks: baseline ReAct loop,
then solution-tree search, memory and budget-aware scheduling, each measured with ablations.
**Phase 2** – distil successful trajectories into a small open model (LoRA) and compare
quality vs. cost against the large model.


## Layout

| path | what |
|---|---|
| `core/llm.py` | raw HTTP client for `/v1/chat/completions`: retries/backoff, tool-call normalisation, token accounting |
| `core/tools.py` | tool registry: JSON-schema definitions, argument validation/coercion, errors returned to the model |
| `core/sandbox.py` | Docker sandbox: read-only data, no network, CPU/mem/pid limits, in-container kill on timeout |
| `core/search.py` | solution-tree search agent: draft / improve / debug nodes, each a full script run and scored by the harness |
| `core/agent.py` | ReAct baseline agent loop: budgets (steps, wall-clock, tokens), context compaction, submission validation |
| `core/tasks.py` | task spec, submission format checks, graders |
| `core/trace.py` | append-only JSONL trace of every LLM and tool call |
| `run.py` / `prep.py` / `test.py` | run one episode / build task dirs / unit tests (no GPU or LLM needed) |
| `serve.sh` / `main.py` | start/stop vLLM (own venv, runs as the user) on the highest free GPU or a given one; generic process title |
| `setup.sh` | one-time server setup: vLLM venv, CUDA forward-compat libs, model weights (no root) |
| `batch.sh` | unattended multi-seed run: start server, run seeds in parallel, summarise, release GPU |
| `sweep.sh` | several experiment arms back to back on one server (ablations) |
| `stats.py` | per-run behaviour stats from traces (errors, CV use, early submits, tokens) |
| `env/Dockerfile` | sandbox runtime image |

## Running

```bash
./setup.sh                            # once
./serve.sh                            # vLLM on highest free GPU (3>2>1>0), localhost:8011
docker build -t exp-rt:cpu env/       # sandbox image
python prep.py dev-adult --out data   # smoke-test task
python run.py --task data/dev-adult --time-limit-min 30
python test.py
```
