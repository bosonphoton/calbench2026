# expt-runner

Tiny CLI that loads an experiment YAML, expands batches, runs each game
in-process via `a2a-engine`, and writes one JSON trace per run.

## Install

```bash
uv pip install -e expt-runner
```

## Usage

```bash
a2a-run path/to/experiment.yaml \
    --max-parallelism 4 \
    --results-dir ./results
```

Or directly:

```bash
uv run python -m expt_runner.run_experiment path/to/experiment.yaml
```

## Flags

| Flag | Default | Effect |
|---|---|---|
| `yaml_path` (positional) | required | Experiment YAML to load |
| `--max-parallelism N` | 4 | Threadpool size; 1 = sequential |
| `--results-dir DIR` | `./results` | Traces are written to `<DIR>/<experiment_name>/<game_id>.json` |
| `--dry-run` | off | Forwarded to the game class as a hint to skip LLM calls |
| `--log-level` | `INFO` | Standard logging level |
| `--resume` | off | Skip runs whose `experiment_run_id` already exists in local traces/metadata/manifest |
| `--s3-bucket BUCKET` | `A2A_TRACE_BUCKET` | Best-effort upload each completed trace to S3 after local save |
| `--s3-prefix PREFIX` | `A2A_TRACE_PREFIX` or `calendar-traces` | S3 key prefix |
| `--s3-profile PROFILE` | `AWS_PROFILE` | AWS CLI profile for upload |
| `--s3-uploader NAME` | `A2A_TRACE_USER` or `$USER` | Collaborator prefix and metadata label |

## YAML format

See `a2a-engine/examples/example_experiment.yaml`. Per-batch `config:` is
deep-merged on top of experiment `defaults:`.

## Registering a game

Game classes are looked up by `game_name`. Your benchmark package should
register itself at import time:

```python
# in your_benchmark/__init__.py
from a2a_engine import register_game
from your_benchmark.game import CalendarGame
register_game("calendar", CalendarGame)
```

Then either import that package before invoking `a2a-run`, or wire it through
a Python entry-point in your benchmark's `pyproject.toml`.

A registered game must satisfy:

```python
class GameCls:
    def __init__(self, config: dict, dry_run: bool = False) -> None: ...
    def run(self) -> GameTraceBase: ...
```

## Tracing & Langfuse

Tracing is opt-in. With no `OTEL_*` env vars set, the runner installs an
in-memory `TracerProvider` but no exporter, so no spans are shipped (and the
existing JSON trace files are unaffected).

Each run emits:

- a root `game <game_name>` span with `gen_ai.conversation.id`, `langfuse.session.id`,
  and `langfuse.trace.tags = [experiment_name, batch_label]`
- one `invoke_agent <agent_name>` span per `agent.act()` call
- one `chat <model>` child span per LLM API call, with full
  `gen_ai.*` attributes (provider, model, usage tokens, finish reasons, errors)

### Local debug

```bash
OTEL_TRACES_EXPORTER=console uv run python run.py experiments/example.yaml --dry-run
```

### Local JSONL traces

```bash
A2A_OTEL_TRACES_FILE=./results/otel-spans.jsonl \
  uv run python run.py experiments/example.yaml --dry-run
```

Each completed span is appended as one JSON object per line. You can also set
`OTEL_TRACES_EXPORTER=file`, which writes to `./results/otel-traces.jsonl`.

### Langfuse Cloud

```bash
AUTH=$(printf '%s' "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" | base64)
export OTEL_EXPORTER_OTLP_ENDPOINT="https://us.cloud.langfuse.com/api/public/otel"   # or https://cloud.langfuse.com/... for EU
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH},x-langfuse-ingestion-version=4"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
```

### Capture prompts/completions

By default the spans omit message bodies. To include them as structured
`gen_ai.input.messages` / `gen_ai.output.messages` attributes:

```bash
export A2A_CAPTURE_CONTENT=true
```

## Resume and S3 trace sync

Every non-dry-run trace is first written locally to:

```text
<results-dir>/<experiment_name>/<game_id>.json
```

The runner also writes a small sidecar metadata file:

```text
<results-dir>/<experiment_name>/<game_id>.metadata.json
```

and appends the same record to:

```text
<results-dir>/<experiment_name>/_run_manifest.jsonl
```

Use `--resume` to safely re-run an interrupted experiment collection. Completed
`experiment_run_id` values found in local trace JSON, metadata sidecars, or the
manifest are skipped.

S3 upload is optional and best-effort. If the S3 upload fails, the local trace
and metadata remain written and the experiment run continues.

```bash
A2A_TRACE_BUCKET=calbench-a2a-traces \
AWS_PROFILE=a2a-calendar \
A2A_TRACE_USER=alice \
uv run python run.py experiments/example.yaml --resume
```

Remote traces are written as:

```text
s3://<bucket>/<prefix>/<uploader>/<experiment_name>/<experiment_run_id>/<game_id>.json
s3://<bucket>/<prefix>/<uploader>/<experiment_name>/<experiment_run_id>/<game_id>.metadata.json
```
