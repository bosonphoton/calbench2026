# CalBench

CalBench is a controlled benchmark for studying coordination and privacy trade-offs in multi-agent LLM calendar scheduling. Each agent owns a private calendar, incoming meetings require agreement across multiple agents, and the evaluator compares outcomes against known oracle schedules.

The benchmark is designed to separate:

- task success: how many meetings are scheduled correctly;
- coordination quality: excess displacement cost relative to the CP-SAT oracle;
- communication efficiency: message volume and channel use;
- fairness: how scheduling burden is distributed across agents;
- privacy leakage: structural calendar-state leakage measured with VPS, plus separate semantic-context audits.

## Repository Layout

| Path | Contents |
| --- | --- |
| `calendar_game/` | CalBench environment, agents, clients, task generation, solver, metrics, and tests. |
| `tasks/` | Public task fixtures used by the paper, including 90 uniform-cost and 90 varied-cost tasks. |
| `taskgen_configs/` | YAML configs for regenerating task fixtures. |
| `experiments/` | Example experiment configs for scripted, OpenRouter, Vertex, and mixed-model runs. |
| `analysis/scripts/` | Scripts used to build metrics, plots, VPS estimates, and paper tables. |
| `final_results/` | Aggregated result tables and plots used for the paper. Raw traces are intentionally omitted. |
| `a2a-viewer/` | Static trace viewer for inspecting generated game traces. |
| `vendor/` | Small companion `a2a-engine` and `expt-runner` packages needed to run experiments. |

## Installation

This project uses Python 3.11+ and `uv`.

```bash
git clone <anonymous-review-repository-url>
cd calbench2026
uv sync
```

The release repository vendors the small companion `a2a-engine` and `expt-runner` packages under `vendor/`, so `uv sync` does not require private sibling checkouts.

## Quick Checks

Run the unit tests:

```bash
uv run pytest calendar_game/tests
```

Regenerate a tiny task fixture:

```bash
uv run python -m calendar_game.taskgen --config taskgen_configs/tiny_varied_density_5a3p.yaml
```

Dry-run an experiment without model API calls:

```bash
uv run python run.py experiments/example.yaml --dry-run
```

## Public Fixtures

The main task fixtures are:

| Dataset | Path | Setting |
| --- | --- | --- |
| Uniform 5-agent / 3-participant | `tasks/calbench_90_uniform.jsonl` | all errands and meetings have unit displacement cost |
| Varied 5-agent / 3-participant | `tasks/calbench_90_varied.jsonl` | errands use costs from `{1, 10, 100}` and meetings have unit cost |

Each task row contains scenario parameters, private calendars, incoming meetings, a witness feasible solution, the CP-SAT oracle solution, a greedy baseline solution, and difficulty metadata.

Regenerate the two public 90-task suites:

```bash
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_90_uniform.yaml
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_90_varied.yaml
```

## Running LLM Experiments

LLM experiments are configured with YAML files under `experiments/`.

Provider credentials are read from environment variables or provider-specific application default credentials. Do not commit API keys or service-account JSON files.

Common environment variables:

| Provider | Credential |
| --- | --- |
| OpenAI-compatible APIs | `OPENAI_API_KEY` or provider-specific `api_key` field |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini API | `GOOGLE_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` or an explicit local `vertex_adc_file` path |

Some checked-in Vertex configs use placeholder ADC paths such as `${GOOGLE_APPLICATION_CREDENTIALS}`. Replace these with local credential paths or rely on standard Google ADC before running.

## Results and Traces

`final_results/` contains aggregate CSVs and plots used in the paper. Full raw model traces are not committed because they are larger, may contain provider metadata, and include simulated private calendar contexts. `TRACE_SELECTION.md` documents the canonical trace cohort used to build the reported results.

To rebuild result tables from a local trace directory, use the scripts in `analysis/scripts/`; most scripts accept explicit input/output paths and can be run with `uv run python`.

## Citation

See `CITATION.cff` for citation metadata.
