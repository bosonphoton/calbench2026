# CalBench Trace Selection

This note records where the local trace mirror lives and which CalBench mixed
traces are intended for study cohorts.

## Local Trace Mirror

From this submodule, the trace mirror is two directories up:

```bash
../../shared-traces/calbench-mixed
```

From the parent `a2a-framework` repo root, the same directory is:

```bash
shared-traces/calbench-mixed
```

Each run directory contains worker outputs plus derived indexes/reports:

- raw trace JSONs: `RUN_ID/worker-*/.../*.json`
- run index: `RUN_ID/_index/index.json`
- derived reports: `RUN_ID/_reports/...`

Ignore `*.metadata.json`, `_run_manifest.jsonl`, `_index/`, and `_reports/`
when iterating over raw game traces.

## Main Model Cohort

For systematic model comparison, use the initial CalBench 90-game calibration
runs, not the later live-matchmaking runs. The balanced seven-model cohort is
assembled from the initial five-model mixed run plus the Qwen and DeepSeek
arena-entry calibration runs. The live-matchmaking traces are useful for
deployed leaderboard audits, but they were generated under an adaptive
matchmaking process and should not be mixed into the balanced main cohort.

| Run | Role | Games |
| --- | --- | ---: |
| `mixed5-calbench90-001` | initial five-model cohort | 90 |
| `qwen36-plus-arena-entry-reflection-002` | Qwen 3.6 Plus arena-entry cohort | 90 |
| `deepseek-v4-pro-arena-entry-reflection-002` | DeepSeek V4 Pro uniform half | 45 |
| `deepseek-v4-pro-arena-entry-reflection-002-varied` | DeepSeek V4 Pro varied half | 45 |

Together, the two DeepSeek reflection-002 halves form the intended 90-game
DeepSeek initial-calibration cohort. The older uniform-only
`deepseek-v4-pro-arena-entry-reflection-002` directory should not be double
counted against a combined varied mirror if both copies are present.

## Baseline Cohorts

Static/protocol baselines are not part of model-based leaderboard comparisons,
but they are the comparison point for protocol-semantic VPS and other baseline
analyses:

| Run | Role | Games |
| --- | --- | ---: |
| `baseline-dsm-private-calbench-mixed-001` | DSM private baseline | 90 |
| `baseline-dsm-welfare-calbench-mixed-001` | DSM welfare baseline | 90 |
| `baseline-imap-calbench-mixed-001` | IMAP baseline | 90 |
| `baseline-sd-map-calbench-mixed-001` | SD-MAP baseline | 90 |

## Exclusions

Do not include these in the main cohort:

- `shared-traces/calbench-mixed/live-matchmaking/*`
- smoke tests such as `mixed5-calbench90-smoke-002` and
  `mixed5-calbench90-smoke-003`
- quarantined DeepSeek run `deepseek-v4-pro-arena-entry-001`
- duplicate copies of the 45-game uniform-only DeepSeek reflection-002 slice

## Useful Checks

Count raw traces for a run:

```bash
find ../../shared-traces/calbench-mixed/RUN_ID \
  -type f -name '*.json' \
  ! -name '*.metadata.json' \
  ! -path '*/_reports/*' \
  ! -path '*/_index/*' | wc -l
```

Find existing derived reports:

```bash
find ../../shared-traces/calbench-mixed/RUN_ID/_reports -maxdepth 2 -type f | sort
```

For the current VPS calibration work, the relevant local reports are:

- DSM-private protocol-semantic VPS:
  `../../shared-traces/calbench-mixed/baseline-dsm-private-calbench-mixed-001/_reports/vps/`
- mixed5 reflection VPS:
  `../../shared-traces/calbench-mixed/mixed5-calbench90-001/_reports/reflection_vps/`
  and `../../shared-traces/calbench-mixed/mixed5-calbench90-001/_reports/reflection_vps_excess/`
