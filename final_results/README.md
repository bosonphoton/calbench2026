# Final Results

This directory contains aggregate result tables and plots used by the paper.

Raw model traces are not included in the publication repository. The CSV files here are derived summaries over the canonical cohort described in `../TRACE_SELECTION.md`.

Important files:

- `summary.md`: human-readable overview of headline plots and tables.
- `summary_by_model_setting.csv`: model-level metrics split by uniform and varied settings.
- `summary_by_model_difficulty.csv`: model-level metrics split by difficulty bucket.
- `trace_game_metrics.csv`: per-trace game-level metrics.
- `trace_agent_metrics.csv`: per-agent metrics.
- `failure_reasons.csv` and `decision_failure_causes.csv`: categorized failure diagnostics.

Figures in this directory are generated from the corresponding aggregate CSVs.
