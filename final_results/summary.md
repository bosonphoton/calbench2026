# CalBench Mixed Model Performance

Canonical cohort from `TRACE_SELECTION.md`; Qwen and DeepSeek include only entrant agent 0.

## Outputs
- `coordination_ratio_by_difficulty.png`
- `messaging_efficiency_by_excess_cost.png`
- `messaging_efficiency_by_difficulty.png`
- `excess_cost_by_difficulty_model.png`
- `message_channel_mix_overall.png`
- `fairness_by_difficulty_model.png`
- `fairness_by_setting_model.png`
- `failure_rate_by_meeting_index.png`
- `excess_cost_by_meeting_index.png`
- `excess_cost_by_success_count.png`
- `failure_reason_by_model.png`

## Task Success

| model | easy | medium | hard |
| --- | ---: | ---: | ---: |
| Claude Sonnet 4.6 | 84.4% | 82.2% | 81.1% |
| Gemini 3.1 Pro | 84.4% | 77.8% | 81.1% |
| Gemini 3 Flash | 87.8% | 81.1% | 75.6% |
| GPT-5.4 Mini | 81.1% | 71.1% | 74.4% |
| Llama 4 Maverick | 82.2% | 77.8% | 71.1% |
| Qwen3.6 Plus | 92.2% | 83.3% | 83.3% |
| DeepSeek V4 Pro | 94.4% | 84.4% | 82.2% |

## Failure Reasons

| reason | failed participant-meetings | description |
| --- | ---: | --- |
| blocked/unmovable slot | 164 | A participant attempted to move a blocked calendar item, or the failed outcome recorded a blocked-slot violation. |
| missing schedule action | 64 | The decision batch validator saw zero valid schedule actions for this meeting after parsing, filtering to the current meeting, and any retry attempts. |
| participants chose different slots | 57 | Participants submitted valid schedules, but not all of them chose the same slot. |
| occupied slot not freed | 38 | A schedule or reschedule target was already occupied and was not freed by another action in the same batch. |
| prior-meeting inconsistency | 25 | A reschedule of a previously scheduled meeting would have left its participants inconsistent. |

## Causal Decision Failures

| validator reason | all causal agents | selected model seats | description |
| --- | ---: | ---: | --- |
| empty/no schedule action | 36 | 21 | The final rejected decision batch contained no valid schedule action for this meeting. This can come from empty actions, malformed/truncated JSON that parsed to no actions, or a retry that dropped the schedule action. |
| tried to move blocked item | 32 | 19 | The batch attempted to reschedule a blocked item, which is illegal. |
| scheduled occupied slot | 28 | 10 | The batch scheduled the meeting into an occupied slot that it did not free. |

## Suggested Additional Basic Plots

- Scheduled-only excess burden by difficulty and model.
- Privacy leakage / VPS loss by model and difficulty.
- Message channel mix by model (`dm`, participant groupchat, all-groupchat).
- Failure rate by meeting index, to see whether later rounds compound earlier mistakes.
- Fairness cost / relative excess burden by model.