"""RQ2: evolve DSPy negotiation prompt variants for calendar-game welfare.

This script runs a budgeted GEPA-style search over *appended negotiation policy*
text. It does not edit the base system prompt. Each candidate policy is written
as a prompt variant, evaluated by running calendar games on selected seeds, and
recorded in a JSONL leaderboard.

The optimization target is lower realized scheduling cost while preserving the
existing JSON protocol and privacy constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Any


CALENDAR_ROOT = Path(__file__).resolve().parents[2]
PROMPT_VARIANTS_DIR = CALENDAR_ROOT / "calendar_game" / "prompt_variants"
EXPERIMENTS_DIR = CALENDAR_ROOT / "experiments"
RESULTS_DIR = CALENDAR_ROOT / "results"
OUTPUTS_DIR = CALENDAR_ROOT / "analysis" / "outputs" / "rq2_gepa_negotiation_policy_search"

DEFAULT_HARD_CASE_BASELINES: dict[int, dict[str, float]] = {
    20260514: {"optimal": 8, "llm": 16, "dsm": 12, "v2": 9, "v4": 16, "v5": 15},
    20260515: {"optimal": 9, "llm": 18, "dsm": 16, "v2": 19, "v4": 18, "v5": 20},
    20260516: {"optimal": 16, "llm": 20, "dsm": 18, "v2": 19, "v4": 18, "v5": 18},
    20260535: {"optimal": 7, "llm": 14, "dsm": 13, "v2": 23, "v4": 14, "v5": 17},
    20260539: {"optimal": 4, "llm": 9, "dsm": 5, "v2": 8, "v4": 5, "v5": 4},
}

SEED_POLICIES = [
    "dspy_05042045_v2.md",
    "dspy_05042132_v4.md",
    "dspy_05042142_v5.md",
]


def _now_id() -> str:
    return time.strftime("%m%d%H%M%S")


def _read_seed_policies() -> str:
    chunks = []
    for name in SEED_POLICIES:
        path = PROMPT_VARIANTS_DIR / name
        if path.exists():
            chunks.append(f"--- {name} ---\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(chunks)


def _policy_hash(policy: str) -> str:
    return hashlib.sha1(policy.encode("utf-8")).hexdigest()[:10]


def _clean_policy(raw: str) -> str:
    policy = raw.strip()
    if policy.startswith("```"):
        lines = policy.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        policy = "\n".join(lines).strip()
    if "Do NOT include any text outside the JSON object." not in policy:
        policy += "\n\nDo NOT include any text outside the JSON object."
    return policy + "\n"


def _write_experiment(
    *,
    experiment_name: str,
    prompt_variant: str,
    seeds: list[int],
    max_turns_per_round: int,
    dm_cap: int,
) -> Path:
    agents = "\n".join(
        [
            "    - type: dspy\n"
            "      model: gemini-3-flash-preview\n"
            f"      prompt_variant: {prompt_variant}"
            for _ in range(3)
        ]
    )
    batches = "\n".join(
        [
            f"  - label: seed_{seed}\n"
            "    count: 1\n"
            f"    config: {{seed: {seed}}}"
            for seed in seeds
        ]
    )
    text = f"""name: {experiment_name}
defaults:
  game_name: calendar
  num_agents: 3
  num_slots: 16
  density: 1.0
  pref_level: 3
  num_meetings: 6
  num_participants: 2
  max_turns_per_round: {max_turns_per_round}
  dm_cap: {dm_cap}
  decision_retries: 2
  enable_fallback: false
  agents:
{agents}
batches:
{batches}
"""
    path = EXPERIMENTS_DIR / f"{experiment_name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _run_experiment(experiment_path: Path, *, max_parallelism: int) -> None:
    cmd = [
        sys.executable,
        "run.py",
        str(experiment_path.relative_to(CALENDAR_ROOT)),
        "--max-parallelism",
        str(max_parallelism),
    ]
    subprocess.run(cmd, cwd=CALENDAR_ROOT, check=True)


def _load_result_rows(experiment_name: str) -> list[dict[str, Any]]:
    result_dir = RESULTS_DIR / experiment_name
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name.endswith(".metadata.json"):
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": int(trace["config"]["seed"]),
                "trace_path": str(path.relative_to(CALENDAR_ROOT)),
                "metrics": trace.get("metrics", {}),
            }
        )
    return rows


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    baselines: dict[int, dict[str, float]],
) -> tuple[float, dict[str, Any], str]:
    costs = {row["seed"]: float(row["metrics"]["realized_cost"]) for row in rows}
    seeds = sorted(costs)
    total = sum(costs.values())
    optimal_total = sum(baselines[seed]["optimal"] for seed in seeds)
    llm_total = sum(baselines[seed]["llm"] for seed in seeds)
    dsm_total = sum(baselines[seed]["dsm"] for seed in seeds)
    denom = max(llm_total - optimal_total, 1.0)
    score = max(0.0, min(1.0, (llm_total - total) / denom))

    per_seed = []
    feedback_lines = [
        f"Total realized cost {total:.0f}; lower is better.",
        f"Reference totals on these seeds: optimal={optimal_total:.0f}, DSM={dsm_total:.0f}, baseline Gemini={llm_total:.0f}.",
    ]
    for seed in seeds:
        b = baselines[seed]
        cost = costs[seed]
        per_seed.append(
            {
                "seed": seed,
                "realized_cost": cost,
                "optimal_cost": b["optimal"],
                "llm_cost": b["llm"],
                "dsm_cost": b["dsm"],
                "delta_vs_llm": cost - b["llm"],
                "delta_vs_dsm": cost - b["dsm"],
            }
        )
        if cost <= b["dsm"]:
            feedback_lines.append(f"Seed {seed}: strong result, cost {cost:.0f} meets/beats DSM {b['dsm']:.0f}.")
        elif cost < b["llm"]:
            feedback_lines.append(f"Seed {seed}: partial improvement, cost {cost:.0f} beats Gemini {b['llm']:.0f} but trails DSM {b['dsm']:.0f}.")
        else:
            feedback_lines.append(f"Seed {seed}: weak result, cost {cost:.0f} does not improve over Gemini {b['llm']:.0f}.")

    summary = {
        "score": score,
        "total_realized_cost": total,
        "total_optimal_cost": optimal_total,
        "total_llm_cost": llm_total,
        "total_dsm_cost": dsm_total,
        "per_seed": per_seed,
    }
    return score, summary, "\n".join(feedback_lines)


def _build_context(seeds: list[int]) -> str:
    baseline_lines = []
    for seed in seeds:
        b = DEFAULT_HARD_CASE_BASELINES[seed]
        baseline_lines.append(
            f"- {seed}: optimal={b['optimal']}, baseline Gemini={b['llm']}, DSM={b['dsm']}, "
            f"v2={b.get('v2')}, v4={b.get('v4')}, v5={b.get('v5')}"
        )
    return f"""
We are optimizing ONLY an appended NEGOTIATION POLICY for LLM agents in a dense calendar scheduling game.

Non-negotiable constraints:
- Preserve the established JSON/action protocol.
- Do not change game rules, calendar rendering, hidden metadata, or privacy-label hydration.
- Do not reveal exact costs, private event details, private labels, hidden labels, seeds, DSM results, or experiment names to the agents.
- Agents may use qualitative language about flexibility, scarcity, bottlenecks, and low/high disruption.
- Keep the policy concise enough for agents to act within max_turns_per_round=6.

Task setting:
- 3 agents, 2 participants per meeting, 6 meetings, 16 slots, density=1.0.
- Lower realized cost is better.

Training/evaluation seeds and known reference costs:
{chr(10).join(baseline_lines)}

Prior candidate policies:
{_read_seed_policies()}

Generate a stronger appended negotiation policy. It should be general, not seed-specific, and should fit naturally after the base system prompt.
""".strip()


def run_candidate(
    *,
    policy: str,
    run_id: str,
    candidate_index: int,
    seeds: list[int],
    max_parallelism: int,
    max_turns_per_round: int,
    dm_cap: int,
    out_dir: Path,
) -> tuple[float, dict[str, Any], str]:
    cleaned = _clean_policy(policy)
    digest = _policy_hash(cleaned)
    variant_name = f"gepa_{run_id}_c{candidate_index:03d}_{digest}.md"
    experiment_name = f"gepa_{run_id}_c{candidate_index:03d}_{digest}"

    variant_path = PROMPT_VARIANTS_DIR / variant_name
    variant_path.write_text(cleaned, encoding="utf-8")
    experiment_path = _write_experiment(
        experiment_name=experiment_name,
        prompt_variant=variant_name,
        seeds=seeds,
        max_turns_per_round=max_turns_per_round,
        dm_cap=dm_cap,
    )
    _run_experiment(experiment_path, max_parallelism=max_parallelism)
    rows = _load_result_rows(experiment_name)
    score, summary, feedback = _score_rows(rows, baselines=DEFAULT_HARD_CASE_BASELINES)

    record = {
        "run_id": run_id,
        "candidate_index": candidate_index,
        "variant_name": variant_name,
        "experiment_name": experiment_name,
        "prompt_variant_path": str(variant_path.relative_to(CALENDAR_ROOT)),
        "experiment_path": str(experiment_path.relative_to(CALENDAR_ROOT)),
        "policy_sha1": digest,
        "policy": cleaned,
        **summary,
    }
    with (out_dir / "leaderboard.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return score, record, feedback


class PolicyProgram:
    """Lazy DSPy module wrapper to avoid importing DSPy for --dry-run."""

    def __init__(self) -> None:
        import dspy

        class GeneratePolicy(dspy.Signature):
            """Generate a concise appended negotiation policy for calendar-game agents."""

            context = dspy.InputField()
            policy = dspy.OutputField()

        class Module(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self.generate_policy = dspy.Predict(GeneratePolicy)

            def forward(self, context: str):
                return self.generate_policy(context=context)

        self.module = Module()


class ReflectivePolicyProgram:
    """DSPy proposer for the explicit reflective prompt-evolution loop."""

    def __init__(self) -> None:
        import dspy

        class GeneratePolicy(dspy.Signature):
            """Generate one improved appended negotiation policy from prior evaluation feedback."""

            context = dspy.InputField()
            search_state = dspy.InputField()
            policy = dspy.OutputField()

        class Module(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self.generate_policy = dspy.Predict(GeneratePolicy)

            def forward(self, context: str, search_state: str):
                return self.generate_policy(context=context, search_state=search_state)

        self.module = Module()


def _summarize_record(record: dict[str, Any]) -> str:
    per_seed = ", ".join(
        f"{row['seed']}={row['realized_cost']:.0f}"
        for row in record.get("per_seed", [])
    )
    policy = record.get("policy", "").strip().splitlines()
    headline = policy[0] if policy else ""
    return (
        f"- c{record['candidate_index']:03d} {record['policy_sha1']}: "
        f"score={record['score']:.3f}, total={record['total_realized_cost']:.0f}, "
        f"per_seed=[{per_seed}], headline={headline!r}"
    )


def _build_search_state(records: list[dict[str, Any]], *, max_records: int = 8) -> str:
    if not records:
        return """
No candidates have been evaluated yet.

Generate a first candidate that synthesizes the strongest ideas from prior variants:
- v2's assertive bottleneck protection and willingness to push back
- v4's global coordination and slot-scarcity language
- v5's fast convergence and explicit unblocking behavior

Avoid seed-specific wording. Keep the policy concise, actionable, and compatible with the JSON-only action protocol.
""".strip()

    best = min(records, key=lambda r: (r["total_realized_cost"], -r["score"]))
    recent = records[-max_records:]
    return f"""
Best candidate so far:
{_summarize_record(best)}

Recent candidates:
{chr(10).join(_summarize_record(record) for record in recent)}

Reflect on the failures by seed. Lower total cost is better.
Write a NEW appended negotiation policy, not a duplicate of any prior policy.
Preserve what helped, but mutate the policy in a concrete way: change the decision rules, priority ordering, or negotiation tactics.
Do not mention scores, seeds, DSM, experiments, hidden metadata, exact costs, or private details to agents.
""".strip()


def run_gepa(args: argparse.Namespace) -> None:
    import dspy
    from dspy.teleprompt import GEPA

    run_id = args.run_id or _now_id()
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    context = _build_context(seeds)
    (out_dir / "context.txt").write_text(context, encoding="utf-8")

    lm = dspy.LM(
        args.model,
        max_tokens=args.proposer_max_tokens,
        temperature=args.temperature,
        vertex_location=args.vertex_location,
        vertex_project=args.vertex_project,
    )
    dspy.configure(lm=lm)

    candidate_counter = {"value": 0}
    policy_cache: dict[str, tuple[float, dict[str, Any], str]] = {}

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        cleaned = _clean_policy(pred.policy)
        digest = _policy_hash(cleaned)
        if digest in policy_cache:
            score, record, feedback = policy_cache[digest]
            feedback = (
                "Duplicate policy proposal; reused the existing evaluated score "
                f"for {record['variant_name']} without rerunning games.\n"
                + feedback
            )
            return dspy.Prediction(score=score, feedback=feedback + f"\nVariant: {record['variant_name']}")

        candidate_counter["value"] += 1
        score, record, feedback = run_candidate(
            policy=cleaned,
            run_id=run_id,
            candidate_index=candidate_counter["value"],
            seeds=seeds,
            max_parallelism=args.max_parallelism,
            max_turns_per_round=args.max_turns_per_round,
            dm_cap=args.dm_cap,
            out_dir=out_dir,
        )
        policy_cache[digest] = (score, record, feedback)
        return dspy.Prediction(score=score, feedback=feedback + f"\nVariant: {record['variant_name']}")

    student = PolicyProgram().module
    trainset = [dspy.Example(context=context).with_inputs("context")]
    optimizer = GEPA(
        metric=metric,
        max_full_evals=args.candidates,
        reflection_lm=lm,
        reflection_minibatch_size=1,
        candidate_selection_strategy="current_best",
        log_dir=str(out_dir / "gepa_logs"),
        track_stats=True,
        track_best_outputs=True,
        seed=args.seed,
    )
    optimizer.compile(student, trainset=trainset)
    print(f"GEPA search complete. Leaderboard: {out_dir / 'leaderboard.jsonl'}")


def run_reflective_gepa(args: argparse.Namespace) -> None:
    import dspy

    run_id = args.run_id or _now_id()
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _build_context(args.seeds)
    (out_dir / "context.txt").write_text(context, encoding="utf-8")

    lm = dspy.LM(
        args.model,
        max_tokens=args.proposer_max_tokens,
        temperature=args.temperature,
        vertex_location=args.vertex_location,
        vertex_project=args.vertex_project,
    )
    dspy.configure(lm=lm)
    program = ReflectivePolicyProgram().module

    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for idx in range(1, args.candidates + 1):
        search_state = _build_search_state(records)
        (out_dir / f"search_state_c{idx:03d}.txt").write_text(search_state, encoding="utf-8")
        pred = program(context=context, search_state=search_state)
        cleaned = _clean_policy(pred.policy)
        digest = _policy_hash(cleaned)
        if digest in seen_hashes:
            search_state += (
                "\n\nThe previous proposal was a duplicate. Produce a substantially different policy "
                "with different concrete negotiation rules."
            )
            pred = program(context=context, search_state=search_state)
            cleaned = _clean_policy(pred.policy)
            digest = _policy_hash(cleaned)
        seen_hashes.add(digest)

        score, record, feedback = run_candidate(
            policy=cleaned,
            run_id=run_id,
            candidate_index=idx,
            seeds=args.seeds,
            max_parallelism=args.max_parallelism,
            max_turns_per_round=args.max_turns_per_round,
            dm_cap=args.dm_cap,
            out_dir=out_dir,
        )
        records.append(record)
        print(f"candidate {idx}: score={score:.3f} cost={record['total_realized_cost']:.0f} hash={record['policy_sha1']}")
        print(textwrap.indent(feedback, "  "))

    best = min(records, key=lambda r: (r["total_realized_cost"], -r["score"])) if records else None
    if best:
        print(f"Best candidate: c{best['candidate_index']:03d} {best['variant_name']} cost={best['total_realized_cost']:.0f} score={best['score']:.3f}")
    print(f"Reflective GEPA search complete. Leaderboard: {out_dir / 'leaderboard.jsonl'}")


def run_single_proposal(args: argparse.Namespace) -> None:
    import dspy

    run_id = args.run_id or _now_id()
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    context = _build_context(args.seeds)
    (out_dir / "context.txt").write_text(context, encoding="utf-8")

    lm = dspy.LM(
        args.model,
        max_tokens=args.proposer_max_tokens,
        temperature=args.temperature,
        vertex_location=args.vertex_location,
        vertex_project=args.vertex_project,
    )
    dspy.configure(lm=lm)
    program = PolicyProgram().module
    for idx in range(1, args.candidates + 1):
        pred = program(context=context)
        score, record, feedback = run_candidate(
            policy=pred.policy,
            run_id=run_id,
            candidate_index=idx,
            seeds=args.seeds,
            max_parallelism=args.max_parallelism,
            max_turns_per_round=args.max_turns_per_round,
            dm_cap=args.dm_cap,
            out_dir=out_dir,
        )
        print(f"candidate {idx}: score={score:.3f} cost={record['total_realized_cost']:.0f}")
        print(textwrap.indent(feedback, "  "))
    print(f"Search complete. Leaderboard: {out_dir / 'leaderboard.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gepa", "reflective-gepa", "single-proposal"], default="gepa")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidates", type=int, default=10, help="Approximate number of candidate prompt evaluations.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260514, 20260516, 20260539])
    parser.add_argument("--max-parallelism", type=int, default=3)
    parser.add_argument("--max-turns-per-round", type=int, default=6)
    parser.add_argument("--dm-cap", type=int, default=100)
    parser.add_argument("--model", default="vertex_ai/gemini-3-flash-preview")
    parser.add_argument("--vertex-location", default="global")
    parser.add_argument("--vertex-project", default="YOUR_VERTEX_PROJECT")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--proposer-max-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.seeds) - set(DEFAULT_HARD_CASE_BASELINES))
    if unknown:
        raise SystemExit(f"No built-in baselines for seeds: {unknown}")
    if args.mode == "gepa":
        run_gepa(args)
    elif args.mode == "reflective-gepa":
        run_reflective_gepa(args)
    else:
        run_single_proposal(args)


if __name__ == "__main__":
    main()
