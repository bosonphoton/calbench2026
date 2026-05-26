"""Privacy label hydration for LLM-facing calendar context.

The base scenario/task format intentionally stays numeric and simple. This
module is an interdiction layer: it decorates the text sent to LLM clients with
private event descriptions drawn from label banks, without changing solver
inputs, DSM inputs, or checked-in task fixtures. Non-description fields such as
public labels and forbidden terms stay in the bank for post-hoc analysis, not in
the agent prompt.
"""

from __future__ import annotations

from copy import deepcopy
import json
import random
import re
from pathlib import Path
from typing import Any

DEFAULT_MEETING_LABEL_BANK = "tasks/label_banks/meeting_bank_v1.json"
DEFAULT_ERRAND_LABEL_BANK = "tasks/label_banks/errand_bank_v1.json"
ERRAND_PRIVACY_TIERS = ("sensitive", "very_sensitive", "public")

_LABEL_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _calendar_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    package_candidate = _calendar_root() / candidate
    if package_candidate.exists():
        return package_candidate
    return Path.cwd() / candidate


def load_label_bank(path: str | Path, *, expected_bank_type: str) -> list[dict[str, Any]]:
    bank_path = _resolve_path(path)
    cache_key = (str(bank_path), expected_bank_type)
    if cache_key in _LABEL_CACHE:
        return _LABEL_CACHE[cache_key]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("bank_type") != expected_bank_type:
        raise ValueError(
            f"{bank_path} has bank_type={bank.get('bank_type')!r}; expected {expected_bank_type!r}"
        )
    items = bank.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{bank_path} does not contain label items")
    _LABEL_CACHE[cache_key] = items
    return items


def _score_for_label(label: dict[str, Any]) -> int | None:
    for key in ("score", "priority_score", "cost_score", "semantic_priority"):
        value = label.get(key)
        if isinstance(value, int):
            return value
    return None


def _choose_label(labels: list[dict[str, Any]], *, score: int, stable_key: str) -> dict[str, Any]:
    score_pool = [label for label in labels if _score_for_label(label) == score]
    pool = score_pool or labels
    rng = random.Random(stable_key)
    return pool[rng.randrange(len(pool))]


def _choose_errand_label(
    labels: list[dict[str, Any]],
    *,
    tier: str,
    score: int,
    stable_key: str,
    label_index: int,
    used_label_ids: set[str] | None = None,
) -> dict[str, Any]:
    tier_pool = [label for label in labels if label.get("privacy_tier") == tier]
    pool = tier_pool or labels
    # Keep the score in the signature for compatibility with earlier callers,
    # but prioritize cross-agent label uniqueness within each privacy tier.
    _ = score
    shuffled = list(pool)
    random.Random(f"errand-label-pool:{tier}:{stable_key}").shuffle(shuffled)
    label = shuffled[label_index % len(shuffled)]
    if used_label_ids is not None:
        used_label_ids.add(str(label.get("label_id")))
    return label


def _agent_id_from_errand_id(errand_id: int, *, num_slots: int = 16) -> int:
    return max(0, (errand_id - 1) // num_slots)


def _local_errand_index(errand_id: int, *, num_slots: int = 16) -> int:
    return max(0, (errand_id - 1) % num_slots)


def _tier_rank_before(tiers: list[str], *, index: int, tier: str) -> int:
    return sum(1 for prior in tiers[:index] if prior == tier)


def _global_errand_label_index(errand_id: int, tiers: list[str], tier: str, *, errand_position: int) -> int:
    """Return a cross-agent stable index within a tier pool.

    Generated calendar errands use globally unique 1-based ids in blocks of 16
    per agent. The errand_position is the rendered order within this agent's
    current calendar, which keeps labels unique even when errand ids are shuffled
    across slots.
    """
    per_agent_tier_count = tiers.count(tier)
    return (_agent_id_from_errand_id(errand_id) * per_agent_tier_count) + _tier_rank_before(
        tiers,
        index=errand_position,
        tier=tier,
    )


def _balanced_errand_tiers(count: int, *, stable_key: str) -> list[str]:
    """Return a deterministic, equal-count privacy-tier plan for an agent calendar."""
    _ = stable_key  # kept for API stability; equal-count agents should share the same plan.
    return [ERRAND_PRIVACY_TIERS[index % len(ERRAND_PRIVACY_TIERS)] for index in range(count)]


def _label_fields(label: dict[str, Any]) -> dict[str, Any]:
    fields = {"label_id", "private_label"}
    return {key: deepcopy(value) for key, value in label.items() if key in fields}


def hydrate_meeting_for_llm(meeting: dict[str, Any], *, stable_key: str) -> dict[str, Any]:
    """Return a meeting dict decorated with LLM-only privacy labels."""
    labels = load_label_bank(DEFAULT_MEETING_LABEL_BANK, expected_bank_type="meeting")
    score = int(meeting.get("cost", 1))
    label = _choose_label(labels, score=score, stable_key=f"meeting:{stable_key}:{meeting.get('id')}")
    hydrated = deepcopy(meeting)
    hydrated.update(_label_fields(label))
    return hydrated


def hydrate_calendar_render_for_llm(calendar_render: str, *, stable_key: str) -> str:
    """Decorate Calendar.render() output with labels for LLM context only."""
    errand_labels = load_label_bank(DEFAULT_ERRAND_LABEL_BANK, expected_bank_type="errand")
    meeting_labels = load_label_bank(DEFAULT_MEETING_LABEL_BANK, expected_bank_type="meeting")
    source_lines = calendar_render.splitlines()
    errand_matches = [
        (line_index, re.search(r"Errand #(\d+) \(cost=(\d+)\)", line))
        for line_index, line in enumerate(source_lines)
    ]
    errand_matches = [(line_index, match) for line_index, match in errand_matches if match is not None]
    errand_tiers = _balanced_errand_tiers(len(errand_matches), stable_key=stable_key)
    errand_tier_by_line = {
        line_index: errand_tiers[index]
        for index, (line_index, _match) in enumerate(errand_matches)
    }
    errand_position_by_line = {
        line_index: index
        for index, (line_index, _match) in enumerate(errand_matches)
    }
    used_errand_label_ids: set[str] = set()

    lines: list[str] = []
    for line_index, line in enumerate(source_lines):
        errand_match = re.search(r"Errand #(\d+) \(cost=(\d+)\)", line)
        meeting_match = re.search(r"Meeting M(\d+) \(cost=(\d+)\)", line)
        if errand_match:
            item_id = int(errand_match.group(1))
            score = int(errand_match.group(2))
            tier = errand_tier_by_line[line_index]
            label_index = _global_errand_label_index(
                item_id,
                errand_tiers,
                tier,
                errand_position=errand_position_by_line[line_index],
            )
            label = _choose_errand_label(
                errand_labels,
                tier=tier,
                score=score,
                stable_key="global",
                label_index=label_index,
                used_label_ids=used_errand_label_ids,
            )
            lines.append(f"{line} {_render_label_note(label)}")
        elif meeting_match:
            item_id = int(meeting_match.group(1))
            score = int(meeting_match.group(2))
            label = _choose_label(meeting_labels, score=score, stable_key=f"prior-meeting:{stable_key}:{item_id}")
            lines.append(f"{line} {_render_label_note(label)}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _render_label_note(label: dict[str, Any]) -> str:
    return f"private_label={label.get('private_label')!r}"
