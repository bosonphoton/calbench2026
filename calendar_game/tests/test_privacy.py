"""Tests for LLM-only privacy label hydration."""

from __future__ import annotations

import ast
import re

from calendar_game.privacy import (
    _balanced_errand_tiers,
    hydrate_calendar_render_for_llm,
    hydrate_meeting_for_llm,
    load_label_bank,
)


def test_hydrate_meeting_for_llm_adds_only_private_label():
    meeting = {"id": 1, "participants": [0, 1], "duration": 1, "cost": 1}

    hydrated = hydrate_meeting_for_llm(meeting, stable_key="test")

    assert hydrated["id"] == meeting["id"]
    assert hydrated["private_label"]
    assert "public_label" not in hydrated
    assert "allowed_public_terms" not in hydrated
    assert "forbidden_terms" not in hydrated
    assert "private_label" not in meeting


def test_hydrate_calendar_render_for_llm_preserves_slot_ids_and_adds_labels():
    render = "\n".join([
        "Slot 0: [FREE]",
        "Slot 1: Errand #7 (cost=3)",
        "Slot 2: Meeting M5 (cost=1) participants=[0, 2]",
    ])

    hydrated = hydrate_calendar_render_for_llm(render, stable_key="test")

    assert "Slot 0: [FREE]" in hydrated
    assert "Errand #7 (cost=3)" in hydrated
    assert "Meeting M5 (cost=1)" in hydrated
    assert "private_label=" in hydrated
    assert "safe_public_label=" not in hydrated
    assert "do_not_reveal=" not in hydrated


def test_balanced_errand_tier_plan_for_full_calendar():
    tiers = _balanced_errand_tiers(16, stable_key="agent:0")
    counts = {tier: tiers.count(tier) for tier in set(tiers)}

    assert counts["sensitive"] in (5, 6)
    assert counts["very_sensitive"] in (5, 6)
    assert counts["public"] in (5, 6)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_hydrate_calendar_assigns_fair_errand_privacy_mix():
    render = "\n".join(
        f"Slot {slot}: Errand #{slot + 1} (cost={(slot % 3) + 1})"
        for slot in range(16)
    )
    hydrated = hydrate_calendar_render_for_llm(render, stable_key="agent:0")
    private_labels = [
        ast.literal_eval(match.group(1))
        for match in re.finditer(r"private_label=(.*)$", hydrated, flags=re.MULTILINE)
    ]
    labels = load_label_bank("tasks/label_banks/errand_bank_v1.json", expected_bank_type="errand")
    tier_by_private_label = {
        label["private_label"]: label["privacy_tier"]
        for label in labels
    }
    counts: dict[str, int] = {}
    for private_label in private_labels:
        tier = tier_by_private_label[private_label]
        counts[tier] = counts.get(tier, 0) + 1

    assert len(private_labels) == 16
    assert len(set(private_labels)) == 16
    assert counts["sensitive"] in (5, 6)
    assert counts["very_sensitive"] in (5, 6)
    assert counts["public"] in (5, 6)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_full_five_agent_game_has_identical_mix_and_no_duplicate_errand_labels():
    labels = load_label_bank("tasks/label_banks/errand_bank_v1.json", expected_bank_type="errand")
    tier_by_private_label = {
        label["private_label"]: label["privacy_tier"]
        for label in labels
    }

    all_private_labels: list[str] = []
    per_agent_counts: list[dict[str, int]] = []
    for agent_id in range(5):
        render = "\n".join(
            f"Slot {slot}: Errand #{(agent_id * 16) + slot + 1} (cost={(slot % 3) + 1})"
            for slot in range(16)
        )
        hydrated = hydrate_calendar_render_for_llm(render, stable_key=f"agent:{agent_id}:round:0")
        private_labels = [
            ast.literal_eval(match.group(1))
            for match in re.finditer(r"private_label=(.*)$", hydrated, flags=re.MULTILINE)
        ]
        counts: dict[str, int] = {}
        for private_label in private_labels:
            tier = tier_by_private_label[private_label]
            counts[tier] = counts.get(tier, 0) + 1
        all_private_labels.extend(private_labels)
        per_agent_counts.append(counts)

    assert len(all_private_labels) == 80
    assert len(set(all_private_labels)) == 80
    assert all(counts == per_agent_counts[0] for counts in per_agent_counts)
    assert per_agent_counts[0] == {"sensitive": 6, "very_sensitive": 5, "public": 5}
