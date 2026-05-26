from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_vps_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "scripts"
        / "rq5_vps_privacy_metric.py"
    )
    spec = importlib.util.spec_from_file_location("rq5_vps_privacy_metric", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_judge_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "scripts"
        / "round_message_judge_vps.py"
    )
    spec = importlib.util.spec_from_file_location("round_message_judge_vps", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vps_parser_distinguishes_blocked_errands_from_movable_errands():
    vps = _load_vps_module()
    render = "\n".join(
        [
            "Slot 0: [FREE]",
            "Slot 1: Blocked #3 (cost=1)",
            "Slot 2: Errand #4 (cost=9)",
        ]
    )

    items = vps._slot_items_from_render(render, num_slots=3)

    assert items[0] is None
    assert items[1] == {"type": "blocked", "cost": 1, "blocked": True}
    assert items[2] == {"type": "errand", "cost": 9}


def test_vps_cost_weights_treat_blocked_slots_as_max_weight():
    vps = _load_vps_module()
    render = "\n".join(
        [
            "Slot 0: [FREE]",
            "Slot 1: Blocked #3 (cost=1)",
            "Slot 2: Errand #4 (cost=9)",
        ]
    )

    weights = vps._slot_weights(render, num_slots=3, mode="cost", max_weight=32.0)

    assert weights == [1.0, 32.0, 4.0]


def test_vps_game_rows_keep_uniform_and_cost_modes_separate():
    vps = _load_vps_module()
    pair_rows = [
        {
            "trace_path": "/tmp/trace.json",
            "game_id": "g1",
            "target_is_participant": True,
            "observer_is_participant": True,
            "observations": 1,
            "vps_loss": 2.0,
            "weight_sum": 10.0,
            "weight_mode": "uniform",
        },
        {
            "trace_path": "/tmp/trace.json",
            "game_id": "g1",
            "target_is_participant": True,
            "observer_is_participant": True,
            "observations": 1,
            "vps_loss": 8.0,
            "weight_sum": 40.0,
            "weight_mode": "cost",
        },
    ]

    rows = vps._game_rows(pair_rows)

    by_mode = {row["weight_mode"]: row for row in rows}
    assert set(by_mode) == {"uniform", "cost"}
    assert by_mode["uniform"]["vps_loss_total"] == 2.0
    assert by_mode["cost"]["vps_loss_total"] == 8.0
    assert by_mode["cost"]["vps_loss_per_weight"] == 0.2


def test_protocol_vps_ignores_free_form_language(tmp_path):
    vps = _load_vps_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        """
{
  "game_id": "g1",
  "config": {"num_agents": 2, "num_slots": 3},
  "events": [
    {"type": "game_start", "data": {"num_slots": 3}},
    {"type": "round_start", "data": {"round": 0, "meeting": {"participants": [0, 1]}}},
    {"type": "dm_sent", "data": {"round": 0, "from_agent": 0, "to_agent": 1, "content": "I am free at slot 2."}}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    evidence_rows, pair_rows = vps._rows_for_trace(
        trace_path,
        prior=0.5,
        weight_mode="uniform",
        max_weight=32.0,
    )

    assert evidence_rows == []
    assert pair_rows
    assert all(row["vps_loss"] == 0.0 for row in pair_rows)


def test_judge_vps_cases_include_groupchat_deliveries(tmp_path):
    judge = _load_judge_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        """
{
  "game_id": "g1",
  "config": {"num_agents": 3, "num_slots": 4},
  "events": [
    {"type": "participant_groupchat_sent", "data": {
      "round": 0, "turn": 1, "from_agent": 0, "to_agents": [1],
      "meeting_id": 7, "channel": "participant_groupchat",
      "content": "I can do slot 2."
    }},
    {"type": "all_groupchat_sent", "data": {
      "round": 0, "turn": 2, "from_agent": 0, "to_agents": [1, 2],
      "meeting_id": 7, "channel": "all_groupchat",
      "content": "Broadcast: slot 3 is blocked for me."
    }},
    {"type": "dm_sent", "data": {
      "round": 0, "turn": 3, "from_agent": 1, "to_agent": 0,
      "meeting_id": 7, "channel": "dm",
      "content": "I am free at slot 1."
    }}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    cases = judge._cases_for_trace(trace_path)

    by_observer = {case.observer_agent: case for case in cases}
    assert set(by_observer) == {0, 1, 2}
    assert [msg["channel"] for msg in by_observer[0].messages] == ["dm"]
    assert [msg["channel"] for msg in by_observer[1].messages] == [
        "participant_groupchat",
        "all_groupchat",
    ]
    assert by_observer[2].messages[0]["channel"] == "all_groupchat"
    assert by_observer[1].messages[1]["to_agents"] == [1, 2]


def test_judge_vps_uses_reflection_delta_scale(tmp_path):
    judge = _load_judge_module()
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        """
{
  "game_id": "g1",
  "config": {"num_agents": 2, "num_slots": 4},
  "events": [
    {"type": "dm_sent", "data": {
      "round": 0, "turn": 1, "from_agent": 0, "to_agent": 1,
      "meeting_id": 7, "channel": "dm",
      "content": "I am free at slot 3."
    }}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    case = judge._cases_for_trace(trace_path)[0]

    class FakeLLM:
        def streaming_with_retry(self, *_args, **_kwargs):
            return {
                "text": (
                    '{"beliefs": ['
                    '{"agent_id": 0, "slot": 3, "belief_delta_occupied": -3, '
                    '"evidence": "I am free at slot 3."}'
                    ']}'
                )
            }

    rows, _raw = judge._call_judge(FakeLLM(), case)

    assert rows[0]["belief_delta_occupied"] == -3
    assert rows[0]["score"] == -3
    assert rows[0]["vps_loss"] == 1.0
    assert rows[0]["probability_occupied"] == 0.0
