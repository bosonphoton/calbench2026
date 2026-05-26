from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_reflection_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "scripts"
        / "reflection_logprob_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("reflection_logprob_calibration", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_floor_filled_expected_delta_uses_top_logprob_floor_for_missing_values():
    calibration = _load_reflection_module()
    slot = calibration.SlotLogprobs(values={1: -0.2, 0: -1.2, 3: -3.2}, floor=-3.2)

    expected = calibration._floor_filled_expected(slot)

    assert expected == pytest.approx(0.5877, abs=1e-4)


def test_floor_upper_bound_counts_missing_directional_alternatives():
    calibration = _load_reflection_module()
    slot = calibration.SlotLogprobs(values={0: 0.0}, floor=0.0)

    upper = calibration._floor_upper_abs_expected(slot)

    assert upper == pytest.approx(1.5)


def test_reflection_rows_parse_trace_logprobs_and_actual_busy(tmp_path):
    calibration = _load_reflection_module()
    trace = {
        "game_id": "g1",
        "final_state": {
            "calendars": [
                [None, {"errand_id": 1}],
                [{"errand_id": 2}, None],
            ],
        },
        "events": [{
            "type": "reflection_end",
            "data": {
                "agent_id": 0,
                "target_agent_id": 1,
                "num_slots": 2,
                "text": "[0,1]",
                "raw_api_response": {
                    "logprobs": [{
                        "content": [
                            {"token": "[", "logprob": 0.0, "top_logprobs": []},
                            {"token": "0", "logprob": -0.1, "top_logprobs": [{"token": "0", "logprob": -0.1}]},
                            {"token": ",", "logprob": 0.0, "top_logprobs": []},
                            {
                                "token": "1",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": "1", "logprob": -0.2},
                                    {"token": "0", "logprob": -1.2},
                                ],
                            },
                            {"token": "]", "logprob": 0.0, "top_logprobs": []},
                        ],
                    }],
                },
            },
        }],
    }

    rows = calibration._reflection_rows(trace, tmp_path / "trace.json")

    assert len(rows) == 2
    assert [row["actual_busy"] for row in rows] == [1, 0]
    assert rows[0]["sampled_vps_loss"] == 0.0
    assert rows[1]["sampled_vps_loss"] == pytest.approx(1 / 6)
    assert rows[1]["floor_expected_delta"] > 0
