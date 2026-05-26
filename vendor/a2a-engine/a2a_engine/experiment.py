"""YAML experiment loader with defaults + batch override merge semantics.

Schema:

    name: my_experiment
    description: ...
    defaults:
      game_name: calendar
      num_agents: 3
      ...                 # any GameConfig fields
    batches:
      - label: baseline
        count: 5
        config:           # overrides on top of defaults
          seed: 1
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BatchSpec(BaseModel):
    label: str
    count: int = 1
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    name: str
    description: str = ""
    defaults: dict[str, Any] = Field(default_factory=dict)
    batches: list[BatchSpec] = Field(default_factory=list)


def load_experiment(path: str | Path) -> ExperimentSpec:
    """Load and validate an experiment YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentSpec(**raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def expand_batches(spec: ExperimentSpec) -> list[tuple[BatchSpec, dict[str, Any]]]:
    """For each batch, return (batch, resolved_config) where resolved_config
    is the experiment defaults deep-merged with the batch's overrides."""
    out: list[tuple[BatchSpec, dict[str, Any]]] = []
    for batch in spec.batches:
        resolved = _deep_merge(spec.defaults, batch.config)
        out.append((batch, resolved))
    return out
