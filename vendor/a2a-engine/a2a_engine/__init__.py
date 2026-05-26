"""a2a-engine: game-agnostic framework for agent-to-agent coordination experiments."""

from __future__ import annotations

from typing import Any


_EXPORTS = {
    "AgentInterface": ("a2a_engine.agent", "AgentInterface"),
    "LLMAgent": ("a2a_engine.agent", "LLMAgent"),
    "ExperimentSpec": ("a2a_engine.experiment", "ExperimentSpec"),
    "BatchSpec": ("a2a_engine.experiment", "BatchSpec"),
    "load_experiment": ("a2a_engine.experiment", "load_experiment"),
    "expand_batches": ("a2a_engine.experiment", "expand_batches"),
    "run_with_parallelism": ("a2a_engine.parallel", "run_with_parallelism"),
    "AgentInfo": ("a2a_engine.schemas", "AgentInfo"),
    "GameConfigBase": ("a2a_engine.schemas", "GameConfigBase"),
    "GameEvent": ("a2a_engine.schemas", "GameEvent"),
    "GameTraceBase": ("a2a_engine.schemas", "GameTraceBase"),
    "EventLog": ("a2a_engine.tracing", "EventLog"),
    "write_trace": ("a2a_engine.tracing", "write_trace"),
    "read_trace": ("a2a_engine.tracing", "read_trace"),
    "register_game": ("a2a_engine.registry", "register_game"),
    "get_game": ("a2a_engine.registry", "get_game"),
    "list_games": ("a2a_engine.registry", "list_games"),
    "GameDataset": ("a2a_engine.dataset", "GameDataset"),
    "GameMessage": ("a2a_engine.dataset", "GameMessage"),
    "GameRecord": ("a2a_engine.dataset", "GameRecord"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'a2a_engine' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
