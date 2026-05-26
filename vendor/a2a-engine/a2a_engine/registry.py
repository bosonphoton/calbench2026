"""Game-class registry. Downstream benchmarks register their game class by name.

A "game" is any callable/class with the contract:

    class MyGame:
        def __init__(self, config: dict, dry_run: bool = False) -> None: ...
        def run(self) -> GameTraceBase: ...

The runner looks up games by ``game_name`` (the field on GameConfigBase).
"""

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_game(name: str, cls: Callable[..., Any]) -> None:
    """Register a game class under a name. Replaces any existing entry."""
    _REGISTRY[name] = cls


def get_game(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Game {name!r} not registered. Known: {sorted(_REGISTRY)}. "
            "Import the package that calls register_game() before running."
        )
    return _REGISTRY[name]


def list_games() -> list[str]:
    return sorted(_REGISTRY)
