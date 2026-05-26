"""Calendar scheduling benchmark game."""

try:
    from calendar_game.game import CalendarGame, CalendarGameConfig
    __all__ = ["CalendarGame", "CalendarGameConfig"]
except ImportError:
    pass
