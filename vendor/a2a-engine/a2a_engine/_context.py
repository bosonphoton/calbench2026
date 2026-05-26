"""Per-run context vars used to thread identifiers into nested spans."""

from contextvars import ContextVar

current_conversation_id: ContextVar[str | None] = ContextVar(
    "a2a_current_conversation_id", default=None
)
