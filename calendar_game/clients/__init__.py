"""Client implementations for the calendar game."""

from calendar_game.clients.scripted import ScriptedClient
from calendar_game.clients.llm import LLMClient
from calendar_game.clients.dsm import DSMClient, PaperDSMClient, PrivateDSMClient
from calendar_game.clients.imap import IncrementalMAPClient
from calendar_game.clients.sd import SDClient
from calendar_game.clients.dspy import DSPyClient

__all__ = [
    "ScriptedClient",
    "LLMClient",
    "DSMClient",
    "PaperDSMClient",
    "PrivateDSMClient",
    "IncrementalMAPClient",
    "SDClient",
    "DSPyClient",
]
