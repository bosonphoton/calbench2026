# Adding a New Client

A client drives one agent through the game loop. The game engine calls the client's methods at each phase; the client decides what tool calls to return. To add a new client, subclass `BaseClient` from `calendar_game.agents`.

## The Interface

```python
from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult

class MyClient(BaseClient):
    def register(self, agent_id: int, game_config: GameConfig) -> None:
        """Called once at game start. Store agent_id and any config you need."""

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        """Called at the start of each round the agent participates in.
        meeting: {"id": int, "participants": [int, ...], "duration": int, "cost": int}
        calendar_render: current calendar as a human-readable string
        """

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        """CHEAP_TALK phase. messages is the drained inbox in arrival order.
        Each message includes:
          {"from": int, "to": int | None, "channel": str, "meeting_id": int, "content": str}
        turn_index is zero-based; max_turns_per_round is the round's cheap-talk cap.
        Return TurnResult. tool_calls should be cheap-talk dicts or [] to pass.
        """

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        """DECISION phase. calendar_render is a frozen snapshot.
        Return DecideResult. tool_calls should be schedule/reschedule dicts.
        Every reschedule dict must include a non-empty justification string
        explaining why the user's existing commitment needs to move.
        """

    # Optional — default is a no-op pass
    def retry_decide(self, attempt: int, max_attempts: int, conflict: str) -> DecideResult:
        """Called when decide() returned an invalid batch. attempt is 1-indexed."""

    # Optional — default is a no-op pass
    def voluntary_decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        """VOLUNTARY phase. Called for non-participants activated by DM or all_groupchat.
        Return reschedule actions to honor coordination commitments, or [] to pass.
        Every reschedule dict must include a non-empty justification string.
        """
```

### Return types

```python
TurnResult(
    tool_calls=[{"type": "dm", "to": 1, "meeting_id": 3, "content": "use slot 5"}],
    text=None,       # raw model output text, or None
    thinking=None,   # reasoning trace, or None
    usage=None,      # TokenUsage dataclass, or None
    latency_ms=None, # wall-clock ms, or None
    raw=None,        # raw API response dict, or None
)

DecideResult(
    tool_calls=[
        {
            "type": "reschedule",
            "item_id": 7,
            "from_slot": 4,
            "to_slot": 9,
            "justification": "Free the agreed meeting slot after checking easier alternatives.",
        },
        {"type": "schedule", "meeting_id": 3, "slot": 4},
    ],
    ...,             # same optional fields as TurnResult
    retry_count=0,   # how many retries were used (first attempt = 0)
)
```

Return `tool_calls=[]` to pass without action at any phase.

Cheap-talk tool calls are controlled by `communication_protocol`:

| Tool | Required fields | Audience |
| --- | --- | --- |
| `dm` | `to`, `content` | One private recipient |
| `participant_groupchat` | `content` | Other participants in the current meeting |
| `all_groupchat` | `content` | Every other agent in the task |

`communication_protocol` may be a single channel, a list/set of channels, or an
alias. Common aliases are `groupchat` for `all_groupchat`,
`dm_and_groupchat` for `dm` plus `all_groupchat`, and `all` for all three
channels. Participants are active by default. Non-participants are activated
only after receiving a DM or `all_groupchat`; `participant_groupchat` remains
within the current meeting participants.

A CHEAP_TALK response may contain multiple communication actions. If the active
protocol enables both private DMs and groupchat, the client can return both in
the same `tool_calls` list and the game will deliver them in order.

## Minimal Example

```python
# calendar_game/clients/always_free.py
from calendar_game.agents import BaseClient, DecideResult, GameConfig, TurnResult

class AlwaysFreeClient(BaseClient):
    """Schedules into the first free slot, never DMs anyone."""

    def register(self, agent_id: int, game_config: GameConfig) -> None:
        self._agent_id = agent_id
        self._meeting = None
        self._free_slots = []

    def start_round(self, meeting: dict, calendar_render: str, round_num: int) -> None:
        self._meeting = meeting
        self._free_slots = [
            int(line.split("Slot")[1].split(":")[0].strip())
            for line in calendar_render.splitlines()
            if "[FREE]" in line
        ]

    def turn(
        self,
        messages: list[dict],
        turn_index: int | None = None,
        max_turns_per_round: int | None = None,
    ) -> TurnResult:
        return TurnResult(tool_calls=[], text=None, thinking=None,
                          usage=None, latency_ms=None, raw=None)

    def decide(self, meeting: dict, calendar_render: str) -> DecideResult:
        slot = self._free_slots[0] if self._free_slots else None
        calls = [{"type": "schedule", "meeting_id": meeting["id"], "slot": slot}] if slot is not None else []
        return DecideResult(tool_calls=calls, text=None, thinking=None,
                            usage=None, latency_ms=None, raw=None)
```

Then export it from `__init__.py`:

```python
# calendar_game/clients/__init__.py
from calendar_game.clients.always_free import AlwaysFreeClient
```

## Using Custom Clients in a Game

### Inject clients directly (for testing or scripted experiments)

```python
from calendar_game.game import CalendarGame, CalendarGameConfig
from calendar_game.agents import Agent
from calendar_game.calendar import Calendar
from calendar_game.clients.always_free import AlwaysFreeClient

config = CalendarGameConfig(seed=42, num_agents=2, num_meetings=1)
game = CalendarGame(config)
scenario = game.generate_scenario()

agents = []
for agent_id in range(config.num_agents):
    agent = Agent(AlwaysFreeClient())
    cal = Calendar(num_slots=config.num_slots)
    cal.slots = list(scenario["calendars"][agent_id])
    agent.calendar = cal
    agents.append(agent)

trace = game._run_with_agents(agents, scenario)
```

### Mix client types across agents

```python
from calendar_game.clients import LLMClient, ScriptedClient
from a2a_engine.llm.factory import make_llm_client

clients = [
    LLMClient(make_llm_client({"model": "gpt-4o"})),  # agent 0: LLM
    ScriptedClient(),                                   # agent 1: deterministic
    AlwaysFreeClient(),                                 # agent 2: custom
]
```

### Dry-run mode (ScriptedClient for all agents)

Setting `dry_run=True` on `CalendarGame` replaces every agent with `ScriptedClient` automatically — no API keys needed.

```python
game = CalendarGame(config, dry_run=True)
trace = game.run()
```

### LLM agents via experiment YAML

When running through `run.py`, each agent spec maps to an `LLMClient` wrapping the requested model backend. The `agents` list in the YAML config controls which model each agent uses:

```yaml
defaults:
  game_name: calendar
  num_agents: 2
  communication_protocol: all
  agents:
    - {type: llm, model: gpt-4o}
    - {type: llm, model: claude-sonnet-4-6}
```

Provider credentials are read from the environment unless `api_key` is set
directly on the agent spec:

| Provider | Detection | Credential |
| --- | --- | --- |
| OpenAI | model contains `gpt`, `openai`, `o1`, `o3`, or `o4` | `OPENAI_API_KEY` |
| Anthropic | model contains `claude` | `ANTHROPIC_API_KEY` |
| Gemini API | model contains `gemini` | `GOOGLE_API_KEY` |
| Vertex AI | publisher model names such as `publishers/google/models/...` | Google ADC |
| OpenRouter | model contains `/` | `OPENROUTER_API_KEY` |
| Ollama | model contains `llama`, `mistral`, or `qwen` | local Ollama server |

For a custom OpenAI-compatible provider:

```yaml
agents:
  - type: llm
    model: my-model
    api_format: openai
    api_base: https://provider.example/v1
    api_key: ${MY_PROVIDER_API_KEY}
```

## Existing Clients

| Client | File | When to use |
|--------|------|-------------|
| `ScriptedClient` | `scripted.py` | Dry-runs and unit tests. Sends free-slot DMs on turn 1, picks first free slot in DECISION. |
| `DSMClient` | `dsm.py` | Distributed score-based multi-round negotiation baseline. It proposes slots, exchanges local scores, and chooses a mutually feasible high-scoring slot. |
| `PaperDSMClient` | `dsm.py` | Paper-style privacy-preserving DSM preset with bounded offer sets. |
| `PrivateDSMClient` | `dsm.py` | Higher-privacy DSM preset tuned to reveal fewer options. |
| `SDClient` | `sd.py` | Scheduling Difficulty IL-MAP baseline with tentative bumping and reschedule messages. |
| `IncrementalMAPClient` | `imap.py` | Complete-information incremental MAP baseline for current-meeting insertion. |
| `LLMClient` | `llm.py` | Production. Wraps any `a2a-engine` provider backend; maintains full conversation history. |
| `DSPyClient` | `dspy.py` | LLM client that uses a named prompt variant from `prompt_variants/` or `prompt_variants_redteam/`. |
| `CapturingClient` | `calendar_game/agents.py` | Test doubles. Records every call; returns configurable canned responses. |
