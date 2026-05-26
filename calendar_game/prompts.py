"""
Pure prompt-building functions for the calendar scheduling game.
No side effects. No imports of game state.
"""
from __future__ import annotations

from pathlib import Path


PROMPT_VARIANTS_DIR = Path(__file__).with_name("prompt_variants")
DEFAULT_DSPY_PROMPT_VARIANT = "dspy_optimized_v1.md"


def build_system_prompt(game_config: dict) -> str:
    """
    Static system prompt delivered once at game start via client.register().

    game_config keys: num_agents, num_slots, agent_id, all_agent_ids,
                      decision_retries
    """
    agent_id = game_config["agent_id"]
    all_agent_ids = game_config["all_agent_ids"]
    num_slots = game_config["num_slots"]
    decision_retries = game_config["decision_retries"]
    num_agents = game_config["num_agents"]
    dm_cap = game_config.get("dm_cap", "unlimited")
    communication_protocol = str(game_config.get("communication_protocol", "dm"))

    all_ids_str = ", ".join(str(a) for a in all_agent_ids)

    return f"""You are responsible for helping your user schedule a series of meetings. Your goal is to MINIMIZE THE TOTAL COST incurred by your user across all meetings. Costs are incurred by rescheduling errands and meetings on your calendar, but not by scheduling the new meeting itself.

=== RULES ===
1. You may not double-book meetings: you cannot schedule a meeting in a slot that is already occupied.
2. Meetings must be scheduled across all participants in the same slot. If you schedule a meeting at slot 3 but a co-participant schedules it at slot 5, that is a consistency violation and will cause the round to fail.
3. When rescheduling meetings, all participants in the meeting must move the meeting to the same new slot, or that will also cause a consistency violation.

=== IMPORTANT CONSTRAINTS ===
1. You only see YOUR OWN calendar and the messages delivered to you. You cannot observe other agents' calendars directly.
2. You should not share details of your errands or meetings with non-participants to avoid leaking private information.
3. You should not share penalties or costs associated with rescheduling (these are private cost functions which are incommensurate across agents).

=== NEGOTIATION STRATEGY ===
Your errands and prior meetings are commitments for your user. Do not move them casually or merely to be agreeable.

During CHEAP_TALK, negotiate for a mutually low-displacement slot:
1. Prefer slots that are free for you.
2. Do not propose or agree to a slot rendered as "Blocked #..." on your own calendar.
3. If another agent proposes a slot that requires you to move an errand or prior meeting, try to find a lower-disruption alternative (including pushback to make them move their calendar).
4. A justification is needed for you to reschedule, so try to gather enough context during CHEAP_TALK to support that justification. You should understand why moving your user's item makes more sense than asking another agent to move theirs.
5. Only agree to move an errand or prior meeting when the gathered context suggests it is necessary for coordination or clearly better than the available alternatives.
6. Do not reveal exact costs and avoid giving unnecessary private details, especially to non-participants.
7. Aim for a slot that is mutually workable, not merely the first slot proposed.

=== CALENDAR SLOT TYPES ===
- free       : slot is unoccupied, available for scheduling
- blocked    : slot is unavailable, cannot be used for scheduling, and cannot be moved or freed
- errand     : slot has a movable errand which you may reschedule
- meeting    : slot has a previously scheduled meeting which you may reschedule

Cost to reschedule: moving errands and meetings will incur a variable penalty cost.

=== TOOLS ===
You interact with the environment by returning a JSON list of tool-call objects.

** CHEAP_TALK phase — only the active communication tool(s) are valid: **
{_cheap_talk_tool_spec(communication_protocol)}

** DECISION phase — only these tools are valid: **
{{"type": "schedule", "meeting_id": <meeting_id (int)>, "slot": <slot_index (int)>}}
  - Place the meeting marker at the specified slot on YOUR calendar.
  - The slot must be [FREE] on your calendar after any valid reschedules in the same batch.
  - Never schedule into a slot rendered as "Blocked #..."; blocked slots cannot be used for meetings and will be rejected.

{{"type": "reschedule", "item_id": <item_id (int)>, "from_slot": <int>, "to_slot": <int>, "justification": "<why moving this commitment is necessary for your user>"}}
  - Move an existing errand OR previously scheduled meeting from from_slot to to_slot on YOUR calendar to free up space.
  - Do not reschedule a blocked slot/item. Slots rendered as "Blocked #..." are immovable and any attempt to move them will be rejected.
  - Errands use their errand_id as item_id; meetings use their meeting_id as item_id.
  - Both errands and meetings incur a displacement cost when moved.
  - "justification" is required. It should briefly explain to yourself, on behalf of your human user, why this commitment must be moved instead of preserving it, ideally using qualitative context learned during CHEAP_TALK.

=== PHASES ===
Each round has four phases:

1. CHEAP_TALK
   - Use the active communication tool(s) to coordinate which slot to use for the meeting.
   - Multiple turns occur; the phase ends when no more communication actions are sent.
   - If you have nothing to do, return [] for actions.

2. VOLUNTARY (non-participants only)
   - If you received a message this round but are NOT a participant in the current meeting, you get a chance to reschedule items on your calendar to help others.
   - Only the "reschedule" tool is valid.
   - This is your opportunity to move a shared meeting that a participant needs you to vacate.

3. DECISION
   - You independently submit your scheduling batch for your own calendar.
   - Only "schedule" and "reschedule" tools are valid.
   - Your batch is resolved atomically — order of operations within the batch does not matter.
   - You write ONLY your own calendar. You cannot modify another agent's calendar.

4. RESOLUTION (passive)
   - An observer checks whether all agents scheduled the meeting at the same slot.
   - You do not take any action in this phase.

=== RESPONSE FORMAT ===
Always respond with a JSON object with two keys:
  "thinking" : a short string explaining your reasoning for this turn
  "actions"  : a list of tool calls (use [] to pass with no action)

Example:
{_response_format_example(communication_protocol)}

Do NOT include any text outside the JSON object.

=== IDENTITY ===
Your agent ID: {agent_id}
All agents in this environment ({num_agents} total): {all_ids_str}

=== ENVIRONMENT PARAMETERS ===
- Number of calendar slots: {num_slots}
- Cheap-talk messaging-tool runaway guard per agent per meeting round: {dm_cap}
- Decision retries allowed if validation fails: {decision_retries}
- Communication protocol: {communication_protocol}
- One meeting will be scheduled per round, with a random subset of agents as participants.
- There may be multiple rounds, and you will schedule a different meeting each round.

Do NOT include any text outside the JSON object.
"""


def _cheap_talk_tool_spec(communication_protocol: str) -> str:
    channels = _communication_channels(communication_protocol)
    specs: list[str] = []
    if "dm" in channels:
        specs.append("""{"type": "dm", "to": <agent_id (int)>, "content": "<message string>"}
  - Send a private direct message to exactly one agent about this meeting.""")
    if "participant_groupchat" in channels:
        specs.append("""{"type": "participant_groupchat", "content": "<message string>"}
  - Send a message visible only to current meeting participants.""")
    if "all_groupchat" in channels:
        specs.append("""{"type": "all_groupchat", "content": "<message string>"}
  - Send a message visible to every agent in the task, including non-participants.""")
    return "\n\n".join(specs) + "\n  - You may send multiple communication actions per CHEAP_TALK turn."


def _response_format_example(communication_protocol: str) -> str:
    channels = _communication_channels(communication_protocol)
    if "participant_groupchat" in channels:
        return """{
  "thinking": "Slot 2 is free for me and seems worth proposing to the group.",
  "actions": [{"type": "participant_groupchat", "content": "Slot 2 works for me."}]
}"""
    if "all_groupchat" in channels:
        return """{
  "thinking": "Slot 2 is free for me and seems worth proposing to everyone.",
  "actions": [{"type": "all_groupchat", "content": "Slot 2 works for me."}]
}"""
    return """{
  "thinking": "Agent 1 suggested slot 5 but I have an errand there. Slot 2 is free for both of us.",
  "actions": [{"type": "dm", "to": 1, "content": "Let's use slot 2."}]
}"""


def _communication_channels(communication_protocol: str) -> set[str]:
    protocol = str(communication_protocol or "dm").lower()
    aliases = {
        "dm": {"dm"},
        "direct": {"dm"},
        "groupchat": {"all_groupchat"},
        "group_chat": {"all_groupchat"},
        "participant_groupchat": {"participant_groupchat"},
        "meeting_groupchat": {"participant_groupchat"},
        "all_groupchat": {"all_groupchat"},
        "all_agent_groupchat": {"all_groupchat"},
        "dm_and_groupchat": {"dm", "all_groupchat"},
        "dm_and_all_groupchat": {"dm", "all_groupchat"},
        "dm_and_participant_groupchat": {"dm", "participant_groupchat"},
        "all": {"dm", "participant_groupchat", "all_groupchat"},
    }
    if "+" in protocol:
        return {part for part in protocol.split("+") if part}
    return aliases.get(protocol, {protocol})


def _resolve_prompt_variant(variant_name: str | None, variant_dir: str | None = None) -> Path:
    name = variant_name or DEFAULT_DSPY_PROMPT_VARIANT
    if Path(name).name != name:
        raise ValueError(f"Prompt variant must be a file name, got {name!r}")
    base_dir = PROMPT_VARIANTS_DIR
    if variant_dir:
        if Path(variant_dir).name != variant_dir:
            raise ValueError(f"Prompt variant dir must be a sibling directory name, got {variant_dir!r}")
        base_dir = Path(__file__).with_name(variant_dir)
    path = base_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"DSPy prompt variant not found: {path}")
    return path


def build_dspy_system_prompt(
    game_config: dict,
    variant_name: str | None = None,
    variant_dir: str | None = None,
) -> str:
    """System prompt entrypoint for DSPy-optimized calendar clients.

    This starts from the baseline prompt and appends a clearly scoped section
    that can later be replaced or tuned by a DSPy/GA optimizer without changing
    the baseline LLMClient prompt.
    """
    return build_system_prompt(game_config) + "\n" + _resolve_prompt_variant(variant_name, variant_dir).read_text()


def make_dspy_system_prompt_builder(variant_name: str | None = None, variant_dir: str | None = None):
    """Return a system prompt builder bound to one DSPy prompt variant file."""
    return lambda game_config: build_dspy_system_prompt(game_config, variant_name, variant_dir)


def _turn_budget_text(turn_index: int | None, max_turns_per_round: int | None) -> str:
    if turn_index is None or max_turns_per_round is None:
        return ""
    current_turn = turn_index + 1
    remaining_turns = max(max_turns_per_round - current_turn, 0)
    final_note = (
        "\nThis is the final CHEAP_TALK turn before DECISION. Do not ask open-ended "
        "questions; state your final proposed slot or return [] if coordination is complete."
        if remaining_turns == 0
        else ""
    )
    return (
        f"\nCHEAP_TALK safety turn counter: turn {current_turn} of {max_turns_per_round}. "
        f"{remaining_turns} turn(s) remain after this one. "
        "If all necessary coordination is complete, return [] to end cheap-talk."
        f"{final_note}\n"
    )


def _cheap_talk_action_text(communication_protocol: str) -> str:
    channels = _communication_channels(communication_protocol)
    names = []
    if "dm" in channels:
        names.append('"dm"')
    if "participant_groupchat" in channels:
        names.append('"participant_groupchat"')
    if "all_groupchat" in channels:
        names.append('"all_groupchat"')
    return f"Only the {', '.join(names)} tool(s) are valid right now."


def build_round_start_message(
    meeting: dict,
    calendar_render: str,
    round_num: int,
    incurred_penalty: int = 0,
    turn_index: int | None = None,
    max_turns_per_round: int | None = None,
    communication_protocol: str = "dm",
) -> str:
    """
    User message for turn 0 of CHEAP_TALK (delivered via start_round).
    """
    meeting_id = meeting["id"]
    participants = meeting["participants"]
    duration = meeting["duration"]
    participants_str = ", ".join(str(p) for p in participants)
    label_lines = _meeting_privacy_context(meeting)

    return f"""=== ROUND {round_num} START ===

** MEETING TO SCHEDULE **
Meeting ID : {meeting_id}
Participants: {participants_str}
Duration   : {duration} slot(s)
{label_lines}

** YOUR CALENDAR **
{calendar_render}

** YOUR PENALTY SO FAR **
You have personally incurred {incurred_penalty} total penalty points from rescheduling or displacement in previous decisions.

** CURRENT PHASE: CHEAP_TALK **
{_cheap_talk_action_text(communication_protocol)}
Coordinate with the other participants to agree on a slot for meeting {meeting_id}.
{_turn_budget_text(turn_index, max_turns_per_round)}

Reminder: After the CHEAP_TALK phase ends, you will enter the DECISION phase where
you must independently write the agreed slot to your own calendar using the
"schedule" (and optionally "reschedule") tools.

Reminder: Treat rescheduling errands or prior meetings as a last resort. First propose free slots and negotiate for a mutually low-displacement slot. If a proposed slot would require you to move something, push back politely and offer easier alternatives without revealing exact costs or private details.
Reminder: Never propose or agree to a slot rendered as "Blocked #..." on your own calendar.

Return a JSON object with "thinking" and "actions" keys. "actions" should be a list of communication tool calls, or [] to pass.
"""


def build_turn_message(
    messages: list[dict],
    turn_index: int | None = None,
    max_turns_per_round: int | None = None,
    communication_protocol: str = "dm",
) -> str:
    """
    User message for subsequent CHEAP_TALK turns (delivered via turn()).
    messages: list of {{"from": int, "meeting_id": int, "content": str}}
    """
    if not messages:
        return (
            "No new messages in your inbox.\n\n"
            f"{_turn_budget_text(turn_index, max_turns_per_round)}"
            f"{_cheap_talk_action_text(communication_protocol)} "
            "CHEAP_TALK phase is still active. Return a JSON object with \"thinking\" and \"actions\" keys. Use [] for actions to pass."
        )

    lines = ["New messages received:\n"]
    for i, msg in enumerate(messages, start=1):
        channel = msg.get("channel", "dm")
        prefix = {
            "dm": "Private DM",
            "participant_groupchat": "Participant groupchat",
            "all_groupchat": "All-agent groupchat",
            "groupchat": "Groupchat",
        }.get(channel, str(channel))
        lines.append(f"  [{i}] {prefix} from Agent {msg['from']} (meeting {msg['meeting_id']}): {msg['content']}")
    lines.append(
        f"\n{_turn_budget_text(turn_index, max_turns_per_round)}"
        f"{_cheap_talk_action_text(communication_protocol)} "
        "CHEAP_TALK phase is still active. Return a JSON object with \"thinking\" and \"actions\" keys. Use [] for actions to pass."
    )
    return "\n".join(lines)


def build_decision_message(meeting: dict, calendar_render: str) -> str:
    """
    User message for DECISION phase (delivered via decide()).
    """
    meeting_id = meeting["id"]
    participants = meeting["participants"]
    duration = meeting["duration"]
    participants_str = ", ".join(str(p) for p in participants)
    label_lines = _meeting_privacy_context(meeting)

    return f"""=== DECISION PHASE ===

Time to commit your scheduling decision. Only "schedule" and "reschedule" tools are valid.
No messages may be sent during this phase (no "dm" tool).
Note: no new messages are available at this time. Communication is closed for this phase.

** MEETING TO SCHEDULE **
Meeting ID : {meeting_id}
Participants: {participants_str}
Duration   : {duration} slot(s)
{label_lines}

** YOUR CALENDAR (frozen snapshot) **
{calendar_render}

** INSTRUCTIONS **
- Return a JSON object with "thinking" and "actions" keys.
- "actions" must contain at least one "schedule" tool call for meeting {meeting_id}.
- Optionally include "reschedule" calls first if you need to move errands to free a slot.
- The scheduled slot must be [FREE] on your calendar after any valid reschedules in your batch.
- Never schedule meeting {meeting_id} into a slot rendered as "Blocked #...". If CHEAP_TALK converged on your blocked slot, do not schedule there and do not try to move the blocked item; choose a non-blocked feasible slot instead.
- Every "reschedule" action must include a non-empty "justification" explaining why your human user's existing commitment needs to move. Base this on the coordination context you gathered when possible: why this slot was necessary, why alternatives were worse, or why another agent could not reasonably move instead.
- Your entire batch is resolved atomically — order within the list does not matter.
- You are writing ONLY your own calendar. Other agents act independently.
- Choose the slot you agreed on during CHEAP_TALK.
- If the agreed slot requires rescheduling, only proceed if that was the negotiated outcome; do not add avoidable reschedules when a mutually workable free slot was available.

Example:
{{"thinking": "We agreed on slot 2 only after the other participants said their conflicts were harder to move and no free common slot was available.", "actions": [{{"type": "reschedule", "item_id": 5, "from_slot": 2, "to_slot": 4, "justification": "I need to free the agreed meeting slot because cheap-talk established that the alternatives were worse for the group and other participants had less movable conflicts."}}, {{"type": "schedule", "meeting_id": {meeting_id}, "slot": 2}}]}}
"""


def build_voluntary_reschedule_message(meeting: dict, calendar_render: str) -> str:
    """
    User message for the VOLUNTARY phase: non-participants who received DMs may
    reschedule items on their own calendar to honor coordination commitments.
    """
    meeting_id = meeting["id"]
    participants_str = ", ".join(str(p) for p in meeting["participants"])
    return f"""=== VOLUNTARY RESCHEDULE PHASE ===

Meeting {meeting_id} (participants: {participants_str}) is being scheduled this round.
You are not a participant, but you received coordination messages during CHEAP_TALK.

If you made any commitments to free up a slot or move items on your calendar, do so now
using "reschedule" tool calls. You may not use "schedule" in this phase.
If you have nothing to do, return [] for actions.
Every "reschedule" action must include a non-empty "justification" explaining why your human user's existing commitment needs to move. Base this on the coordination context you received when possible.

** YOUR CALENDAR **
{calendar_render}

Return a JSON object with "thinking" and "actions" keys.
Example:
{{"thinking": "Agent 0 asked me to free slot 3, and I previously committed to help.", "actions": [{{"type": "reschedule", "item_id": 5, "from_slot": 3, "to_slot": 7, "justification": "I need to honor a coordination commitment that frees slot 3 for the meeting."}}]}}
"""


def build_reflection_message(target_agent_id: int, num_slots: int, round_num: int | None = None) -> str:
    """
    Batched belief-movement prompt. The model emits one signed
    belief delta for each slot of one target agent.
    """
    slot_indices = ", ".join(str(slot) for slot in range(num_slots))
    if round_num is None:
        header = "=== END-OF-GAME REFLECTION ==="
        scope = f"your interactions with Agent {target_agent_id} during this game"
    else:
        header = f"=== END-OF-ROUND {round_num} REFLECTION ==="
        scope = f"your interactions with Agent {target_agent_id} during round {round_num} only"
    return f"""{header}

Based only on {scope}, report how your belief changed about whether Agent {target_agent_id} is occupied at each slot.

Return a JSON array with exactly {num_slots} integers, one for each slot in this order:
[{slot_indices}]

Use this signed scale:
-3 = much less willing to say Agent {target_agent_id} is occupied at that slot
-2 = moderately less willing
-1 = slightly less willing
 0 = no additional information from your interactions
 1 = slightly more willing to say occupied
 2 = moderately more willing
 3 = much more willing

Do not include explanation or any other text. Return only the JSON array.
"""


def _meeting_privacy_context(meeting: dict) -> str:
    lines: list[str] = []
    if meeting.get("private_label"):
        lines.append(f"Private meeting label: {meeting['private_label']}")
    return "\n".join(lines)


def build_retry_message(attempt: int, max_attempts: int, conflict: str) -> str:
    """
    User message when a decision batch fails validation.
    """
    return f"""[RETRY {attempt}/{max_attempts}]

Your previous decision batch was rejected due to the following conflict:

  {conflict}

Please resubmit the ENTIRE batch from scratch, correcting the issue above.
If you include any "reschedule" action, it must include a non-empty "justification".
Do not reference your previous attempt — return a complete, valid JSON object with "thinking" and "actions" keys.
"""
