import { renderEvent } from "./renderers/index.js";

const CHANNEL_EVENT_TYPES = {
  dm_sent: "dm",
  participant_groupchat_sent: "participant_groupchat",
  all_groupchat_sent: "all_groupchat",
};

const AGENT_TURN_EVENTS = new Set(["turn_start", "turn_end"]);
const ACTION_EVENTS = new Set([
  "decide_start",
  "decide_end",
  "batch_applied",
  "batch_rejected",
  "batch_rolled_back",
  "decision_failed",
  "resolution",
  "fallback_start",
  "fallback_end",
]);

function eventGroup(event) {
  if (CHANNEL_EVENT_TYPES[event.type]) return "channel";
  if (AGENT_TURN_EVENTS.has(event.type)) return "agent_turns";
  if (ACTION_EVENTS.has(event.type)) return "actions";
  return "system";
}

function isVisible(event, filters) {
  const channel = CHANNEL_EVENT_TYPES[event.type] ?? event.data?.channel;
  if (channel && channel in (filters?.channels ?? {})) {
    return filters.channels[channel];
  }
  const group = eventGroup(event);
  return filters?.groups?.[group] ?? true;
}

export function renderTimeline(trace, filters = {}) {
  const root = document.createElement("div");
  const h2 = document.createElement("h2");
  const events = trace.events ?? [];
  const visibleEvents = events.filter((event) => isVisible(event, filters));
  h2.textContent = `Events (${visibleEvents.length} of ${events.length})`;
  root.appendChild(h2);

  if (!visibleEvents.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No events match the active filters.";
    root.appendChild(empty);
    return root;
  }

  for (const ev of visibleEvents) {
    root.appendChild(renderEvent(ev));
  }
  return root;
}
