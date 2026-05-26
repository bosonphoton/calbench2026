// Calendar-game renderers for negotiation, communication, and scheduling events.
import { registerRenderer } from "./index.js";

const CHANNEL_LABELS = {
  dm: "DM",
  participant_groupchat: "Participant groupchat",
  all_groupchat: "All groupchat",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function valueText(value) {
  if (value === undefined || value === null || value === "") return "-";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function baseEvent(event, kind = "") {
  const data = event.data ?? {};
  const wrap = el("div", `event calendar-event ${kind}`.trim());
  const meta = el("div", "event-meta");
  const parts = [];
  if (event.timestamp) parts.push(event.timestamp);
  if (data.phase) parts.push(data.phase);
  if (data.round !== undefined) parts.push(`round ${data.round}`);
  if (data.turn !== undefined) parts.push(`turn ${data.turn}`);
  if (data.agent_id !== undefined && data.agent_id !== null) parts.push(`agent ${data.agent_id}`);
  parts.push(event.type);
  meta.textContent = parts.join(" - ");
  wrap.appendChild(meta);
  return wrap;
}

function pill(text, className = "") {
  return el("span", `pill ${className}`.trim(), text);
}

function detailsBlock(title, payload, className = "") {
  if (payload === undefined || payload === null || payload === "") return null;
  const details = document.createElement("details");
  details.className = className;
  const summary = document.createElement("summary");
  summary.textContent = title;
  details.appendChild(summary);
  const pre = el("pre", "json-text");
  pre.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  details.appendChild(pre);
  return details;
}

function renderMessageCard(data, channelOverride = null) {
  const channel = channelOverride ?? data.channel ?? "dm";
  const card = el("div", `message-card channel-${channel}`);
  const head = el("div", "message-head");
  const from = data.from_agent ?? data.from ?? data.agent_id ?? "?";
  const to = data.to_agent ?? data.to ?? data.to_agents;
  const target = Array.isArray(to)
    ? `agents ${to.join(", ")}`
    : (to === undefined || to === null ? "everyone in channel" : `agent ${to}`);
  head.append(
    pill(CHANNEL_LABELS[channel] ?? channel, `channel-pill ${channel}`),
    el("span", "message-route", `Agent ${from} -> ${target}`),
  );
  card.appendChild(head);

  const metaParts = [];
  if (data.meeting_id !== undefined) metaParts.push(`meeting M${data.meeting_id}`);
  const chars = data.content_chars ?? String(data.content ?? "").length;
  metaParts.push(`${chars} chars`);
  card.appendChild(el("div", "message-submeta", metaParts.join(" - ")));
  card.appendChild(el("div", "message-body", data.content ?? ""));
  return card;
}

function communicationEvent(event, channel) {
  const wrap = baseEvent(event, `communication channel-${channel}`);
  wrap.appendChild(renderMessageCard(event.data ?? {}, channel));
  return wrap;
}

function gameStart(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "system-event");
  const title = el("div", "event-title", "Game start");
  const facts = el("div", "fact-row");
  facts.append(
    pill(`agents ${valueText(data.num_agents)}`),
    pill(`slots ${valueText(data.num_slots)}`),
    pill(`protocol ${valueText(data.communication_protocol)}`),
    pill(`difficulty ${valueText(data.difficulty)}`),
  );
  wrap.append(title, facts);
  wrap.appendChild(detailsBlock("Game start payload", data));
  return wrap;
}

function gameEnd(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, data.won ? "system-event success" : "system-event failure");
  const title = el("div", "event-title", data.won ? "Game ended: won" : "Game ended");
  const facts = el("div", "fact-row");
  for (const [key, value] of Object.entries(data)) {
    if (["round", "turn", "phase", "agent_id"].includes(key)) continue;
    facts.appendChild(pill(`${key} ${valueText(value)}`));
  }
  wrap.append(title, facts);
  return wrap;
}

function agentRegistered(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "system-event");
  wrap.appendChild(el("div", "event-title", `Agent ${valueText(data.agent_id)} registered`));
  const facts = el("div", "fact-row");
  facts.append(
    pill(`model ${valueText(data.model)}`),
    pill(`type ${valueText(data.agent_type)}`),
    pill(`density ${valueText(data.initial_calendar_density)}`),
    pill(`elo ${valueText(data.representation_elo)}`),
  );
  if (data.is_nosy_agent) facts.appendChild(pill("nosy agent", "warning"));
  wrap.appendChild(facts);
  wrap.appendChild(detailsBlock("System prompt", data.system_prompt));
  return wrap;
}

function roundStart(event) {
  const data = event.data ?? {};
  const meeting = data.meeting ?? {};
  const wrap = baseEvent(event, "system-event round-start");
  wrap.appendChild(el("div", "event-title", `Round ${valueText(data.round)} - meeting M${valueText(meeting.id)}`));
  const facts = el("div", "fact-row");
  facts.append(
    pill(`participants ${valueText(meeting.participants)}`),
    pill(`speaker order ${valueText(data.speaker_order ?? meeting.speaker_order)}`),
    pill(`protocol ${valueText(data.communication_protocol)}`),
  );
  wrap.appendChild(facts);
  return wrap;
}

function turnStart(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "turn-event");
  const inbox = data.inbox_drained ?? [];
  wrap.appendChild(el("div", "event-title", `Agent ${valueText(data.agent_id)} turn start`));
  wrap.appendChild(el("div", "turn-summary", `${inbox.length} inbox message${inbox.length === 1 ? "" : "s"} drained`));

  if (inbox.length) {
    const inboxWrap = el("div", "inbox-list");
    for (const msg of inbox) {
      inboxWrap.appendChild(renderMessageCard(msg, msg.channel));
    }
    wrap.appendChild(inboxWrap);
  }
  wrap.appendChild(detailsBlock("Calendar shown to agent", data.calendar_render));
  wrap.appendChild(detailsBlock("Prompt sent", data.prompt_sent));
  return wrap;
}

function renderToolCall(action) {
  const card = el("div", "tool-card");
  const type = action?.type ?? "tool";
  const title = el("div", "tool-title", type);
  card.appendChild(title);

  if (type === "dm") {
    card.appendChild(el("div", "tool-body", `to agent ${valueText(action.to)}: ${valueText(action.content)}`));
  } else if (type === "participant_groupchat" || type === "all_groupchat") {
    card.appendChild(el("div", "tool-body", valueText(action.content)));
  } else if (type === "schedule") {
    card.appendChild(el("div", "tool-body", `meeting M${valueText(action.meeting_id)} at slot ${valueText(action.slot)}`));
  } else if (type === "reschedule") {
    card.appendChild(el("div", "tool-body", `item ${valueText(action.item_id)} from slot ${valueText(action.from_slot)} to slot ${valueText(action.to_slot)}`));
    if (action.justification) card.appendChild(el("div", "tool-note", action.justification));
  } else {
    card.appendChild(el("pre", "json-text", JSON.stringify(action, null, 2)));
  }
  return card;
}

function agentOutput(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "turn-event");
  const title = event.type === "turn_end"
    ? `Agent ${valueText(data.agent_id)} cheap-talk output`
    : `Agent ${valueText(data.agent_id)} decision output`;
  wrap.appendChild(el("div", "event-title", title));

  const facts = el("div", "fact-row");
  if (data.latency_ms !== undefined) facts.appendChild(pill(`${Math.round(Number(data.latency_ms))} ms`));
  if (data.retry_count !== undefined) facts.appendChild(pill(`retries ${data.retry_count}`));
  if (data.status) facts.appendChild(pill(`status ${data.status}`));
  wrap.appendChild(facts);

  if (data.thinking) {
    wrap.appendChild(el("div", "thinking", data.thinking));
  }

  const calls = data.tool_calls ?? [];
  if (calls.length) {
    const tools = el("div", "tool-list");
    for (const call of calls) tools.appendChild(renderToolCall(call));
    wrap.appendChild(tools);
  }
  wrap.appendChild(detailsBlock("Raw text", data.text));
  wrap.appendChild(detailsBlock("Usage", data.usage));
  return wrap;
}

function decideStart(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "action-event");
  wrap.appendChild(el("div", "event-title", `${valueText(data.phase)} prompt for agent ${valueText(data.agent_id)}`));
  wrap.appendChild(detailsBlock("Calendar shown to agent", data.calendar_render ?? data.calendar_snapshot_render));
  wrap.appendChild(detailsBlock("Prompt sent", data.prompt_sent));
  return wrap;
}

function batchEvent(event) {
  const data = event.data ?? {};
  const isRejected = event.type === "batch_rejected";
  const isRollback = event.type === "batch_rolled_back";
  const wrap = baseEvent(event, isRejected || isRollback ? "action-event failure" : "action-event success");
  const title = event.type.replace(/_/g, " ");
  wrap.appendChild(el("div", "event-title", `${title} for agent ${valueText(data.agent_id)}`));

  if (data.conflict_description) {
    wrap.appendChild(el("div", "warning-text", data.conflict_description));
  }
  if (data.reason) {
    wrap.appendChild(el("div", "warning-text", data.reason));
  }

  const actions = data.actions ?? [];
  if (actions.length) {
    const tools = el("div", "tool-list");
    for (const action of actions) tools.appendChild(renderToolCall(action));
    wrap.appendChild(tools);
  }
  wrap.appendChild(detailsBlock("Calendar after", data.calendar_render_after));
  return wrap;
}

function resolution(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, data.coordinated ? "action-event success" : "action-event failure");
  wrap.appendChild(el("div", "event-title", `Resolution for meeting M${valueText(data.meeting_id)}`));
  const facts = el("div", "fact-row");
  facts.append(
    pill(`coordinated ${valueText(data.coordinated)}`),
    pill(`slots ${valueText(data.per_agent_slot)}`),
    pill(`conflicts ${valueText(data.slot_conflicts)}`),
  );
  wrap.appendChild(facts);
  wrap.appendChild(detailsBlock("Blocked slot violations", data.blocked_slot_violations));
  return wrap;
}

function invalidToolCall(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "action-event failure");
  wrap.appendChild(el("div", "event-title", `Invalid tool call from agent ${valueText(data.agent_id)}`));
  wrap.appendChild(el("div", "warning-text", valueText(data.reason)));
  wrap.appendChild(detailsBlock("Tool call", data.tool_call));
  return wrap;
}

function decisionFailed(event) {
  const data = event.data ?? {};
  const wrap = baseEvent(event, "action-event failure");
  wrap.appendChild(el("div", "event-title", `Decision failed for agent ${valueText(data.agent_id)}`));
  wrap.appendChild(el("div", "warning-text", `Attempts exhausted: ${valueText(data.attempts_exhausted)}`));
  return wrap;
}

registerRenderer("game_start", gameStart);
registerRenderer("game_end", gameEnd);
registerRenderer("agent_registered", agentRegistered);
registerRenderer("round_start", roundStart);
registerRenderer("dm_sent", (event) => communicationEvent(event, "dm"));
registerRenderer("participant_groupchat_sent", (event) => communicationEvent(event, "participant_groupchat"));
registerRenderer("all_groupchat_sent", (event) => communicationEvent(event, "all_groupchat"));
registerRenderer("turn_start", turnStart);
registerRenderer("turn_end", agentOutput);
registerRenderer("decide_start", decideStart);
registerRenderer("decide_end", agentOutput);
registerRenderer("batch_applied", batchEvent);
registerRenderer("batch_rejected", batchEvent);
registerRenderer("batch_rolled_back", batchEvent);
registerRenderer("resolution", resolution);
registerRenderer("invalid_tool_call", invalidToolCall);
registerRenderer("decision_failed", decisionFailed);
