// Renders the trace header panel: ids, agents, communication, metrics, final state.

function row(dt, dd) {
  const a = document.createElement("dt");
  a.textContent = dt;
  const b = document.createElement("dd");
  b.textContent = formatValue(dd);
  return [a, b];
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") return "-";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function section(root, title) {
  const h2 = document.createElement("h2");
  h2.textContent = title;
  if (root.children.length) h2.style.marginTop = "14px";
  root.appendChild(h2);
  return h2;
}

function countEvents(trace, type) {
  return (trace.events ?? []).filter((event) => event.type === type).length;
}

function firstEvent(trace, type) {
  return (trace.events ?? []).find((event) => event.type === type)?.data ?? {};
}

function communicationStats(trace) {
  const metrics = trace.metrics ?? {};
  const dmCount = metrics.total_dms_sent ?? countEvents(trace, "dm_sent");
  const participantCount = (
    metrics.total_participant_groupchat_messages_sent
    ?? countEvents(trace, "participant_groupchat_sent")
  );
  const allCount = (
    metrics.total_all_groupchat_messages_sent
    ?? countEvents(trace, "all_groupchat_sent")
  );
  const dmChars = metrics.total_dm_chars ?? sumChars(trace, "dm_sent");
  const participantChars = (
    metrics.total_participant_groupchat_chars
    ?? sumChars(trace, "participant_groupchat_sent")
  );
  const allChars = metrics.total_all_groupchat_chars ?? sumChars(trace, "all_groupchat_sent");

  return {
    protocol: metrics.communication_protocol ?? trace.config?.communication_protocol ?? "-",
    dm_messages: dmCount,
    participant_groupchat_messages: participantCount,
    all_groupchat_messages: allCount,
    total_messages: dmCount + participantCount + allCount,
    dm_chars: dmChars,
    participant_groupchat_chars: participantChars,
    all_groupchat_chars: allChars,
  };
}

function sumChars(trace, type) {
  return (trace.events ?? [])
    .filter((event) => event.type === type)
    .reduce((total, event) => {
      const data = event.data ?? {};
      return total + (Number.isFinite(data.content_chars) ? Number(data.content_chars) : String(data.content ?? "").length);
    }, 0);
}

function renderAgents(trace) {
  const agents = trace.config?.agents ?? [];
  const gameStart = firstEvent(trace, "game_start");
  const registered = new Map(
    (trace.events ?? [])
      .filter((event) => event.type === "agent_registered")
      .map((event) => [event.data?.agent_id, event.data ?? {}]),
  );

  const root = document.createElement("div");
  root.className = "agent-grid";
  const maxAgents = Math.max(agents.length, Number(trace.config?.num_agents ?? 0), registered.size);

  for (let agentId = 0; agentId < maxAgents; agentId += 1) {
    const fromConfig = agents[agentId] ?? {};
    const fromEvent = registered.get(agentId) ?? {};
    const card = document.createElement("div");
    card.className = "agent-card";

    const title = document.createElement("div");
    title.className = "agent-title";
    title.textContent = `Agent ${agentId}`;
    card.appendChild(title);

    const model = fromEvent.model ?? fromConfig.model ?? gameStart.agent_models?.[agentId] ?? "-";
    const type = fromEvent.agent_type ?? fromConfig.type ?? gameStart.agent_types?.[agentId] ?? "-";
    const density = fromEvent.initial_calendar_density ?? gameStart.initial_calendar_density_by_agent?.[agentId];
    const elo = fromEvent.representation_elo ?? gameStart.representation_elo_by_agent?.[agentId];

    const dl = document.createElement("dl");
    dl.className = "kv compact";
    dl.append(...row("model", model));
    dl.append(...row("type", type));
    dl.append(...row("density", density));
    dl.append(...row("elo", elo));
    card.appendChild(dl);
    root.appendChild(card);
  }

  return root;
}

export function renderHeader(trace) {
  const root = document.createElement("div");
  section(root, "Trace");

  const dl = document.createElement("dl");
  dl.className = "kv";
  dl.append(...row("game_id", trace.game_id ?? "?"));
  dl.append(...row("game_name", trace.config?.game_name ?? "?"));
  dl.append(...row("task_id", trace.metrics?.task_id ?? trace.config?.task_id ?? "-"));
  dl.append(...row("experiment", trace.config?.experiment_name ?? "-"));
  dl.append(...row("events", (trace.events ?? []).length));
  dl.append(...row("started_at", trace.started_at ?? "-"));
  dl.append(...row("ended_at", trace.ended_at ?? "-"));
  root.appendChild(dl);

  section(root, "Agents");
  root.appendChild(renderAgents(trace));

  section(root, "Communication");
  const dlComm = document.createElement("dl");
  dlComm.className = "kv";
  for (const [key, value] of Object.entries(communicationStats(trace))) {
    dlComm.append(...row(key, value));
  }
  root.appendChild(dlComm);

  const metrics = trace.metrics ?? {};
  if (Object.keys(metrics).length) {
    section(root, "Metrics");
    const dl2 = document.createElement("dl");
    dl2.className = "kv";
    for (const key of [
      "difficulty",
      "difficulty_score",
      "coordination_rate",
      "slot_conflict_rate",
      "efficiency",
      "meetings_scheduled",
      "realized_cost",
      "optimal_cost",
      "greedy_cost",
      "success_score",
      "cost_score",
    ]) {
      if (key in metrics) dl2.append(...row(key, metrics[key]));
    }
    root.appendChild(dl2);
  }

  const finalState = trace.final_state ?? {};
  if (Object.keys(finalState).length) {
    section(root, "Final State");
    const dl3 = document.createElement("dl");
    dl3.className = "kv";
    for (const [key, value] of Object.entries(finalState)) {
      if (key === "calendars") continue;
      dl3.append(...row(key, value));
    }
    root.appendChild(dl3);
  }

  return root;
}
