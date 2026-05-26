// Renderer registry. Game-specific renderers register themselves by event type.
export const renderers = {};

export function registerRenderer(eventType, fn) {
  renderers[eventType] = fn;
}

export function renderEvent(event) {
  const fn = renderers[event.type] ?? renderers.default;
  return fn(event);
}

// ---------- Default renderer: pretty-printed JSON in a collapsible block.
function defaultRenderer(event) {
  const wrap = document.createElement("div");
  wrap.className = "event default-event";
  const meta = document.createElement("div");
  meta.className = "event-meta";
  meta.textContent = `${event.timestamp ?? ""} · ${event.type}`;
  wrap.appendChild(meta);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `${event.type} payload`;
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(event.data ?? {}, null, 2);
  details.appendChild(pre);
  wrap.appendChild(details);
  return wrap;
}

registerRenderer("default", defaultRenderer);
