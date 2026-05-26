// Bootstrap: file picker, drag-and-drop, render orchestration.
import "./renderers/index.js";       // registers default
import "./renderers/message.js";     // chat-bubble
import "./renderers/game_lifecycle.js"; // game_start + game_end
import "./renderers/calendar.js";    // calendar game events and communication
// Downstream games can add custom renderers by importing them here, e.g.:
//   import "./renderers/word_guess.js";

import { renderHeader } from "./header.js";
import { renderTimeline } from "./timeline.js";

const fileInput = document.getElementById("file-input");
const headerPanel = document.getElementById("header-panel");
const timelinePanel = document.getElementById("timeline");
const status = document.getElementById("status");
const dropzone = document.getElementById("dropzone");
const channelFilterInputs = [...document.querySelectorAll("[data-channel-filter]")];
const eventFilterInputs = [...document.querySelectorAll("[data-event-filter]")];

let currentTrace = null;

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.classList.toggle("error", isError);
}

function render(trace) {
  currentTrace = trace;
  headerPanel.replaceChildren(renderHeader(trace));
  timelinePanel.replaceChildren(renderTimeline(trace, currentFilters()));
}

function currentFilters() {
  return {
    channels: Object.fromEntries(
      channelFilterInputs.map((input) => [input.dataset.channelFilter, input.checked]),
    ),
    groups: Object.fromEntries(
      eventFilterInputs.map((input) => [input.dataset.eventFilter, input.checked]),
    ),
  };
}

function rerenderTimeline() {
  if (!currentTrace) return;
  timelinePanel.replaceChildren(renderTimeline(currentTrace, currentFilters()));
}

function parseTraceText(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("file is empty");

  try {
    return JSON.parse(trimmed);
  } catch (jsonError) {
    const lines = trimmed.split(/\r?\n/).filter(Boolean);
    const records = lines.map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (lineError) {
        throw new Error(`line ${index + 1}: ${lineError.message}`);
      }
    });
    const traceRecord = records.find((record) => Array.isArray(record?.events));
    if (traceRecord) return traceRecord;
    return {
      game_id: "jsonl-trace",
      config: records.find((record) => record?.config)?.config ?? {},
      events: records,
      metrics: {},
      final_state: {},
      parse_warning: jsonError.message,
    };
  }
}

async function loadFile(file) {
  try {
    const text = await file.text();
    const trace = parseTraceText(text);
    render(trace);
    setStatus(`Loaded ${file.name} (${(trace.events ?? []).length} events)`);
  } catch (err) {
    setStatus(`Failed to load ${file.name}: ${err.message}`, true);
  }
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) loadFile(f);
  fileInput.value = "";
});

for (const input of [...channelFilterInputs, ...eventFilterInputs]) {
  input.addEventListener("change", rerenderTimeline);
}

// Drag-and-drop on the whole window.
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types?.includes("Files")) return;
  e.preventDefault();
  dragDepth += 1;
  dropzone.classList.remove("hidden");
});
window.addEventListener("dragover", (e) => {
  if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropzone.classList.add("hidden");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropzone.classList.add("hidden");
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});

const pendingTrace = sessionStorage.getItem("a2a_pending_trace");
if (pendingTrace) {
  sessionStorage.removeItem("a2a_pending_trace");
  try {
    const trace = parseTraceText(pendingTrace);
    render(trace);
    setStatus(`Loaded routed trace (${(trace.events ?? []).length} events)`);
  } catch (err) {
    setStatus(`Failed to load routed trace: ${err.message}`, true);
  }
}
