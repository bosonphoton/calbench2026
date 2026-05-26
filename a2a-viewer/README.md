# a2a Trace Viewer

Static, dependency-free viewer for CalBench/a2a trace JSON files. It is plain
HTML, CSS, and ES modules, so there is no build step.

## Run

```sh
cd a2a-viewer
python3 -m http.server 8000
# open http://localhost:8000
```

Some browsers block ES modules from `file://`, so the local HTTP server is the
recommended path.

Load a trace with **Open trace** or by dropping a `.json` or `.jsonl` file onto
the page. Current calendar traces under `results/` work directly, for example:

```text
results/live_prior_meeting_varied_diagnostic_fast/fast_varied_1_10_100.json
```

## Current UI

The viewer now opens directly at `index.html`; it no longer depends on a
separate per-game route such as `/games/calendar/tasks/viewer.html`.

The header summarizes:

- trace and task identifiers
- agents, model names, agent types, initial calendar densities, and ELOs
- communication counts for DMs, participant groupchat, and all-agent groupchat
- selected run metrics and final-state fields

The timeline supports filters for:

- DMs
- participant groupchat
- all-agent groupchat
- agent turns
- decisions and applied or rejected actions
- system events

## Calendar Events Rendered

Calendar-game traces use dedicated renderers for:

- `game_start`, `game_end`
- `agent_registered`
- `round_start`
- `turn_start`, including drained inbox messages
- `turn_end`, including thinking and tool calls
- `dm_sent`
- `participant_groupchat_sent`
- `all_groupchat_sent`
- `decide_start`, `decide_end`
- `batch_applied`, `batch_rejected`, `batch_rolled_back`
- `resolution`
- `invalid_tool_call`, `decision_failed`

Unknown event types fall back to a collapsible JSON block, so newer traces
remain readable even before a custom renderer is added.

## Layout

```text
index.html
css/styles.css
games.json              legacy registry, currently maps calendar to ./index.html
js/
  main.js               file loading, drag/drop, JSON/JSONL parsing, filters
  header.js             trace summary, agents, communication stats, metrics
  timeline.js           event filtering and event-list rendering
  renderers/
    index.js            renderer registry and default JSON renderer
    calendar.js         calendar-game events and communication renderers
    game_lifecycle.js   legacy lifecycle renderer
    message.js          legacy generic message renderer
```

## Adding a Renderer

Create `js/renderers/<name>.js`:

```js
import { registerRenderer } from "./index.js";

registerRenderer("my_event_type", (event) => {
  const div = document.createElement("div");
  div.className = "event calendar-event";
  div.textContent = `my_event_type: ${JSON.stringify(event.data)}`;
  return div;
});
```

Then import it from `js/main.js` so registration runs when the viewer loads.
