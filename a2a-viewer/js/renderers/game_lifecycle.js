// Renderers for "game_start" and "game_end" events.
import { registerRenderer } from "./index.js";

function gameStart(event) {
  const div = document.createElement("div");
  div.className = "lifecycle";
  const data = event.data ?? {};
  const bits = Object.entries(data).map(([k, v]) => `${k}=${v}`).join(" · ");
  div.innerHTML = `<span class="pill">game_start ${bits}</span>`;
  return div;
}

function gameEnd(event) {
  const div = document.createElement("div");
  const won = !!(event.data && event.data.won);
  div.className = "lifecycle " + (won ? "win" : "lose");
  const turns = (event.data ?? {}).turns_used ?? "?";
  div.innerHTML = `<span class="pill">game_end · ${won ? "WON" : "LOST"} · turns=${turns}</span>`;
  return div;
}

registerRenderer("game_start", gameStart);
registerRenderer("game_end", gameEnd);
