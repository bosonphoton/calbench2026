// Chat-bubble renderer for "message" events. Speaker on right is "guesser".
import { registerRenderer } from "./index.js";

function messageRenderer(event) {
  const { speaker = "?", text = "" } = event.data ?? {};
  const isGuesser = String(speaker).toLowerCase() === "guesser";
  const row = document.createElement("div");
  row.className = "bubble-row " + (isGuesser ? "right" : "left");
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isGuesser ? " right" : "");
  const sp = document.createElement("div");
  sp.className = "speaker";
  sp.textContent = speaker;
  const tx = document.createElement("div");
  tx.className = "text";
  tx.textContent = text;
  bubble.append(sp, tx);
  row.appendChild(bubble);
  return row;
}

registerRenderer("message", messageRenderer);
