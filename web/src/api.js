// Thin wrapper around the FastAPI server endpoints.
// Change API_BASE if your server runs on a different port.
const API_BASE = "http://127.0.0.1:8080";

export async function getModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`GET /models failed: ${res.status}`);
  return res.json();
}

export async function getAttention(text, model) {
  const res = await fetch(`${API_BASE}/attention`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  if (!res.ok) throw new Error(`POST /attention failed: ${res.status}`);
  return res.json();
  // shape: { tokens, attention: [layer][head][from][to], n_layer, n_head }
}

// SSE generation. Calls onToken(payload) for each streamed token,
// where payload = { token, token_id, attention_row }.
// Returns an object with a stop() method to cancel the stream early.
export function streamGenerate(req, onToken, onDone) {
  const controller = new AbortController();

  fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: controller.signal,
  }).then(async (res) => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep the last (possibly incomplete) chunk
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const payload = JSON.parse(line.slice(6));
          onToken(payload);
        }
      }
    }
    onDone?.();
  }).catch((err) => {
    if (err.name !== "AbortError") console.error("stream error:", err);
  });

  return { stop: () => controller.abort() };
}
