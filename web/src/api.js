const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8080";

export async function getModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`GET /models failed: ${res.status}`);
  return res.json();
}

export async function getCorpus() {
  const res = await fetch(`${API_BASE}/corpus`);
  if (!res.ok) throw new Error(`GET /corpus failed: ${res.status}`);
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
}

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
      buffer = lines.pop();
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