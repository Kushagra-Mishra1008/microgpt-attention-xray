import { useState, useRef } from "react";
import { streamGenerate } from "../api";

export default function Generate({ selectedModel }) {
  const [prompt, setPrompt] = useState("First Citizen:");
  const [temperature, setTemperature] = useState(0.8);
  const [generatedText, setGeneratedText] = useState("");
  const [lastWeight, setLastWeight] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const streamRef = useRef(null);

  function start() {
    setGeneratedText(prompt);
    setIsGenerating(true);
    streamRef.current = streamGenerate(
      { prompt, max_new_tokens: 300, temperature, model: selectedModel },
      (payload) => {
        setGeneratedText((prev) => prev + payload.token);
        const maxAttn = Math.max(...payload.attention_row);
        setLastWeight(maxAttn);
      },
      () => setIsGenerating(false)
    );
  }

  function stop() {
    streamRef.current?.stop();
    setIsGenerating(false);
  }

  return (
    <>
      <div className="canvas" style={{ minHeight: 60, flexDirection: "row", alignItems: "center", gap: 24 }}>
        <div className="slider-group" style={{ maxWidth: 320 }}>
          <div className="row">
            <span>Temperature</span>
            <span className="value">{temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.01}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </div>
        <button onClick={isGenerating ? stop : start} disabled={!selectedModel}>
          {isGenerating ? "Stop" : "Generate"}
        </button>
        {lastWeight !== null && (
          <span className="status-text">max attention weight: {lastWeight.toFixed(3)}</span>
        )}
      </div>

      <div className="prompt-row" style={{ marginTop: 12 }}>
        <input value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={isGenerating} />
      </div>

      <div className="canvas">
        <p style={{ fontFamily: "var(--mono)", fontSize: 14, lineHeight: 1.7, whiteSpace: "pre-wrap" }}>
          {generatedText || "Press Generate to stream text from the model."}
          {isGenerating && <span style={{ opacity: 0.5 }}>▍</span>}
        </p>
      </div>
    </>
  );
}
