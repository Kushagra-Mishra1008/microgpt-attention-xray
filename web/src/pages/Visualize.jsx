import { useState, useEffect } from "react";
import TokenArcCanvas from "../components/TokenArcCanvas";
import { getAttention } from "../api";

export default function Visualize({ models, selectedModel }) {
  const [text, setText] = useState("First Citizen: You are all resolved");
  const [inputValue, setInputValue] = useState(text);
  const [data, setData] = useState(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [focusIndex, setFocusIndex] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runVisualize() {
    setLoading(true);
    setError(null);
    try {
      const result = await getAttention(inputValue, selectedModel);
      setData(result);
      setText(inputValue);
      setLayer(0);
      setHead(0);
      setFocusIndex(result.tokens.length - 1);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (text) runVisualize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  const modelInfo = models.find((m) => m.name === selectedModel);

  return (
    <>
      <div className="prompt-row">
        <input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runVisualize()}
          placeholder="Enter sequence for analysis..."
        />
        <button onClick={runVisualize} disabled={loading}>
          {loading ? "Loading..." : "Visualize"}
        </button>
      </div>

      {error && <p className="status-text" style={{ color: "#e24b4a" }}>{error}</p>}

      <div className="canvas">
        <div className="stream-badge">
          <span className="stream-dot"></span>
          Active head stream
        </div>
        <div className="status-text" style={{ marginBottom: 8, textAlign: "center" }}>
          {modelInfo
            ? `${modelInfo.name} · ${modelInfo.n_embd}d · ${modelInfo.n_layer}L · ${modelInfo.n_head}H · val loss ${modelInfo.val_loss.toFixed(4)}`
            : "loading model info..."}
        </div>

        {data ? (
          <TokenArcCanvas
            tokens={data.tokens}
            attention={data.attention}
            layer={layer}
            head={head}
            focusIndex={focusIndex}
            onFocusChange={setFocusIndex}
          />
        ) : (
          <p className="status-text">Enter text and hit Visualize to see attention.</p>
        )}

        {data && (
          <div className="controls">
            <div className="slider-group">
              <div className="row">
                <span>Layer</span>
                <span className="value">{layer + 1} / {data.n_layer}</span>
              </div>
              <input
                type="range"
                min={0}
                max={data.n_layer - 1}
                value={layer}
                onChange={(e) => setLayer(Number(e.target.value))}
              />
            </div>
            <div className="slider-group">
              <div className="row">
                <span>Head</span>
                <span className="value">{head + 1} / {data.n_head}</span>
              </div>
              <input
                type="range"
                min={0}
                max={data.n_head - 1}
                value={head}
                onChange={(e) => setHead(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </div>
    </>
  );
}