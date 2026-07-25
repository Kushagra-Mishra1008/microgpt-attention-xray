import { useState, useEffect } from "react";
import "./theme.css";
import Visualize from "./pages/Visualize";
import Generate from "./pages/Generate";
import { getModels } from "./api";

export default function App() {
  const [tab, setTab] = useState("visualize");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    getModels()
      .then((list) => {
        setModels(list);
        if (list.length) setSelectedModel(list[0].name);
      })
      .catch((e) => console.error("failed to load models:", e));
  }, []);

  return (
    <div className="app-shell">
      <div className="top-bar">
        <div style={{ display: "flex", alignItems: "center" }}>
          <h1>Attention X-Ray</h1>
          <div className="tabs">
            <button className={tab === "visualize" ? "active" : ""} onClick={() => setTab("visualize")}>
              Visualize
            </button>
            <button className={tab === "generate" ? "active" : ""} onClick={() => setTab("generate")}>
              Generate
            </button>
          </div>
        </div>

        <div className="model-toggle">
          {models.map((m) => (
            <button
              key={m.name}
              className={selectedModel === m.name ? "active" : ""}
              onClick={() => setSelectedModel(m.name)}
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>

      <div className="main">
        {selectedModel ? (
          tab === "visualize" ? (
            <Visualize models={models} selectedModel={selectedModel} />
          ) : (
            <Generate selectedModel={selectedModel} />
          )
        ) : (
          <p className="status-text">Loading models from server...</p>
        )}
      </div>
    </div>
  );
}