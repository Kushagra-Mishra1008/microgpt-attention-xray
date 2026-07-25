import { useState, useEffect } from "react";
import "./theme.css";
import Sidebar from "./components/Sidebar";
import Visualize from "./pages/Visualize";
import Generate from "./pages/Generate";
import { getModels } from "./api";

export default function App() {
  const [tab, setTab] = useState("visualize");
  const [models, setModels] = useState([]);
  const [family, setFamily] = useState("word");
  const [size, setSize] = useState("small");

  useEffect(() => {
    getModels().then(setModels).catch((e) => console.error("failed to load models:", e));
  }, []);

  const selectedModel = `${size}-${family}`;
  const modelInfo = models.find((m) => m.name === selectedModel);
  const modelExists = !!modelInfo;

  const familiesAvailable = {
    char: models.some((m) => m.name.endsWith("-char")),
    word: models.some((m) => m.name.endsWith("-word")),
  };
  const sizesForFamily = (fam) => models.filter((m) => m.name.endsWith(`-${fam}`)).map((m) => m.name.split("-")[0]);

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

        <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
          <div className="model-toggle">
            {["word", "char"].map((f) => (
              <button
                key={f}
                className={family === f ? "active" : ""}
                disabled={!familiesAvailable[f]}
                style={!familiesAvailable[f] ? { opacity: 0.3, cursor: "default" } : {}}
                onClick={() => setFamily(f)}
              >
                {f === "word" ? "Word" : "Character"}
              </button>
            ))}
          </div>
          <div className="model-toggle" style={{ padding: 3 }}>
            {sizesForFamily(family).map((s) => (
              <button key={s} className={size === s ? "active" : ""} onClick={() => setSize(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minWidth: 0 }}>
        <Sidebar modelInfo={modelInfo} />

        <div className="main" style={{ flex: 1, minWidth: 0 }}>
          {modelExists ? (
            tab === "visualize" ? (
              <Visualize models={models} selectedModel={selectedModel} />
            ) : (
              <Generate selectedModel={selectedModel} />
            )
          ) : (
            <p className="status-text">
              {models.length ? `Model '${selectedModel}' not available yet.` : "Loading models from server..."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}