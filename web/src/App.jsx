import { useState, useEffect } from "react";
import "./theme.css";
import Sidebar from "./components/Sidebar";
import Visualize from "./pages/Visualize";
import Generate from "./pages/Generate";
import ModelInfoPage from "./pages/ModelInfoPage";
import { getModels, prepareModel } from "./api";

export default function App() {
  const [tab, setTab] = useState("visualize");
  const [page, setPage] = useState("app");
  const [models, setModels] = useState([]);
  const [family, setFamily] = useState("char");
  const [size, setSize] = useState("small");

  const selectedModel = `${size}-${family}`;

  useEffect(() => {
    let cancelled = false;
    function poll() {
      getModels()
        .then((list) => { if (!cancelled) setModels(list); })
        .catch((e) => console.error("failed to load models:", e));
    }
    poll();
    const interval = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  useEffect(() => {
    prepareModel(selectedModel);
  }, [selectedModel]);

  const modelInfo = models.find((m) => m.name === selectedModel);
  const modelExists = !!modelInfo;
  const modelReady = modelInfo?.ready ?? false;

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
            <button
              className={page === "app" && tab === "visualize" ? "active" : ""}
              onClick={() => { setPage("app"); setTab("visualize"); }}
            >
              Visualize
            </button>
            <button
              className={page === "app" && tab === "generate" ? "active" : ""}
              onClick={() => { setPage("app"); setTab("generate"); }}
            >
              Generate
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <Sidebar
          modelInfo={modelInfo}
          family={family}
          setFamily={setFamily}
          size={size}
          setSize={setSize}
          familiesAvailable={familiesAvailable}
          sizesForFamily={sizesForFamily}
          page={page}
          onNavigate={setPage}
        />

        <div className="main" style={{ flex: 1, minWidth: 0, overflowY: "auto" }}>
          {page === "info" ? (
            <ModelInfoPage modelInfo={modelInfo} />
          ) : !modelExists ? (
            <p className="status-text">
              {models.length ? `Model '${selectedModel}' not available.` : "Loading models from server..."}
            </p>
          ) : !modelReady ? (
            <div className="canvas" style={{ alignItems: "center", textAlign: "center", gap: 12 }}>
              <div className="stream-dot" style={{ margin: "0 auto" }}></div>
              <p style={{ margin: 0 }}>Preparing this model on the server (first time only)...</p>
              <p className="status-text" style={{ margin: 0 }}>
                This can take a minute the first time each model is used. Checking again automatically.
              </p>
            </div>
          ) : tab === "visualize" ? (
            <Visualize models={models} selectedModel={selectedModel} />
          ) : (
            <Generate selectedModel={selectedModel} />
          )}
        </div>
      </div>
    </div>
  );
}