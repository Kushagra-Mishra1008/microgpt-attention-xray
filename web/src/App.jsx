import { useState, useEffect } from "react";
import "./theme.css";
import Sidebar from "./components/Sidebar";
import Visualize from "./pages/Visualize";
import Generate from "./pages/Generate";
import ModelInfoPage from "./pages/ModelInfoPage";
import { getModels } from "./api";

export default function App() {
  const [tab, setTab] = useState("visualize");
  const [page, setPage] = useState("app");
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
          ) : modelExists ? (
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