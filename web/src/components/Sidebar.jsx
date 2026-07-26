export default function Sidebar({
  modelInfo,
  family,
  setFamily,
  size,
  setSize,
  familiesAvailable,
  sizesForFamily,
  page,
  onNavigate,
}) {
  return (
    <aside
      style={{
        width: 260,
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        padding: "24px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 20,
        overflowY: "auto",
      }}
    >
      <div>
        <div
          className="status-text"
          style={{ textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10, fontSize: 11 }}
        >
          Model
        </div>
        <div className="model-toggle" style={{ width: "100%", marginBottom: 8 }}>
          {["word", "char"].map((f) => (
            <button
              key={f}
              style={{ flex: 1, ...(!familiesAvailable[f] ? { opacity: 0.3, cursor: "default" } : {}) }}
              className={family === f ? "active" : ""}
              disabled={!familiesAvailable[f]}
              onClick={() => setFamily(f)}
            >
              {f === "word" ? "Word" : "Character"}
            </button>
          ))}
        </div>
        <div className="model-toggle" style={{ width: "100%", padding: 3 }}>
          {sizesForFamily(family).map((s) => (
            <button key={s} style={{ flex: 1 }} className={size === s ? "active" : ""} onClick={() => setSize(s)}>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div style={{ borderTop: "1px solid rgba(62,72,79,0.4)", paddingTop: 16 }}>
        <div
          className="status-text"
          style={{ textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12, fontSize: 11 }}
        >
          Transformer config
        </div>

        {modelInfo ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <ConfigRow label="Model" value={modelInfo.name} highlight />
            <ConfigRow label="Tokenizer" value={modelInfo.tokenizer_type === "bpe" ? "BPE (word)" : "Character"} />
            <ConfigRow label="Embed dim" value={modelInfo.n_embd} />
            <ConfigRow label="Layers" value={modelInfo.n_layer} />
            <ConfigRow label="Heads" value={modelInfo.n_head} />
            <ConfigRow label="Context" value={modelInfo.block_size} />
            <ConfigRow label="Val loss" value={modelInfo.val_loss.toFixed(4)} highlight />
          </div>
        ) : (
          <p className="status-text">No model selected</p>
        )}
      </div>

      <button
        onClick={() => onNavigate(page === "info" ? "app" : "info")}
        style={{
          width: "100%",
          background: page === "info" ? "var(--primary-container)" : "var(--surface-highest)",
          color: page === "info" ? "var(--on-primary-container)" : "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          padding: "10px 0",
          fontFamily: "var(--mono)",
          fontSize: 12,
          fontWeight: 700,
          cursor: "pointer",
        }}
      >
        {page === "info" ? "\u2190 Back to app" : "Model details \u2192"}
      </button>

      {page !== "info" && (
        <div style={{ borderTop: "1px solid rgba(62,72,79,0.4)", paddingTop: 16, marginTop: "auto" }}>
          <div
            className="status-text"
            style={{ textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, fontSize: 11 }}
          >
            About
          </div>
          <p className="status-text" style={{ lineHeight: 1.6 }}>
            A GPT-style transformer built from scratch in PyTorch, trained on Tiny
            Shakespeare. Switch between character and word-level tokenization to
            compare how attention patterns differ.
          </p>
        </div>
      )}
    </aside>
  );
}

function ConfigRow({ label, value, highlight }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span className="status-text" style={{ fontSize: 12 }}>{label}</span>
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: 13,
          color: highlight ? "var(--primary)" : "var(--text)",
          fontWeight: highlight ? 700 : 400,
        }}
      >
        {value}
      </span>
    </div>
  );
}