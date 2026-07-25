export default function Sidebar({ modelInfo }) {
  return (
    <aside
      style={{
        width: 240,
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        padding: "24px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <div>
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

      <div style={{ borderTop: "1px solid rgba(62,72,79,0.4)", paddingTop: 16 }}>
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