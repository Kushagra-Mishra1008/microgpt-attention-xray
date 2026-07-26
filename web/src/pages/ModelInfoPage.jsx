import { useState, useEffect } from "react";
import { getCorpus } from "../api";

export default function ModelInfoPage({ modelInfo }) {
  const [corpus, setCorpus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getCorpus()
      .then(setCorpus)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div className="canvas" style={{ minHeight: "auto" }}>
        <h2 style={{ margin: "0 0 12px", fontSize: 18 }}>
          {modelInfo ? modelInfo.name : "No model selected"}
        </h2>
        {modelInfo ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.6, margin: 0 }}>
              A decoder-only transformer trained from scratch in PyTorch on the Tiny
              Shakespeare corpus. This checkpoint uses{" "}
              {modelInfo.tokenizer_type === "bpe" ? "byte-pair encoding (word/subword-level)" : "character-level"}{" "}
              tokenization, with {modelInfo.n_layer} transformer blocks, {modelInfo.n_head} attention
              heads per layer, and a {modelInfo.n_embd}-dimensional embedding space. It reached a
              validation loss of {modelInfo.val_loss.toFixed(4)} during training, and can attend
              to up to {modelInfo.block_size} tokens of context at once.
            </p>
          </div>
        ) : (
          <p className="status-text">Select a model to see its details.</p>
        )}
      </div>

      <div className="canvas" style={{ minHeight: "auto" }}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15 }}>Training corpus</h3>
        {loading && <p className="status-text">Loading corpus...</p>}
        {error && <p className="status-text" style={{ color: "#e24b4a" }}>{error}</p>}
        {corpus && (
          <>
            <p className="status-text" style={{ marginBottom: 12 }}>
              {corpus.source} — {corpus.length.toLocaleString()} characters
            </p>
            <pre
              style={{
                background: "#010f1f",
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: 16,
                maxHeight: 420,
                overflowY: "auto",
                fontFamily: "var(--mono)",
                fontSize: 13,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
                margin: 0,
              }}
            >
              {corpus.text}
            </pre>
          </>
        )}
      </div>
    </div>
  );
}