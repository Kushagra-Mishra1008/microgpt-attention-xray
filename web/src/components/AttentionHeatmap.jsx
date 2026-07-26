import { useMemo } from "react";

function cellColor(w) {
  const lo = [13, 28, 45];
  const hi = [142, 213, 255];
  const t = Math.min(1, w * 1.6);
  const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export default function AttentionHeatmap({ tokens, attention, layer, head }) {
  const matrix = attention[layer][head];

  const stats = useMemo(() => {
    let max = 0;
    let entropySum = 0;
    let n = 0;
    for (const row of matrix) {
      let rowMax = Math.max(...row);
      if (rowMax > max) max = rowMax;
      const rowEntropy = -row.reduce((acc, p) => (p > 0 ? acc + p * Math.log2(p) : acc), 0);
      entropySum += rowEntropy;
      n++;
    }
    return { max, avgEntropy: n ? entropySum / n : 0 };
  }, [matrix]);

  const T = tokens.length;
  const cell = 12;
  const size = T * cell;
  const PANEL_SIZE = 200;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0, width: "100%" }}>
      <div>
        <h3 className="status-text" style={{ textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>
          Head map
        </h3>
        <svg
          viewBox={`0 0 ${size} ${size}`}
          width={PANEL_SIZE}
          height={PANEL_SIZE}
          preserveAspectRatio="xMidYMid meet"
          style={{ display: "block", border: "1px solid rgba(62,72,79,0.4)", borderRadius: 6 }}
        >
          {matrix.map((row, i) =>
            row.map((w, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * cell}
                y={i * cell}
                width={cell}
                height={cell}
                fill={j > i ? "#051424" : cellColor(w)}
              />
            ))
          )}
        </svg>
      </div>

      <div style={{ padding: 12, background: "var(--surface-highest)", borderRadius: 8, border: "1px solid rgba(62,72,79,0.5)" }}>
        <div className="status-text" style={{ fontSize: 10, textTransform: "uppercase", marginBottom: 4 }}>Max weight</div>
        <div style={{ fontFamily: "var(--mono)", color: "var(--primary)" }}>{stats.max.toFixed(4)}</div>
      </div>
      <div style={{ padding: 12, background: "var(--surface-highest)", borderRadius: 8, border: "1px solid rgba(62,72,79,0.5)" }}>
        <div className="status-text" style={{ fontSize: 10, textTransform: "uppercase", marginBottom: 4 }}>Avg entropy</div>
        <div style={{ fontFamily: "var(--mono)" }}>{stats.avgEntropy.toFixed(2)} bits</div>
      </div>
    </div>
  );
}