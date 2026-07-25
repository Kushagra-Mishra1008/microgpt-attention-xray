import { useMemo } from "react";

function weightColor(w) {
  const lo = [62, 72, 79];
  const hi = [142, 213, 255];
  const t = Math.min(1, w * 1.4);
  const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export default function TokenArcCanvas({ tokens, attention, layer, head, focusIndex, onFocusChange }) {
  const spacing = 46;
  const pillW = 34;
  const pillY = 130;
  const startX = 40;
  const width = startX * 2 + tokens.length * spacing;
  const height = 260;

  const positions = useMemo(
    () => tokens.map((_, i) => startX + i * spacing),
    [tokens.length]
  );

  const weights = useMemo(() => {
    if (focusIndex === null || !attention.length) return [];
    return attention[layer][head][focusIndex];
  }, [attention, layer, head, focusIndex]);

  return (
    <div style={{ overflowX: "auto", width: "100%" }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        {focusIndex !== null &&
          weights.map((w, toIdx) => {
            if (toIdx === focusIndex || w < 0.015) return null;
            const x1 = positions[toIdx];
            const x2 = positions[focusIndex];
            const dist = Math.abs(focusIndex - toIdx);
            const arcHeight = 18 + dist * 6;
            const midX = (x1 + x2) / 2;
            return (
              <path
                key={toIdx}
                fill="none"
                stroke={weightColor(w)}
                strokeWidth={0.5 + w * 7}
                strokeLinecap="round"
                opacity={0.35 + w * 0.65}
                d={`M ${x1} ${pillY} Q ${midX} ${pillY - arcHeight} ${x2} ${pillY}`}
              />
            );
          })}

        {tokens.map((tok, i) => {
          const selected = i === focusIndex;
          const label = tok === " " ? "\u00b7" : tok === "\n" ? "\u21b5" : tok;
          const w = focusIndex !== null && weights.length ? weights[i] : 0;
          return (
            <g key={i} onClick={() => onFocusChange(i)} style={{ cursor: "pointer" }}>
              <rect
                x={positions[i] - pillW / 2}
                y={pillY - 16}
                width={pillW}
                height={32}
                rx={16}
                fill={selected ? "#38bdf8" : i !== focusIndex && w > 0.05 ? "rgba(142,213,255,0.15)" : "#1c2b3c"}
                stroke={selected ? "#8ed5ff" : "#3e484f"}
                strokeWidth={selected ? 2 : 1}
              />
              <text
                x={positions[i]}
                y={pillY + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fontFamily="var(--mono)"
                fontSize={13}
                fontWeight={selected ? 700 : 400}
                fill={selected ? "#00354a" : "#d4e4fa"}
              >
                {label}
              </text>
              <text
                x={positions[i]}
                y={pillY + 30}
                textAnchor="middle"
                fontFamily="var(--mono)"
                fontSize={9}
                fill="#5a6a75"
              >
                {i}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}