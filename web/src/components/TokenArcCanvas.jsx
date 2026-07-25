import { useMemo, useRef, useEffect } from "react";

function weightColor(w) {
  const lo = [100, 130, 155];
  const hi = [142, 213, 255];
  const t = Math.min(1, 0.3 + w * 1.1);
  const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function labelFor(tok) {
  if (tok === " ") return "\u00b7";
  if (tok === "\n") return "\u21b5";
  if (tok.startsWith(" ")) return "\u00b7" + tok.slice(1);
  return tok;
}

export default function TokenArcCanvas({ tokens, attention, layer, head, focusIndex, onFocusChange }) {
  const containerRef = useRef(null);

  const labels = useMemo(() => tokens.map(labelFor), [tokens]);

  const isWordLevel = useMemo(() => {
    const avgLen = labels.reduce((a, l) => a + l.length, 0) / (labels.length || 1);
    return avgLen > 1.6;
  }, [labels]);

  const scale = isWordLevel
    ? { charW: 12, minW: 44, pillH: 44, fontSize: 16, gap: 20, pillY: 150, height: 300 }
    : { charW: 9, minW: 32, pillH: 32, fontSize: 13, gap: 14, pillY: 130, height: 260 };

  const widths = useMemo(
    () => labels.map((l) => Math.max(scale.minW, l.length * scale.charW + 24)),
    [labels, scale.minW, scale.charW]
  );

  const positions = useMemo(() => {
    const pos = [];
    let x = 40;
    for (let i = 0; i < widths.length; i++) {
      pos.push(x + widths[i] / 2);
      x += widths[i] + scale.gap;
    }
    return pos;
  }, [widths, scale.gap]);

  const width = positions.length
    ? positions[positions.length - 1] + widths[widths.length - 1] / 2 + 40
    : 600;

  const weights = useMemo(() => {
    if (focusIndex === null || !attention.length) return [];
    return attention[layer][head][focusIndex];
  }, [attention, layer, head, focusIndex]);

  useEffect(() => {
    if (focusIndex === null || !containerRef.current || !positions[focusIndex]) return;
    const el = containerRef.current;
    const target = positions[focusIndex] - el.clientWidth / 2;
    el.scrollLeft = Math.max(0, target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens]);

  return (
    <div
      ref={containerRef}
      style={{ overflowX: "auto", width: "100%", display: "flex", justifyContent: width < 900 ? "center" : "flex-start" }}
    >
      <svg width={width} height={scale.height} style={{ display: "block", flexShrink: 0 }}>
        {focusIndex !== null &&
          weights.map((w, toIdx) => {
            if (toIdx === focusIndex || w < 0.01) return null;
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
                strokeWidth={1 + w * 6}
                strokeLinecap="round"
                opacity={0.6 + w * 0.4}
                d={`M ${x1} ${scale.pillY} Q ${midX} ${scale.pillY - arcHeight} ${x2} ${scale.pillY}`}
              />
            );
          })}

        {labels.map((label, i) => {
          const selected = i === focusIndex;
          const w = focusIndex !== null && weights.length ? weights[i] : 0;
          return (
            <g key={i} onClick={() => onFocusChange(i)} style={{ cursor: "pointer" }}>
              <rect
                x={positions[i] - widths[i] / 2}
                y={scale.pillY - scale.pillH / 2}
                width={widths[i]}
                height={scale.pillH}
                rx={scale.pillH / 2}
                fill={selected ? "#38bdf8" : i !== focusIndex && w > 0.05 ? "rgba(142,213,255,0.15)" : "#1c2b3c"}
                stroke={selected ? "#8ed5ff" : "#3e484f"}
                strokeWidth={selected ? 2 : 1}
              />
              <text
                x={positions[i]}
                y={scale.pillY + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fontFamily="var(--mono)"
                fontSize={scale.fontSize}
                fontWeight={selected ? 700 : 400}
                fill={selected ? "#00354a" : "#d4e4fa"}
              >
                {label}
              </text>
              <text
                x={positions[i]}
                y={scale.pillY + scale.pillH / 2 + 16}
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