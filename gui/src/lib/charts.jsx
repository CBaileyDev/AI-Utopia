/* AI Utopia — lightweight SVG charts. AreaStack, Donut, LineChart, Sparkline. */

function buildPath(points) {
  if (!points.length) return '';
  // smooth catmull-rom -> bezier
  const p = points;
  let d = `M ${p[0].x} ${p[0].y}`;
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[i - 1] || p[i];
    const p1 = p[i];
    const p2 = p[i + 1];
    const p3 = p[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2.x} ${p2.y}`;
  }
  return d;
}

// Stacked area chart for agent activity (4 roles)
export function AreaStack({ data, keys, colors, height = 200 }) {
  const W = 760, H = height, padB = 22, padT = 10;
  const innerH = H - padB - padT;
  const n = data.length;
  const totals = data.map((d) => keys.reduce((s, k) => s + d[k], 0));
  const max = Math.max(...totals) * 1.1;
  const xAt = (i) => (i / (n - 1)) * W;
  const yAt = (v) => padT + innerH - (v / max) * innerH;

  // build cumulative bands
  let cum = data.map(() => 0);
  const bands = keys.map((k) => {
    const lower = cum.map((c) => c);
    cum = cum.map((c, i) => c + data[i][k]);
    const upper = cum.map((c) => c);
    const top = upper.map((v, i) => ({ x: xAt(i), y: yAt(v) }));
    const bot = lower.map((v, i) => ({ x: xAt(i), y: yAt(v) }));
    const topPath = buildPath(top);
    const botPath = buildPath([...bot].reverse());
    const area = `${topPath} L ${bot[bot.length - 1].x} ${bot[bot.length - 1].y} ` +
      botPath.replace('M', 'L') + ' Z';
    return { key: k, area, line: topPath };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        {keys.map((k) => (
          <linearGradient key={k} id={`grad-${k}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={colors[k]} stopOpacity="0.35" />
            <stop offset="100%" stopColor={colors[k]} stopOpacity="0.02" />
          </linearGradient>
        ))}
      </defs>
      {bands.map((b) => (
        <g key={b.key} className="area-band">
          <path d={b.area} fill={`url(#grad-${b.key})`} />
          <path d={b.line} fill="none" stroke={colors[b.key]} strokeWidth="1.5" opacity="0.85" />
        </g>
      ))}
      {data.map((d, i) => (i % 2 === 0 ? (
        <text key={i} x={xAt(i)} y={H - 6} fontSize="9" fill="#4A4A66" textAnchor="middle" fontFamily="'Geist Mono', monospace">{d.time}</text>
      ) : null))}
    </svg>
  );
}

// Donut chart
export function Donut({ data, size = 150, thickness = 22 }) {
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const r = (size - thickness) / 2;
  const cx = size / 2, cy = size / 2;
  const C = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#181824" strokeWidth={thickness} />
      {data.map((d, i) => {
        const frac = d.value / total;
        const len = frac * C;
        const gap = C - len;
        const el = (
          <circle
            key={i} cx={cx} cy={cy} r={r} fill="none"
            stroke={d.color} strokeWidth={thickness}
            strokeDasharray={`${Math.max(len - 3, 0)} ${gap + 3}`}
            strokeDashoffset={-offset} strokeLinecap="round"
            style={{ transition: 'stroke-dasharray 0.9s cubic-bezier(0.16,1,0.3,1)' }}
          />
        );
        offset += len;
        return el;
      })}
    </svg>
  );
}

// Line chart with gradient fill + dashed baseline
export function LineChart({ data, height = 280, color = '#00E5CC', baseline }) {
  const W = 720, H = height, padB = 24, padT = 12, padL = 4, padR = 4;
  const innerH = H - padB - padT;
  const innerW = W - padL - padR;
  const vals = data.map((d) => d.reward);
  const min = Math.min(...vals) * 0.95;
  const max = Math.max(...vals) * 1.05;
  const n = data.length;
  const xAt = (i) => padL + (i / (n - 1)) * innerW;
  const yAt = (v) => padT + innerH - ((v - min) / (max - min)) * innerH;
  const pts = data.map((d, i) => ({ x: xAt(i), y: yAt(d.reward) }));
  const line = buildPath(pts);
  const area = `${line} L ${pts[pts.length - 1].x} ${padT + innerH} L ${pts[0].x} ${padT + innerH} Z`;
  const baseY = baseline != null ? yAt(baseline) : null;
  const gridLines = 4;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <linearGradient id="lc-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {Array.from({ length: gridLines + 1 }, (_, i) => {
        const y = padT + (i / gridLines) * innerH;
        return <line key={i} x1={padL} y1={y} x2={W - padR} y2={y} stroke="#1a1a28" strokeWidth="1" strokeDasharray="2 4" />;
      })}
      <path d={area} fill="url(#lc-fill)" />
      {baseY != null && (
        <line x1={padL} y1={baseY} x2={W - padR} y2={baseY} stroke="#4A4A66" strokeWidth="1" strokeDasharray="5 5" />
      )}
      <path d={line} fill="none" stroke={color} strokeWidth="2" className="line-draw" />
      <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r="3.5" fill={color} className="pulse-dot" />
      {data.map((d, i) => (i % Math.ceil(n / 6) === 0 ? (
        <text key={i} x={xAt(i)} y={H - 7} fontSize="9" fill="#4A4A66" textAnchor="middle" fontFamily="'Geist Mono', monospace">{d.epoch}</text>
      ) : null))}
    </svg>
  );
}

export function Sparkline({ data, width = 84, height = 30, color = '#00E5CC' }) {
  const max = Math.max(...data), min = Math.min(...data);
  const pts = data.map((v, i) => ({
    x: (i / (data.length - 1)) * width,
    y: height - ((v - min) / ((max - min) || 1)) * (height - 4) - 2,
  }));
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <path d={buildPath(pts)} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}
