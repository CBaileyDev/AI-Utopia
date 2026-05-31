/* AI Utopia — icon set (lucide paths, MIT). window.Icon */
const ICON_PATHS = {
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
  brain: '<path d="M12 5a3 3 0 1 0-5.997.142 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.142 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
  database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
  clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
  up: '<path d="m6 9 6-6 6 6"/><path d="M6 9v6a6 6 0 0 0 6 6"/>',
  down: '<path d="m18 15-6 6-6-6"/><path d="M18 15V9a6 6 0 0 0-6-6"/>',
  trendUp: '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
  trendDown: '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
  minus: '<path d="M5 12h14"/>',
  axe: '<path d="m14 12-8.5 8.5a2.12 2.12 0 1 1-3-3L11 9"/><path d="M15 13 9 7l4-4 6 6h3a8 8 0 0 1-7 7z"/>',
  hammer: '<path d="m15 12-8.373 8.373a1 1 0 1 1-3-3L12 9"/><path d="m18 15 4-4"/><path d="m21.5 11.5-1.914-1.914A2 2 0 0 1 19 8.172V7l-2.26-2.26a6 6 0 0 0-4.202-1.756L9 2.96l.92.82A6.18 6.18 0 0 1 12 8.4V10l2 2h1.172a2 2 0 0 1 1.414.586z"/>',
  wheat: '<path d="M2 22 16 8"/><path d="M3.47 12.53 5 11l1.53 1.53a3.5 3.5 0 0 1 0 4.94L5 19l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/><path d="M7.47 8.53 9 7l1.53 1.53a3.5 3.5 0 0 1 0 4.94L9 15l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/><path d="M11.47 4.53 13 3l1.53 1.53a3.5 3.5 0 0 1 0 4.94L13 11l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z"/><path d="M20 2h2v2a4 4 0 0 1-4 4h-2V6a4 4 0 0 1 4-4Z"/><path d="M11.47 17.47 13 19l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L5 19l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/><path d="M15.47 13.47 17 15l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L9 15l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/><path d="M19.47 9.47 21 11l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L13 11l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z"/>',
  shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
  userX: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="17" x2="22" y1="8" y2="13"/><line x1="22" x2="17" y1="8" y2="13"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  loader: '<line x1="12" x2="12" y1="2" y2="6"/><line x1="12" x2="12" y1="18" y2="22"/><line x1="4.93" x2="7.76" y1="4.93" y2="7.76"/><line x1="16.24" x2="19.07" y1="16.24" y2="19.07"/><line x1="2" x2="6" y1="12" y2="12"/><line x1="18" x2="22" y1="12" y2="12"/><line x1="4.93" x2="7.76" y1="19.07" y2="16.24"/><line x1="16.24" x2="19.07" y1="7.76" y2="4.93"/>',
  circle: '<circle cx="12" cy="12" r="10"/>',
  pause: '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  save: '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
  trophy: '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
  stop: '<path d="m2.45 7.4 5.18-5.18a1 1 0 0 1 .71-.29h7.39a1 1 0 0 1 .71.29l5.18 5.18a1 1 0 0 1 .29.71v7.39a1 1 0 0 1-.29.71l-5.18 5.18a1 1 0 0 1-.71.29H8.34a1 1 0 0 1-.71-.29L2.45 16.5a1 1 0 0 1-.29-.71V8.11a1 1 0 0 1 .29-.71z"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
  zoomIn: '<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="11" x2="11" y1="8" y2="14"/><line x1="8" x2="14" y1="11" y2="11"/>',
  zoomOut: '<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/><line x1="8" x2="14" y1="11" y2="11"/>',
  grid: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
  send: '<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"/><path d="m21.854 2.147-10.94 10.939"/>',
  message: '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
  server: '<rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
  cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
  folder: '<path d="M6 14a2 2 0 0 0 2-2V8a2 2 0 0 1 2-2h2.343a2 2 0 0 1 1.414.586l.828.828A2 2 0 0 0 16 8h2a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2z"/><path d="M2 8a2 2 0 0 1 2-2h2.343a2 2 0 0 1 1.414.586l.828.828A2 2 0 0 0 10 8h6"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>',
  gear: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  activity: '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
  link: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
  sparkles: '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
  map: '<path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/>',
  chevron: '<path d="m9 18 6-6-6-6"/>',
  refresh: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
  layers: '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>',
  target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  pulse: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  network: '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/>',
};

function Icon({ name, size = 16, stroke = 1.5, fill = 'none', style = {}, className = '' }) {
  const d = ICON_PATHS[name];
  if (!d) return null;
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg" width={size} height={size}
      viewBox="0 0 24 24" fill={fill} stroke="currentColor"
      strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round"
      className={className} style={style}
      dangerouslySetInnerHTML={{ __html: d }}
    />
  );
}

window.Icon = Icon;


/* AI Utopia — lightweight SVG charts. window.AreaStack, Donut, LineChart, Sparkline */

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
function AreaStack({ data, keys, colors, height = 200 }) {
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
function Donut({ data, size = 150, thickness = 22 }) {
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
function LineChart({ data, height = 280, color = '#00E5CC', baseline }) {
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

function Sparkline({ data, width = 84, height = 30, color = '#00E5CC' }) {
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

Object.assign(window, { AreaStack, Donut, LineChart, Sparkline });


/* AI Utopia — shared UI primitives */
const { useState, useEffect, useRef } = React;

// Reveal-on-mount wrapper with stagger via CSS var delay
function Reveal({ children, delay = 0, className = '', style = {}, as = 'div', ...rest }) {
  const Tag = as;
  return (
    <Tag className={`reveal ${className}`} style={{ ...style, animationDelay: `${delay}ms` }} {...rest}>
      {children}
    </Tag>
  );
}

function Card({ children, className = '', style = {}, span, elevated, hover = true, ...rest }) {
  return (
    <div
      className={`card ${elevated ? 'card-elevated' : ''} ${hover ? 'card-hover' : ''} ${className}`}
      style={{ ...(span ? { gridColumn: `span ${span}` } : {}), ...style }}
      {...rest}
    >
      {children}
    </div>
  );
}

function SectionTitle({ children, right }) {
  return (
    <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
      <h3 className="t-section">{children}</h3>
      {right}
    </div>
  );
}

function StatusDot({ status, size = 8, pulse = true }) {
  const c = window.AIU.statusColor[status] || '#7A7A99';
  return (
    <span
      className={pulse ? 'pulse-dot-soft' : ''}
      style={{ width: size, height: size, borderRadius: 999, background: c, boxShadow: `0 0 8px ${c}66`, display: 'inline-block', flexShrink: 0 }}
    />
  );
}

function RoleBadge({ role, withDot = true }) {
  const c = window.AIU.getRoleColor(role);
  return (
    <span className="role-badge" style={{ background: `${c}15`, border: `1px solid ${c}40`, color: c }}>
      {withDot && <span style={{ width: 5, height: 5, borderRadius: 999, background: c, display: 'inline-block' }} />}
      <span style={{ textTransform: 'capitalize' }}>{role}</span>
    </span>
  );
}

// Placeholder avatar — role-colored ring + role glyph (clean, no external images)
function Avatar({ role, name, size = 36, ring = true }) {
  const c = window.AIU.getRoleColor(role);
  const icon = window.AIU.roleMeta[role].icon;
  return (
    <div
      style={{
        width: size, height: size, borderRadius: 999, flexShrink: 0,
        border: ring ? `2px solid ${c}` : 'none',
        boxShadow: ring ? `0 0 12px ${c}22, inset 0 0 12px ${c}10` : 'none',
        background: `radial-gradient(circle at 30% 25%, ${c}26, ${c}08 60%, #0c0c14)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', color: c,
        position: 'relative', overflow: 'hidden',
      }}
      title={name}
    >
      <window.Icon name={icon} size={size * 0.42} stroke={1.6} />
    </div>
  );
}

function Toggle({ value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="toggle"
      style={{ background: value ? 'var(--accent-cyan)' : 'var(--border-subtle)', boxShadow: value ? '0 0 10px rgba(0,229,204,0.35)' : 'none' }}
      aria-pressed={value}
    >
      <span className="toggle-knob" style={{ left: value ? 16 : 2 }} />
    </button>
  );
}

function Slider({ value, min, max, step = 1, onChange, format }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex items-center" style={{ gap: 12, width: '100%' }}>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="range"
        style={{ background: `linear-gradient(to right, var(--accent-cyan) 0%, var(--accent-cyan) ${pct}%, var(--border-subtle) ${pct}%, var(--border-subtle) 100%)` }}
      />
      <span className="t-data" style={{ fontSize: 12, minWidth: 34, textAlign: 'right' }}>
        {format ? format(value) : value}
      </span>
    </div>
  );
}

function Segmented({ options, value, onChange, accent = 'var(--accent-cyan)' }) {
  return (
    <div className="segmented">
      {options.map((o) => {
        const v = typeof o === 'object' ? o.value : o;
        const label = typeof o === 'object' ? o.label : o;
        const active = v === value;
        return (
          <button
            key={v} onClick={() => onChange(v)} className="segmented-btn"
            style={{
              background: active ? accent : 'transparent',
              color: active ? 'var(--void)' : 'var(--text-secondary)',
              fontWeight: active ? 600 : 400,
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

function GhostButton({ children, color = 'var(--accent-cyan)', icon, onClick, full, style = {} }) {
  return (
    <button
      onClick={onClick}
      className="ghost-btn"
      style={{ '--bc': color, color, width: full ? '100%' : 'auto', ...style }}
    >
      {icon && <window.Icon name={icon} size={13} />}
      {children}
    </button>
  );
}

// Animated number that eases to its target & re-runs when target changes
function AnimatedNumber({ value, decimals = 0, prefix = '', suffix = '', duration = 900 }) {
  const [display, setDisplay] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    const start = Date.now();
    const a = from.current, b = value;
    const id = setInterval(() => {
      const t = Math.min((Date.now() - start) / duration, 1);
      const e = 1 - Math.pow(1 - t, 3);
      setDisplay(a + (b - a) * e);
      if (t >= 1) { clearInterval(id); from.current = b; }
    }, 16);
    return () => clearInterval(id);
  }, [value]);
  return <span>{prefix}{display.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

function Field({ label, description, children, full }) {
  return (
    <div style={{ padding: '12px 0', ...(full ? { gridColumn: '1 / -1' } : {}) }}>
      <div className="flex items-center justify-between" style={{ gap: 16 }}>
        <div>
          <label className="t-label">{label}</label>
          {description && <p className="t-caption" style={{ marginTop: 3 }}>{description}</p>}
        </div>
        <div style={{ flexShrink: 0 }}>{children}</div>
      </div>
    </div>
  );
}

Object.assign(window, {
  Reveal, Card, SectionTitle, StatusDot, RoleBadge, Avatar, Toggle, Slider,
  Segmented, GhostButton, AnimatedNumber, Field,
});


