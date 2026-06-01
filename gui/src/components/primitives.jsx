/* AI Utopia — shared UI primitives.
   Reveal / Card / SectionTitle / StatusDot / RoleBadge / Avatar / Toggle /
   Slider / Segmented / GhostButton / AnimatedNumber / Field. */

import { useState, useEffect, useRef } from 'react';
import { Icon } from '../lib/icons.jsx';
import { getRoleColor, roleMeta, statusColor } from '../mockData.js';

// Reveal-on-mount wrapper with stagger via CSS var delay
export function Reveal({ children, delay = 0, className = '', style = {}, as = 'div', ...rest }) {
  const Tag = as;
  return (
    <Tag className={`reveal ${className}`} style={{ ...style, animationDelay: `${delay}ms` }} {...rest}>
      {children}
    </Tag>
  );
}

export function Card({ children, className = '', style = {}, span, elevated, hover = true, ...rest }) {
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

export function SectionTitle({ children, right }) {
  return (
    <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
      <h3 className="t-section">{children}</h3>
      {right}
    </div>
  );
}

export function StatusDot({ status, size = 8, pulse = true }) {
  const c = statusColor[status] || '#7A7A99';
  return (
    <span
      className={pulse ? 'pulse-dot-soft' : ''}
      style={{ width: size, height: size, borderRadius: 999, background: c, boxShadow: `0 0 8px ${c}66`, display: 'inline-block', flexShrink: 0 }}
    />
  );
}

export function RoleBadge({ role, withDot = true }) {
  const c = getRoleColor(role);
  return (
    <span className="role-badge" style={{ background: `${c}15`, border: `1px solid ${c}40`, color: c }}>
      {withDot && <span style={{ width: 5, height: 5, borderRadius: 999, background: c, display: 'inline-block' }} />}
      <span style={{ textTransform: 'capitalize' }}>{role}</span>
    </span>
  );
}

// Placeholder avatar — role-colored ring + role glyph (clean, no external images)
export function Avatar({ role, name, size = 36, ring = true }) {
  const c = getRoleColor(role);
  const icon = roleMeta[role].icon;
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
      <Icon name={icon} size={size * 0.42} stroke={1.6} />
    </div>
  );
}

export function Toggle({ value, onChange }) {
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

export function Slider({ value, min, max, step = 1, onChange, format }) {
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

export function Segmented({ options, value, onChange, accent = 'var(--accent-cyan)' }) {
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

export function GhostButton({ children, color = 'var(--accent-cyan)', icon, onClick, full, style = {} }) {
  return (
    <button
      onClick={onClick}
      className="ghost-btn"
      style={{ '--bc': color, color, width: full ? '100%' : 'auto', ...style }}
    >
      {icon && <Icon name={icon} size={13} />}
      {children}
    </button>
  );
}

// Animated number that eases to its target & re-runs when target changes
export function AnimatedNumber({ value, decimals = 0, prefix = '', suffix = '', duration = 900 }) {
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

/* ---------------------------------------------------------------------------
 * Data-state primitives — used by the wired pages to show loading / empty /
 * offline states that look intentional and on-brand rather than raw spinners.
 * ------------------------------------------------------------------------- */

// Small unobtrusive pill shown when the backend is unreachable and a view is
// rendering mock data. Reuses the brand pill look in muted offline-red.
export function OfflinePill({ label = 'BRIDGE OFFLINE · sample data', style = {} }) {
  return (
    <span className="offline-pill" style={style}>
      <span className="offline-pill-dot" />
      <span>{label}</span>
    </span>
  );
}

// Skeleton shimmer block — sized by the caller, rounded, subtle.
export function Skeleton({ width = '100%', height = 14, radius = 6, style = {} }) {
  return <span className="skel" style={{ width, height, borderRadius: radius, ...style }} />;
}

// Centered, low-key empty state for "online but nothing to show yet".
export function EmptyState({ icon, title, hint, action }) {
  return (
    <div className="empty-state">
      {icon && <Icon name={icon} size={26} style={{ color: 'var(--text-tertiary)' }} />}
      <div className="t-body" style={{ color: 'var(--text-secondary)', marginTop: icon ? 10 : 0 }}>{title}</div>
      {hint && <div className="t-caption" style={{ marginTop: 5, maxWidth: 300, textAlign: 'center' }}>{hint}</div>}
      {action && <div style={{ marginTop: 14 }}>{action}</div>}
    </div>
  );
}

// Toast stack — render <Toasts items={...} /> once; push via a small reducer in
// the page. Auto-dismiss is handled by the page (keeps this primitive dumb).
export function Toasts({ items = [], onDismiss }) {
  if (!items.length) return null;
  return (
    <div className="toast-stack">
      {items.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind || 'info'}`} onClick={() => onDismiss && onDismiss(t.id)}>
          <Icon name={t.kind === 'error' ? 'alert' : t.kind === 'success' ? 'check' : 'activity'} size={13} />
          <span className="t-body" style={{ fontSize: 11.5 }}>{t.message}</span>
        </div>
      ))}
    </div>
  );
}

export function Field({ label, description, children, full }) {
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
