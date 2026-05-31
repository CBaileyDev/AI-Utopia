/* AI Utopia — App shell: window chrome + horizontal tabs + status bar.
   Plain useState tab-switching (no router) — matches the prototype. */
import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { Icon } from './lib/icons.jsx';
import { winMinimize, winToggleMaximize, winClose } from './lib/tauri.js';
import { StatusDot } from './components/primitives.jsx';
import Dashboard from './pages/Dashboard.jsx';
import BotConfig from './pages/BotConfig.jsx';
import Training from './pages/Training.jsx';
import Spectate from './pages/Spectate.jsx';
import Settings from './pages/Settings.jsx';

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: 'activity' },
  { id: 'botconfig', label: 'Bot Config', icon: 'cpu' },
  { id: 'training', label: 'Training', icon: 'pulse' },
  { id: 'spectate', label: 'Spectate', icon: 'map' },
  { id: 'settings', label: 'Settings', icon: 'gear' },
];

const SUBTITLES = {
  dashboard: 'Village Overview · 4 Agents Active',
  botconfig: 'Agent Management · Spawn & Configure',
  training: 'RL Pipeline · Epoch 1,247',
  spectate: 'Live World View · Agent Observation',
  settings: 'System Configuration',
};

function Clock() {
  const [t, setT] = useState(new Date());
  useEffect(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  return <span className="t-data mono" style={{ fontSize: 15, color: 'var(--text-primary)' }}>{t.toLocaleTimeString('en-US', { hour12: false })}</span>;
}

function TabBar({ active, setActive }) {
  const refs = useRef({});
  const [ind, setInd] = useState({ left: 0, width: 0 });
  useLayoutEffect(() => {
    const el = refs.current[active];
    if (el) setInd({ left: el.offsetLeft, width: el.offsetWidth });
  }, [active]);
  useEffect(() => {
    const onResize = () => { const el = refs.current[active]; if (el) setInd({ left: el.offsetLeft, width: el.offsetWidth }); };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [active]);
  return (
    <div className="tabbar">
      {TABS.map((t) => (
        <button key={t.id} ref={(el) => (refs.current[t.id] = el)} onClick={() => setActive(t.id)} className={`tab ${active === t.id ? 'active' : ''}`}>
          <Icon name={t.icon} size={15} className="tab-ico" />
          {t.label}
        </button>
      ))}
      <span className="tab-glow" style={{ left: ind.left, width: ind.width }} />
      <span className="tab-indicator" style={{ left: ind.left + 14, width: Math.max(ind.width - 28, 10) }} />
    </div>
  );
}

export default function App() {
  const [active, setActive] = useState('dashboard');
  const [selectedAgentId, setSelectedAgentId] = useState('1');

  const openAgent = (id) => { setSelectedAgentId(id); setActive('spectate'); };

  const pages = {
    dashboard: <Dashboard onOpenAgent={openAgent} />,
    botconfig: <BotConfig />,
    training: <Training />,
    spectate: <Spectate selectedAgentId={selectedAgentId} setSelectedAgentId={setSelectedAgentId} />,
    settings: <Settings />,
  };

  return (
    <div className="desktop">
      <div className="app-window">
        {/* Title bar — drag handle for the borderless OS window.
            `data-tauri-drag-region` applies only to elements that carry the
            attribute, so the brand / pill / clock / buttons below (which lack it)
            stay interactive and do NOT start a window drag. */}
        <div className="titlebar" data-tauri-drag-region>
          <div className="brand">
            <div className="diamond"><span>Ai</span></div>
            <div>
              <div className="wordmark">AI <b>UTOPIA</b></div>
              <div className="brand-sub">Multi-Agent Village</div>
            </div>
          </div>
          {/* stopPropagation on mousedown so clicking the pill / clock / window
              buttons never starts a window drag, regardless of whether Tauri's
              drag-region matcher is exact-target or closest()-based. No-op for
              onClick; just suppresses drag-start on these interactive children. */}
          <div className="win-ctrls" onMouseDown={(e) => e.stopPropagation()}>
            <div className="win-meta">
              <div className="bridge-pill">
                <StatusDot status="alive" size={6} />
                <span className="lbl">BRIDGE ONLINE</span>
              </div>
              <Clock />
            </div>
            <button className="win-btn" title="Minimize" onClick={winMinimize}><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4"><line x1="3" y1="9" x2="11" y2="9" /></svg></button>
            <button className="win-btn" title="Maximize" onClick={winToggleMaximize}><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3"><rect x="3.2" y="3.2" width="7.6" height="7.6" rx="1.4" /></svg></button>
            <button className="win-btn close" title="Close" onClick={winClose}><svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4"><line x1="3.5" y1="3.5" x2="10.5" y2="10.5" /><line x1="10.5" y1="3.5" x2="3.5" y2="10.5" /></svg></button>
          </div>
        </div>

        {/* Tab bar */}
        <TabBar active={active} setActive={setActive} />

        {/* Content */}
        <div className="content">
          <div className="content-inner">
            <div className="flex items-end justify-between" style={{ marginBottom: 22 }}>
              <div>
                <h1 className="t-hero">{TABS.find((t) => t.id === active).label}</h1>
                <p className="t-caption" style={{ marginTop: 5, letterSpacing: '0.04em' }}>{SUBTITLES[active]}</p>
              </div>
            </div>
            {/* key={active} remounts the page so the tab-fade animation replays on switch */}
            <div className="tab-fade" key={active}>{pages[active]}</div>
          </div>
        </div>

        {/* Status bar */}
        <div className="statusbar">
          <span className="si"><StatusDot status="alive" size={6} pulse={false} /> Py4J 25099</span>
          <span className="si" style={{ color: 'var(--text-tertiary)' }}>v21 · iter 4/200</span>
          <span className="si" style={{ color: 'var(--status-training)' }}><Icon name="loader" size={10} className="spin" /> training</span>
          <span className="spacer" />
          <span className="si">RTX 4080 · 84% idle</span>
          <span className="si">MC 1.21.1 · Fabric</span>
          <span className="si" style={{ color: 'var(--accent-cyan)' }}>● 4 instances</span>
        </div>
      </div>
    </div>
  );
}
