/* AI Utopia — Spectate tab */
import { useState, useRef, useEffect } from 'react';
import { Card, Reveal, SectionTitle, Avatar, RoleBadge, StatusDot, EmptyState } from '../components/primitives.jsx';
import { Icon } from '../lib/icons.jsx';
import { getRoleColor } from '../mockData.js';

function WorldViewport({ agents }) {
  const ref = useRef(null);
  const posRef = useRef(agents.map((a, i) => ({
    role: a.role, name: a.name,
    x: 140 + i * 150 + a.x * 0.3, y: 120 + (i % 2) * 140 + a.z * 0.2,
    vx: (Math.sin(i * 9)) * 0.35, vy: (Math.cos(i * 7)) * 0.35, trail: [],
  })));
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const draw = () => {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#07070d'; ctx.fillRect(0, 0, w, h);
      // grid
      ctx.strokeStyle = 'rgba(34,34,51,0.4)'; ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += 38) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 38) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      // center village marker
      ctx.strokeStyle = 'rgba(0,229,204,0.25)'; ctx.setLineDash([4, 6]);
      ctx.beginPath(); ctx.arc(w / 2, h / 2, 60, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      posRef.current.forEach((p) => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 26 || p.x > w - 26) p.vx *= -1;
        if (p.y < 26 || p.y > h - 26) p.vy *= -1;
        p.x = Math.max(26, Math.min(w - 26, p.x)); p.y = Math.max(26, Math.min(h - 26, p.y));
        p.trail.push({ x: p.x, y: p.y }); if (p.trail.length > 34) p.trail.shift();
        const c = getRoleColor(p.role);
        ctx.strokeStyle = c + '33'; ctx.lineWidth = 1.2; ctx.beginPath();
        p.trail.forEach((t, j) => j === 0 ? ctx.moveTo(t.x, t.y) : ctx.lineTo(t.x, t.y)); ctx.stroke();
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 22);
        g.addColorStop(0, c + '55'); g.addColorStop(1, 'transparent');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, 22, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#E8E8F0'; ctx.font = "500 11px 'Geist Mono', monospace"; ctx.textAlign = 'center';
        ctx.fillText(p.name, p.x, p.y - 13);
      });
    };
    const id = setInterval(draw, 33);
    return () => clearInterval(id);
  }, []);
  return (
    <Reveal style={{ gridColumn: 'span 8' }}>
      <Card style={{ overflow: 'hidden', position: 'relative', aspectRatio: '16/9' }}>
        <canvas ref={ref} width={880} height={495} style={{ width: '100%', height: '100%', display: 'block' }} />
        <div style={{ position: 'absolute', top: 14, left: 14 }}>
          <div className="flex items-center" style={{ gap: 7 }}><StatusDot status="alive" size={6} /><span className="t-caption" style={{ color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>OVERWORLD · DAY 1,247 · 14:32</span></div>
          <div className="t-caption mono" style={{ marginTop: 4 }}>spawn (64.5, 66, -47.5)</div>
        </div>
        <div className="flex items-center" style={{ position: 'absolute', bottom: 14, right: 14, gap: 6 }}>
          {['zoomIn', 'zoomOut', 'grid'].map((n) => (
            <button key={n} className="flex items-center justify-center" style={{ width: 30, height: 30, borderRadius: 8, cursor: 'pointer', background: 'rgba(8,8,15,0.8)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', transition: 'color 0.2s' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}>
              <Icon name={n} size={13} />
            </button>
          ))}
        </div>
      </Card>
    </Reveal>
  );
}

function Inspector({ agent }) {
  const [tab, setTab] = useState('state');
  const tabs = [['state', 'State'], ['memory', 'Memory'], ['plan', 'Plan']];
  const memory = [
    { time: '14:30', content: 'Observed oak tree cluster at (142, 64, -89)', imp: 0.7 },
    { time: '14:25', content: 'Chopped 12 oak_log successfully', imp: 0.9 },
    { time: '14:20', content: 'Navigated to village stockpile', imp: 0.5 },
    { time: '14:15', content: 'Deposited resources in chest', imp: 0.8 },
  ];
  const plan = [
    { s: 'active', d: 'Collect 64 oak_log for village', i: 0 },
    { s: 'active', d: 'Navigate to forest biome', i: 1 },
    { s: 'completed', d: 'Locate oak tree cluster', i: 2 },
    { s: 'active', d: 'Chop trees until inventory full', i: 2 },
    { s: 'pending', d: 'Return to stockpile', i: 2 },
    { s: 'pending', d: 'Deposit collected wood', i: 2 },
  ];
  const planColor = { active: 'var(--accent-cyan)', completed: '#66BB6A', pending: 'var(--text-tertiary)' };
  return (
    <Reveal delay={80} style={{ gridColumn: 'span 4' }}>
      <Card style={{ padding: 18, height: '100%' }}>
        <div className="flex items-center" style={{ gap: 12, marginBottom: 16 }}>
          <Avatar role={agent.role} name={agent.name} size={56} />
          <div>
            <div className="t-body" style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: 14 }}>{agent.name}</div>
            <div style={{ marginTop: 5 }}><RoleBadge role={agent.role} /></div>
            <div className="flex items-center" style={{ gap: 6, marginTop: 6 }}><StatusDot status={agent.status} size={6} /><span className="t-caption capitalize">{agent.status}</span></div>
          </div>
        </div>
        <div className="flex" style={{ borderBottom: '1px solid var(--border-subtle)', marginBottom: 14 }}>
          {tabs.map(([k, l]) => (
            <button key={k} onClick={() => setTab(k)} className="relative" style={{ padding: '9px 14px', cursor: 'pointer', background: 'transparent', border: 'none', fontSize: 11.5, color: tab === k ? 'var(--accent-cyan)' : 'var(--text-secondary)', fontWeight: tab === k ? 500 : 400 }}>
              {l}{tab === k && <span style={{ position: 'absolute', bottom: -1, left: 0, right: 0, height: 2, background: 'var(--accent-cyan)' }} />}
            </button>
          ))}
        </div>
        <div style={{ minHeight: 270, maxHeight: 290, overflow: 'auto' }} className="tab-fade" key={tab}>
          {tab === 'state' && (
            <div className="soft mono" style={{ padding: 14, fontSize: 10.5, lineHeight: 1.9, color: 'var(--text-secondary)' }}>
              {[['position', `(${agent.x}, ${agent.z})`], ['health', `${(agent.health ?? 20).toFixed(1)}/20.0`], ['hunger', `${(agent.hunger ?? 20).toFixed(1)}/20.0`]].map(([k, v]) => (
                <div key={k}><span style={{ color: 'var(--text-tertiary)' }}>{k}:</span> {v}</div>
              ))}
              <div><span style={{ color: 'var(--text-tertiary)' }}>inventory:</span></div>
              <div style={{ paddingLeft: 14, color: 'var(--accent-cyan)' }}>oak_log: 12<br />stone: 4<br />wheat_seeds: 8</div>
              <div><span style={{ color: 'var(--text-tertiary)' }}>nearby:</span></div>
              <div style={{ paddingLeft: 14 }}>oak_tree x3 (5.2m)<br />chest (12.8m)</div>
              <div><span style={{ color: 'var(--text-tertiary)' }}>action:</span> "chopping oak_log"</div>
              <div><span style={{ color: 'var(--text-tertiary)' }}>target:</span> (145, 64, -87)</div>
            </div>
          )}
          {tab === 'memory' && (
            <div className="flex flex-col" style={{ gap: 9 }}>
              {memory.map((m, i) => (
                <div key={i} className="soft" style={{ padding: 12 }}>
                  <div className="flex items-center justify-between" style={{ marginBottom: 6 }}>
                    <span className="t-caption mono">{m.time}</span>
                    <div style={{ width: 44, height: 4, borderRadius: 999, background: 'var(--border-subtle)' }}>
                      <div style={{ width: `${m.imp * 100}%`, height: '100%', borderRadius: 999, background: m.imp > 0.7 ? 'var(--accent-cyan)' : m.imp > 0.4 ? 'var(--status-training)' : 'var(--text-tertiary)' }} />
                    </div>
                  </div>
                  <div className="t-body" style={{ fontSize: 11 }}>{m.content}</div>
                </div>
              ))}
            </div>
          )}
          {tab === 'plan' && (
            <div className="flex flex-col" style={{ gap: 3 }}>
              {plan.map((p, i) => (
                <div key={i} className="flex items-start" style={{ gap: 9, padding: '7px 8px', paddingLeft: 8 + p.i * 16, borderLeft: p.s === 'active' ? '2px solid var(--accent-cyan)' : '2px solid transparent', background: p.s === 'active' ? 'rgba(0,229,204,0.04)' : 'transparent', borderRadius: 4 }}>
                  <span style={{ width: 6, height: 6, borderRadius: 999, marginTop: 4, flexShrink: 0, background: planColor[p.s] }} />
                  <span className="t-body" style={{ fontSize: 11, color: p.s === 'pending' ? 'var(--text-tertiary)' : 'var(--text-secondary)' }}>{p.d}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </Reveal>
  );
}

function Chat({ agent, setAgent, agents }) {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const scrollRef = useRef(null);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [msgs, typing]);
  const replies = {
    gatherer: "I'm gathering oak near the forest — 12 logs in inventory, heading to the stockpile next.",
    builder: 'Working on the village center. Could use more stone blocks if any are spare.',
    farmer: "Crops are coming along. I'll have wheat ready for harvest within a few cycles.",
    defender: 'Perimeter is secure. No hostile mobs detected within the arena bounds.',
  };
  const send = () => {
    if (!input.trim()) return;
    const t = new Date().toLocaleTimeString('en-US', { hour12: false });
    setMsgs((m) => [...m, { id: Date.now(), sender: 'user', content: input.trim(), t }]);
    setInput(''); setTyping(true);
    setTimeout(() => {
      setTyping(false);
      setMsgs((m) => [...m, { id: Date.now() + 1, sender: 'agent', name: agent.name, role: agent.role, content: replies[agent.role], t: new Date().toLocaleTimeString('en-US', { hour12: false }) }]);
    }, 1800);
  };
  return (
    <Reveal delay={140} style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 16, height: 210 }}>
        <div className="flex" style={{ height: '100%', gap: 14 }}>
          <div style={{ width: 184, flexShrink: 0, borderRight: '1px solid var(--border-subtle)', paddingRight: 14 }}>
            <h4 className="t-label" style={{ marginBottom: 10 }}>Conversations</h4>
            <div className="flex flex-col" style={{ gap: 3 }}>
              {agents.map((a) => (
                <button key={a.id} onClick={() => setAgent(a.id)} className="flex items-center" style={{ gap: 9, height: 34, padding: '0 8px', borderRadius: 7, cursor: 'pointer', textAlign: 'left', background: a.id === agent.id ? 'rgba(0,229,204,0.06)' : 'transparent', border: '1px solid ' + (a.id === agent.id ? 'rgba(0,229,204,0.2)' : 'transparent') }}
                  onMouseEnter={(e) => { if (a.id !== agent.id) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
                  onMouseLeave={(e) => { if (a.id !== agent.id) e.currentTarget.style.background = 'transparent'; }}>
                  <Avatar role={a.role} name={a.name} size={22} ring={false} />
                  <span className="t-body truncate" style={{ fontSize: 11, color: a.id === agent.id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{a.name}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-col flex-1" style={{ minWidth: 0 }}>
            <div ref={scrollRef} className="flex-1 flex flex-col" style={{ gap: 8, overflowY: 'auto', marginBottom: 10, paddingRight: 4 }}>
              {msgs.length === 0 && !typing && (
                <div className="flex-1 flex flex-col items-center justify-center" style={{ gap: 8 }}>
                  <Icon name="message" size={22} style={{ color: 'var(--text-tertiary)' }} />
                  <span className="t-caption">Message <span style={{ color: getRoleColor(agent.role) }}>@{agent.name}</span> — replies stream from the planner</span>
                </div>
              )}
              {msgs.map((m) => (
                <div key={m.id} className="flex" style={{ justifyContent: m.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                  <div style={{ maxWidth: '72%', padding: '8px 12px', borderRadius: 9, background: m.sender === 'user' ? 'var(--elevated)' : 'var(--surface-2)', borderLeft: m.sender === 'agent' ? `2px solid ${getRoleColor(m.role)}` : 'none' }}>
                    {m.sender === 'agent' && <div className="flex items-center" style={{ gap: 8, marginBottom: 3 }}><span className="t-caption" style={{ color: getRoleColor(m.role), fontWeight: 500 }}>{m.name}</span><span className="t-caption mono">{m.t}</span></div>}
                    <div className="t-body" style={{ fontSize: 11.5, lineHeight: 1.5, color: m.sender === 'user' ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{m.content}</div>
                  </div>
                </div>
              ))}
              {typing && (
                <div className="flex" style={{ justifyContent: 'flex-start' }}>
                  <div style={{ padding: '9px 12px', borderRadius: 9, background: 'var(--surface-2)', borderLeft: `2px solid ${getRoleColor(agent.role)}` }}>
                    <div className="flex items-center" style={{ gap: 4 }}>
                      {[0, 1, 2].map((i) => <span key={i} style={{ width: 5, height: 5, borderRadius: 999, background: 'var(--text-tertiary)', animation: `blink 1s ${i * 0.2}s infinite` }} />)}
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="flex items-center" style={{ gap: 9 }}>
              <input className="inp flex-1" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} placeholder={`@${agent.name} message...`} style={{ height: 34 }} />
              <button onClick={send} className="flex items-center justify-center" style={{ width: 34, height: 34, borderRadius: 9, border: 'none', cursor: 'pointer', background: 'var(--accent-cyan)', color: 'var(--void)' }}><Icon name="send" size={13} /></button>
            </div>
          </div>
        </div>
      </Card>
    </Reveal>
  );
}

export default function Spectate({ selectedAgentId, setSelectedAgentId, agents = [], agentsOnline = false }) {
  if (agents.length === 0) {
    return (
      <div className="grid12">
        <Reveal style={{ gridColumn: '1 / -1' }}>
          <Card style={{ padding: 40, display: 'flex', flexDirection: 'column' }}>
            <EmptyState
              icon="map"
              title={agentsOnline ? 'No agents to observe' : 'Bridge offline'}
              hint={agentsOnline
                ? 'Spawn agents from the Bot Config tab to watch them live in the world view.'
                : 'The world view streams live agent positions once the bridge is reachable.'}
            />
          </Card>
        </Reveal>
      </div>
    );
  }
  const agent = agents.find((a) => a.id === selectedAgentId || a.uuid === selectedAgentId) || agents[0];
  return (
    <div className="grid12">
      <WorldViewport agents={agents} />
      <Inspector agent={agent} />
      <Chat agent={agent} setAgent={setSelectedAgentId} agents={agents} />
    </div>
  );
}
