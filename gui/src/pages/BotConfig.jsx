/* AI Utopia — Bot Config tab */
import { useState } from 'react';
import { Card, Reveal, SectionTitle, Toggle, Slider, Avatar, RoleBadge, StatusDot, GhostButton } from '../components/primitives.jsx';
import { Icon } from '../lib/icons.jsx';
import { agents, roleMeta, getRoleColor } from '../mockData.js';

const roles = Object.keys(roleMeta);

function SpawnController({ selected, setSelected }) {
  const [name, setName] = useState('');
  return (
    <Reveal style={{ gridColumn: '1 / -1' }}>
      <Card elevated className="card-hover" style={{ padding: 18 }}>
        <div className="flex items-center wrap" style={{ gap: 20 }}>
          <span className="t-section">Spawn Agent</span>
          <div className="flex items-center" style={{ gap: 8 }}>
            {roles.map((r) => {
              const c = getRoleColor(r);
              const on = selected === r;
              return (
                <button key={r} onClick={() => setSelected(r)} className="flex items-center" style={{
                  gap: 8, height: 36, padding: '0 14px', borderRadius: 9, cursor: 'pointer',
                  background: on ? `${c}18` : 'var(--surface-2)',
                  border: `1px solid ${on ? `${c}60` : 'var(--border-subtle)'}`,
                  color: on ? c : 'var(--text-secondary)', transition: 'all 0.2s var(--ease)',
                }}>
                  <Icon name={roleMeta[r].icon} size={14} />
                  <span style={{ fontSize: 11.5 }}>{roleMeta[r].label}</span>
                </button>
              );
            })}
          </div>
          <input className="inp" style={{ width: 200 }} placeholder="Auto-generate from pool" value={name} onChange={(e) => setName(e.target.value)} />
          <button className="fill-btn" style={{ background: getRoleColor(selected), boxShadow: `0 0 0 0` }}
            onMouseEnter={(e) => e.currentTarget.style.boxShadow = `0 0 22px ${getRoleColor(selected)}40`}
            onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}>
            <Icon name="sparkles" size={13} /> Spawn
          </button>
        </div>
        <p className="t-caption" style={{ marginTop: 14, maxWidth: 620 }}>
          Agents receive a persistent ULID identity, skin, episodic-memory collection, and ChromaDB indexes on spawn.
        </p>
      </Card>
    </Reveal>
  );
}

function ParamsForm({ role }) {
  const [temp, setTemp] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [mem, setMem] = useState({ episodic: true, skill: true, rag: false });
  return (
    <Reveal delay={80} style={{ gridColumn: 'span 7' }}>
      <Card style={{ padding: 18 }}>
        <SectionTitle>Agent Parameters</SectionTitle>
        <div className="flex flex-col" style={{ gap: 16 }}>
          <div className="grid12" style={{ gap: 14 }}>
            <div style={{ gridColumn: 'span 6' }}>
              <label className="t-label" style={{ display: 'block', marginBottom: 7 }}>Role</label>
              <div className="inp flex items-center capitalize" style={{ color: 'var(--text-tertiary)' }}>{role} · locked after spawn</div>
            </div>
            <div style={{ gridColumn: 'span 6' }}>
              <label className="t-label" style={{ display: 'block', marginBottom: 7 }}>Skin Pool</label>
              <select className="inp full"><option>Default Villager</option><option>Steve Variant</option><option>Alex Variant</option><option>Custom Upload</option></select>
            </div>
            <div style={{ gridColumn: 'span 6' }}>
              <label className="t-label" style={{ display: 'block', marginBottom: 7 }}>Neural Planner</label>
              <select className="inp full"><option>Claude Haiku</option><option>Local Qwen 14B</option><option>Stub Planner</option><option>Manual CLI</option></select>
            </div>
            <div style={{ gridColumn: 'span 6' }}>
              <label className="t-label" style={{ display: 'block', marginBottom: 7 }}>Max Tokens</label>
              <input type="number" className="inp full" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} />
            </div>
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Temperature</label>
            <Slider value={temp} min={0} max={2} step={0.1} onChange={setTemp} format={(v) => v.toFixed(1)} />
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 7 }}>System Prompt</label>
            <textarea className="inp full mono" rows={4} style={{ fontSize: 11, color: 'var(--text-secondary)' }}
              defaultValue={"You are a cooperative AI agent in a Minecraft village. Work with other agents to gather resources, build structures, farm crops, and defend the village. Communicate via chat when coordination is needed. Prioritize the village's collective wellbeing over individual tasks."} />
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 10 }}>Memory Strategy</label>
            <div className="flex flex-col" style={{ gap: 11 }}>
              {[['Episodic Memory', 'episodic'], ['Skill Library', 'skill'], ['RAG Context', 'rag']].map(([lbl, k]) => (
                <div key={k} className="flex items-center justify-between">
                  <span className="t-body">{lbl}</span>
                  <Toggle value={mem[k]} onChange={(v) => setMem((m) => ({ ...m, [k]: v }))} />
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center" style={{ gap: 12, marginTop: 4 }}>
            <GhostButton color="var(--accent-cyan)" icon="check">Apply Changes</GhostButton>
            <button className="ghost-btn" style={{ border: '1px solid transparent', color: 'var(--text-tertiary)' }}>Reset</button>
          </div>
        </div>
      </Card>
    </Reveal>
  );
}

function IdentityPreview({ agent }) {
  const stats = [
    ['Born', '2026-05-26 14:23'], ['Status', agent.status.toUpperCase()],
    ['Skin', agent.skin], ['Memory', '2 collections'],
  ];
  return (
    <Reveal delay={120} style={{ gridColumn: 'span 5' }}>
      <Card className="card-hover" style={{ padding: 22, height: '100%' }}>
        <div className="flex flex-col items-center">
          <Avatar role={agent.role} name={agent.name} size={88} />
          <h4 className="t-hero" style={{ fontSize: 20, marginTop: 16 }}>{agent.name}</h4>
          <div style={{ marginTop: 8 }}><RoleBadge role={agent.role} /></div>
          <code className="t-caption mono" style={{ marginTop: 10, letterSpacing: '0.04em' }}>{agent.uuid}</code>
          <div className="grid12" style={{ gap: 10, width: '100%', marginTop: 20 }}>
            {stats.map(([l, v]) => (
              <div key={l} className="soft" style={{ gridColumn: 'span 6', padding: 12 }}>
                <div className="t-label" style={{ marginBottom: 6 }}>{l}</div>
                <div className="t-body" style={{ color: 'var(--text-primary)', fontSize: 11.5 }}>
                  {l === 'Status' ? <span className="flex items-center" style={{ gap: 6 }}><StatusDot status={agent.status} size={6} />{v}</span> : v}
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center full" style={{ gap: 10, marginTop: 16 }}>
            <GhostButton color="var(--status-offline)" icon="userX" full>Terminate</GhostButton>
            <GhostButton color="var(--accent-cyan)" icon="brain" full>Inspect Memory</GhostButton>
          </div>
        </div>
      </Card>
    </Reveal>
  );
}

export default function BotConfig() {
  const [selected, setSelected] = useState('gatherer');
  const previewAgent = agents.find((a) => a.role === selected) || agents[0];
  return (
    <div className="grid12">
      <SpawnController selected={selected} setSelected={setSelected} />
      <ParamsForm role={selected} />
      <IdentityPreview agent={previewAgent} />
    </div>
  );
}
