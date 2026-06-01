/* AI Utopia — Bot Config tab */
import { useState } from 'react';
import { Card, Reveal, SectionTitle, Toggle, Slider, Avatar, RoleBadge, StatusDot, GhostButton, EmptyState, OfflinePill, Toasts } from '../components/primitives.jsx';
import { Icon } from '../lib/icons.jsx';
import { roleMeta, getRoleColor } from '../mockData.js';
import { spawnAgent, killAgent } from '../api.js';

const roles = Object.keys(roleMeta);

function SpawnController({ selected, setSelected, onSpawn, busy }) {
  const [name, setName] = useState('');
  const submit = async () => {
    if (busy) return;
    await onSpawn(selected, name.trim() || undefined);
    setName('');
  };
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
          <input className="inp" style={{ width: 200 }} placeholder="Auto-generate from pool" value={name}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} />
          <button className="fill-btn" onClick={submit} disabled={busy}
            style={{ background: getRoleColor(selected), boxShadow: `0 0 0 0`, opacity: busy ? 0.6 : 1, cursor: busy ? 'wait' : 'pointer' }}
            onMouseEnter={(e) => e.currentTarget.style.boxShadow = `0 0 22px ${getRoleColor(selected)}40`}
            onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}>
            <Icon name={busy ? 'loader' : 'sparkles'} size={13} className={busy ? 'spin' : ''} /> {busy ? 'Spawning…' : 'Spawn'}
          </button>
        </div>
        <p className="t-caption" style={{ marginTop: 14, maxWidth: 620 }}>
          Agents receive a persistent ULID identity, skin, episodic-memory collection, and ChromaDB indexes on spawn.
          Spawning requires a live Minecraft server on the bridge.
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

function fmtBorn(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return `${d.toISOString().slice(0, 10)} ${d.toLocaleTimeString('en-US', { hour12: false }).slice(0, 5)}`;
  } catch { return String(iso); }
}

function IdentityPreview({ agent, onKill, killing }) {
  if (!agent) {
    return (
      <Reveal delay={120} style={{ gridColumn: 'span 5' }}>
        <Card style={{ padding: 22, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <EmptyState icon="cpu" title="No agent selected" hint="Spawn an agent or pick one from the roster to inspect its identity." />
        </Card>
      </Reveal>
    );
  }
  const stats = [
    ['Born', fmtBorn(agent.born)], ['Status', agent.status.toUpperCase()],
    ['Skin', agent.skin], ['Position', `(${agent.x}, ${agent.z})`],
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
            <GhostButton color="var(--status-offline)" icon={killing ? 'loader' : 'userX'} full onClick={() => onKill(agent)}>
              {killing ? 'Terminating…' : 'Terminate'}
            </GhostButton>
            <GhostButton color="var(--accent-cyan)" icon="brain" full>Inspect Memory</GhostButton>
          </div>
        </div>
      </Card>
    </Reveal>
  );
}

// Live roster — lets the operator pick which real agent to inspect / terminate.
function Roster({ agents, agentsOnline, selectedUuid, onSelect }) {
  return (
    <Reveal delay={60} style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 18 }}>
        <SectionTitle right={agentsOnline ? <span className="t-caption">{agents.length} agent{agents.length === 1 ? '' : 's'}</span> : <OfflinePill />}>
          Agent Roster
        </SectionTitle>
        {agents.length === 0 ? (
          <EmptyState icon="users" title="No agents in the village" hint="Use Spawn Agent above to create your first agent. Spawning needs a live Minecraft server." />
        ) : (
          <div className="flex wrap" style={{ gap: 9 }}>
            {agents.map((a) => {
              const on = a.uuid === selectedUuid;
              const c = getRoleColor(a.role);
              return (
                <button key={a.uuid || a.id} onClick={() => onSelect(a)} className="flex items-center" style={{
                  gap: 10, padding: '9px 13px', borderRadius: 10, cursor: 'pointer', textAlign: 'left',
                  background: on ? `${c}12` : 'var(--surface-2)', border: `1px solid ${on ? `${c}55` : 'var(--border-subtle)'}`,
                  transition: 'all 0.2s var(--ease)',
                }}>
                  <Avatar role={a.role} name={a.name} size={30} ring={false} />
                  <div>
                    <div className="t-body" style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: 12 }}>{a.name}</div>
                    <div className="t-caption capitalize">{a.role} · {a.status}</div>
                  </div>
                  <StatusDot status={a.status} size={6} />
                </button>
              );
            })}
          </div>
        )}
      </Card>
    </Reveal>
  );
}

let _toastId = 0;

export default function BotConfig({ agents = [], agentsOnline = false, refetchAgents }) {
  const [selected, setSelected] = useState('gatherer');
  const [selectedUuid, setSelectedUuid] = useState(null);
  const [spawning, setSpawning] = useState(false);
  const [killingUuid, setKillingUuid] = useState(null);
  const [toasts, setToasts] = useState([]);

  const pushToast = (kind, message) => {
    const id = ++_toastId;
    setToasts((t) => [...t, { id, kind, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  };
  const dismiss = (id) => setToasts((t) => t.filter((x) => x.id !== id));

  // Resolve the agent shown in the identity card: explicit selection, else the
  // first matching the chosen role, else the first agent (may be undefined → empty state).
  const previewAgent =
    agents.find((a) => a.uuid === selectedUuid) ||
    agents.find((a) => a.role === selected) ||
    agents[0] ||
    null;

  const handleSpawn = async (role, name) => {
    setSpawning(true);
    try {
      const res = await spawnAgent(role, name);
      if (res && res.ok === false) {
        pushToast('error', `Spawn failed: ${res.error || 'unknown error'}`);
      } else {
        const a = res && res.agent;
        pushToast('success', `Spawned ${a?.name || name || role} (${role})`);
        if (a?.uuid) setSelectedUuid(a.uuid);
      }
    } catch (e) {
      pushToast('error', e.detail ? `Spawn failed: ${e.detail}` : `Spawn failed — backend unreachable`);
    } finally {
      setSpawning(false);
      refetchAgents && refetchAgents();
    }
  };

  const handleKill = async (agent) => {
    if (!agent?.uuid) return;
    setKillingUuid(agent.uuid);
    try {
      await killAgent(agent.uuid);
      pushToast('success', `Terminated ${agent.name}`);
      if (selectedUuid === agent.uuid) setSelectedUuid(null);
    } catch (e) {
      pushToast('error', e.detail ? `Terminate failed: ${e.detail}` : `Terminate failed — backend unreachable`);
    } finally {
      setKillingUuid(null);
      refetchAgents && refetchAgents();
    }
  };

  return (
    <div className="grid12">
      <SpawnController selected={selected} setSelected={setSelected} onSpawn={handleSpawn} busy={spawning} />
      <Roster agents={agents} agentsOnline={agentsOnline} selectedUuid={previewAgent?.uuid} onSelect={(a) => setSelectedUuid(a.uuid)} />
      <ParamsForm role={selected} />
      <IdentityPreview agent={previewAgent} onKill={handleKill} killing={!!previewAgent && killingUuid === previewAgent.uuid} />
      <Toasts items={toasts} onDismiss={dismiss} />
    </div>
  );
}
