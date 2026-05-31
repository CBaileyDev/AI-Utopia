/* AI Utopia — Dashboard tab */
const Dashboard = (function () {
  const { useState, useEffect } = React;
  const { Card, Reveal, SectionTitle, StatusDot, Avatar, AnimatedNumber, AreaStack, Donut, Icon } = window;
  const AIU = window.AIU;

  function Metric({ icon, value, label, sub, trend, delay, animated, decimals }) {
    const tc = trend === 'up' ? 'var(--accent-cyan)' : trend === 'down' ? 'var(--status-offline)' : 'var(--text-tertiary)';
    return (
      <Reveal delay={delay} style={{ gridColumn: 'span 3' }}>
        <Card className="card-hover" style={{ padding: 18, height: '100%' }}>
          <div className="flex items-center justify-between">
            <Icon name={icon} size={17} style={{ color: 'var(--text-tertiary)' }} />
            <Icon name={trend === 'up' ? 'trendUp' : trend === 'down' ? 'trendDown' : 'minus'} size={13} style={{ color: tc }} />
          </div>
          <div className="t-data" style={{ marginTop: 16 }}>
            {animated ? <AnimatedNumber value={value} decimals={decimals || 0} /> : value}
          </div>
          <div className="t-label" style={{ marginTop: 7 }}>{label}</div>
          <div className="t-caption" style={{ marginTop: 4, color: tc }}>{sub}</div>
        </Card>
      </Reveal>
    );
  }

  function ActivityTimeline() {
    const colors = AIU.roleColors;
    return (
      <Reveal delay={120} style={{ gridColumn: 'span 8' }}>
        <Card style={{ padding: 18 }}>
          <SectionTitle right={<span className="t-caption">last 7 hrs</span>}>Agent Activity Timeline</SectionTitle>
          <AreaStack data={AIU.activity} keys={['gatherer', 'builder', 'farmer', 'defender']} colors={colors} height={196} />
          <div className="flex items-center" style={{ gap: 18, marginTop: 12 }}>
            {Object.keys(AIU.roleMeta).map((r) => (
              <div key={r} className="flex items-center" style={{ gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: 999, background: colors[r] }} />
                <span className="t-caption capitalize">{r}</span>
              </div>
            ))}
          </div>
        </Card>
      </Reveal>
    );
  }

  function RewardDistribution() {
    const agents = AIU.agents;
    const total = agents.reduce((s, a) => s + a.rewards, 0);
    const data = agents.map((a) => ({ name: a.name, value: a.rewards, color: AIU.getRoleColor(a.role) }));
    const sorted = [...agents].sort((a, b) => b.rewards - a.rewards);
    return (
      <Reveal delay={150} style={{ gridColumn: 'span 4' }}>
        <Card style={{ padding: 18, height: '100%' }}>
          <SectionTitle>Reward Distribution</SectionTitle>
          <div className="flex items-center" style={{ gap: 16 }}>
            <div className="relative" style={{ width: 132, height: 132, flexShrink: 0 }}>
              <Donut data={data} size={132} thickness={20} />
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div className="t-data" style={{ fontSize: 22 }}>+<AnimatedNumber value={total} /></div>
                <div className="t-label" style={{ marginTop: 2 }}>Total</div>
              </div>
            </div>
            <div className="flex-1 flex flex-col" style={{ gap: 9 }}>
              {sorted.map((a, i) => (
                <div key={a.id} className="flex items-center justify-between">
                  <div className="flex items-center" style={{ gap: 8 }}>
                    <span style={{ width: 7, height: 7, borderRadius: 999, background: AIU.getRoleColor(a.role) }} />
                    <span className="t-body" style={{ color: 'var(--text-primary)' }}>{a.name}</span>
                  </div>
                  <span className="t-caption mono">+{a.rewards}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </Reveal>
    );
  }

  const typeColors = { AGENT: 'var(--accent-cyan)', TRAIN: 'var(--accent-lavender)', SYSTEM: 'var(--text-secondary)', CHAT: '#66BB6A' };

  function ActivityFeed() {
    const [logs, setLogs] = useState(AIU.logs);
    return (
      <Reveal delay={180} style={{ gridColumn: 'span 7' }}>
        <Card style={{ padding: 18, display: 'flex', flexDirection: 'column', height: 360 }}>
          <SectionTitle right={<span className="flex items-center t-caption" style={{ gap: 6, color: 'var(--accent-cyan)' }}><StatusDot status="alive" size={6} /> LIVE</span>}>Live Activity Feed</SectionTitle>
          <div className="flex-1" style={{ overflowY: 'auto', marginRight: -6, paddingRight: 6 }}>
            {logs.map((log, i) => (
              <div key={log.id} className="flex items-start" style={{ gap: 10, padding: '6px 4px', borderRadius: 6, background: i % 2 ? 'transparent' : 'rgba(255,255,255,0.012)' }}>
                <span className="t-caption mono" style={{ minWidth: 52 }}>{log.timestamp}</span>
                <span className="t-caption mono" style={{ color: typeColors[log.type], minWidth: 50, fontWeight: 500 }}>[{log.type}]</span>
                <span className="t-body" style={{ lineHeight: 1.5 }}>{log.message}</span>
              </div>
            ))}
          </div>
        </Card>
      </Reveal>
    );
  }

  function ActiveAgents({ onOpen }) {
    return (
      <Reveal delay={210} style={{ gridColumn: 'span 5' }}>
        <Card style={{ padding: 18, height: 360, display: 'flex', flexDirection: 'column' }}>
          <SectionTitle right={<span className="t-caption">{AIU.agents.length} online</span>}>Active Agents</SectionTitle>
          <div className="flex flex-col flex-1" style={{ gap: 9 }}>
            {AIU.agents.map((a) => (
              <button key={a.id} onClick={() => onOpen(a.id)} className="agent-row flex items-center" style={{
                gap: 12, padding: 10, borderRadius: 10, textAlign: 'left', cursor: 'pointer', flex: 1,
                background: 'var(--surface-2)', border: '1px solid var(--border-subtle)', transition: 'all 0.2s var(--ease)',
              }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-active)'; e.currentTarget.style.transform = 'translateX(2px)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.transform = 'none'; }}>
                <Avatar role={a.role} name={a.name} size={38} />
                <div className="flex-1" style={{ minWidth: 0 }}>
                  <div className="t-body truncate" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{a.name}</div>
                  <div className="t-caption capitalize">{a.role} · {a.status}</div>
                </div>
                <StatusDot status={a.status} />
              </button>
            ))}
          </div>
        </Card>
      </Reveal>
    );
  }

  return function Dashboard({ onOpenAgent }) {
    const [epochs, setEpochs] = useState(1247);
    const [worldTime, setWorldTime] = useState(14.53);
    useEffect(() => {
      const a = setInterval(() => setWorldTime((t) => (t + 0.004) % 24), 120);
      const b = setInterval(() => setEpochs((e) => e + 1), 9000);
      return () => { clearInterval(a); clearInterval(b); };
    }, []);
    const timeStr = `${String(Math.floor(worldTime)).padStart(2, '0')}:${String(Math.floor((worldTime % 1) * 60)).padStart(2, '0')}`;

    return (
      <div className="grid12">
        <Metric icon="users" value={4} animated label="Active Agents" sub="+2 this session" trend="up" delay={0} />
        <Metric icon="brain" value={epochs} animated label="Epochs Completed" sub="Batch 64 / 128" trend="neutral" delay={40} />
        <Metric icon="database" value={8492} animated label="Memory Entries" sub="ChromaDB active" trend="neutral" delay={80} />
        <Metric icon="clock" value={timeStr} label="World Time" sub="Day 1,247" trend="neutral" delay={120} />
        <ActivityTimeline />
        <RewardDistribution />
        <ActivityFeed />
        <ActiveAgents onOpen={onOpenAgent} />
      </div>
    );
  };
})();
window.Dashboard = Dashboard;


/* AI Utopia — Bot Config tab */
const BotConfig = (function () {
  const { useState } = React;
  const { Card, Reveal, SectionTitle, Toggle, Slider, Avatar, RoleBadge, StatusDot, GhostButton, Icon } = window;
  const AIU = window.AIU;
  const roles = Object.keys(AIU.roleMeta);

  function SpawnController({ selected, setSelected }) {
    const [name, setName] = useState('');
    return (
      <Reveal style={{ gridColumn: '1 / -1' }}>
        <Card elevated className="card-hover" style={{ padding: 18 }}>
          <div className="flex items-center wrap" style={{ gap: 20 }}>
            <span className="t-section">Spawn Agent</span>
            <div className="flex items-center" style={{ gap: 8 }}>
              {roles.map((r) => {
                const c = AIU.getRoleColor(r);
                const on = selected === r;
                return (
                  <button key={r} onClick={() => setSelected(r)} className="flex items-center" style={{
                    gap: 8, height: 36, padding: '0 14px', borderRadius: 9, cursor: 'pointer',
                    background: on ? `${c}18` : 'var(--surface-2)',
                    border: `1px solid ${on ? `${c}60` : 'var(--border-subtle)'}`,
                    color: on ? c : 'var(--text-secondary)', transition: 'all 0.2s var(--ease)',
                  }}>
                    <Icon name={AIU.roleMeta[r].icon} size={14} />
                    <span style={{ fontSize: 11.5 }}>{AIU.roleMeta[r].label}</span>
                  </button>
                );
              })}
            </div>
            <input className="inp" style={{ width: 200 }} placeholder="Auto-generate from pool" value={name} onChange={(e) => setName(e.target.value)} />
            <button className="fill-btn" style={{ background: AIU.getRoleColor(selected), boxShadow: `0 0 0 0` }}
              onMouseEnter={(e) => e.currentTarget.style.boxShadow = `0 0 22px ${AIU.getRoleColor(selected)}40`}
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
    const c = AIU.getRoleColor(agent.role);
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

  return function BotConfig() {
    const [selected, setSelected] = useState('gatherer');
    const previewAgent = AIU.agents.find((a) => a.role === selected) || AIU.agents[0];
    return (
      <div className="grid12">
        <SpawnController selected={selected} setSelected={setSelected} />
        <ParamsForm role={selected} />
        <IdentityPreview agent={previewAgent} />
      </div>
    );
  };
})();
window.BotConfig = BotConfig;


/* AI Utopia — Training tab */
const Training = (function () {
  const { useState } = React;
  const { Card, Reveal, SectionTitle, Slider, Segmented, GhostButton, Sparkline, LineChart, Icon } = window;
  const AIU = window.AIU;

  function Pipeline({ active }) {
    const stages = [
      { name: 'Environment Setup', status: 'complete' },
      { name: 'Observation Feed', status: 'active' },
      { name: 'Policy Update', status: 'pending' },
      { name: 'Weight Promotion', status: 'pending' },
    ];
    return (
      <Reveal style={{ gridColumn: '1 / -1' }}>
        <Card style={{ padding: 20 }}>
          <SectionTitle right={<span className="t-caption">PPO + LSTM · seed 42</span>}>Training Pipeline</SectionTitle>
          <div className="flex items-center justify-center" style={{ marginTop: 6 }}>
            {stages.map((s, i) => {
              const done = s.status === 'complete', on = s.status === 'active';
              return (
                <div key={s.name} className="flex items-center">
                  <div className="flex flex-col items-center" style={{ gap: 9, width: 110 }}>
                    <div className="relative flex items-center justify-center" style={{
                      width: 42, height: 42, borderRadius: 999,
                      background: done ? 'var(--accent-cyan)' : 'transparent',
                      border: `2px solid ${done || on ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
                    }}>
                      {on && <span style={{ position: 'absolute', inset: -2, borderRadius: 999, border: '2px solid var(--accent-cyan)', animation: 'ringPing 1.6s ease-out infinite' }} />}
                      {done && <Icon name="check" size={18} style={{ color: 'var(--void)' }} stroke={2.5} />}
                      {on && <Icon name="loader" size={16} className="spin" style={{ color: 'var(--accent-cyan)' }} />}
                      {!done && !on && <Icon name="circle" size={9} style={{ color: 'var(--text-tertiary)' }} fill="var(--text-tertiary)" />}
                    </div>
                    <span className="t-caption" style={{ textAlign: 'center', color: done || on ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>{s.name}</span>
                  </div>
                  {i < stages.length - 1 && (
                    <div style={{ width: 70, height: 2, marginTop: -26, borderRadius: 2, background: done ? 'linear-gradient(90deg, var(--accent-cyan), var(--border-subtle))' : 'var(--border-subtle)' }} />
                  )}
                </div>
              );
            })}
          </div>
          <div className="flex items-center justify-center" style={{ gap: 14, marginTop: 18 }}>
            <span className="t-body">{active ? 'Pipeline active' : 'Pipeline paused'} — Epoch 1,247 in progress</span>
            <span className="t-data mono" style={{ fontSize: 16, color: 'var(--accent-cyan)' }}>00:14:32</span>
          </div>
        </Card>
      </Reveal>
    );
  }

  function Metrics() {
    const spark = Array.from({ length: 28 }, (_, i) => 2 + Math.sin(i * 0.4) * 0.3 + Math.sin(i * 1.3) * 0.08);
    const items = [
      { label: 'Mean Reward', value: '+2.14', trend: 'up', spark: true },
      { label: 'Policy Loss', value: '0.0047', trend: 'down' },
      { label: 'Value Loss', value: '0.0089', trend: 'neutral' },
      { label: 'Entropy', value: '0.312', trend: 'up' },
      { label: 'KL Divergence', value: '0.0021', trend: 'down' },
    ];
    return (
      <Reveal delay={80} style={{ gridColumn: 'span 4' }}>
        <div className="flex flex-col" style={{ gap: 12, height: '100%' }}>
          {items.map((m) => (
            <Card key={m.label} className="card-hover" style={{ padding: 15, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div className="t-label" style={{ marginBottom: 5 }}>{m.label}</div>
                <div className="flex items-center" style={{ gap: 8 }}>
                  <span className="t-data" style={{ fontSize: 21 }}>{m.value}</span>
                  <Icon name={m.trend === 'up' ? 'up' : m.trend === 'down' ? 'down' : 'minus'} size={13}
                    style={{ color: m.trend === 'up' ? 'var(--accent-cyan)' : m.trend === 'down' ? 'var(--status-training)' : 'var(--text-tertiary)' }} />
                </div>
              </div>
              {m.spark && <Sparkline data={spark} />}
            </Card>
          ))}
        </div>
      </Reveal>
    );
  }

  function RewardChart() {
    return (
      <Reveal delay={120} style={{ gridColumn: 'span 5' }}>
        <Card style={{ padding: 18, height: '100%' }}>
          <SectionTitle right={<span className="t-caption flex items-center" style={{ gap: 6 }}><span style={{ width: 14, height: 0, borderTop: '1px dashed var(--text-tertiary)', display: 'inline-block' }} />baseline 1.8</span>}>Reward Over Time</SectionTitle>
          <LineChart data={AIU.rewardCurve} baseline={1.8} height={272} />
        </Card>
      </Reveal>
    );
  }

  function Controls({ active, setActive }) {
    const [lr, setLr] = useState(4);
    const [batch, setBatch] = useState(64);
    const [gamma, setGamma] = useState(99);
    return (
      <Reveal delay={160} style={{ gridColumn: 'span 3' }}>
        <Card style={{ padding: 18, height: '100%' }}>
          <SectionTitle>Controls</SectionTitle>
          <div className="flex flex-col" style={{ gap: 10 }}>
            <GhostButton color="var(--status-training)" icon={active ? 'pause' : 'play'} full onClick={() => setActive(!active)}>{active ? 'Pause Training' : 'Resume Training'}</GhostButton>
            <GhostButton color="var(--accent-cyan)" icon="save" full>Save Checkpoint</GhostButton>
            <GhostButton color="var(--accent-lavender)" icon="trophy" full>
              Promote Weights
              <span style={{ marginLeft: 2, padding: '1px 7px', borderRadius: 999, fontSize: 9, background: 'var(--accent-lavender)', color: 'var(--void)' }}>3</span>
            </GhostButton>
            <GhostButton color="var(--status-offline)" icon="stop" full>Emergency Stop</GhostButton>
          </div>
          <div className="flex flex-col" style={{ gap: 18, marginTop: 22 }}>
            <div>
              <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Learning Rate</label>
              <Slider value={lr} min={2} max={6} step={1} onChange={setLr} format={(v) => `1e-${v}`} />
            </div>
            <div>
              <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Batch Size</label>
              <Segmented options={[16, 32, 64, 128]} value={batch} onChange={setBatch} />
            </div>
            <div>
              <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Gamma</label>
              <Slider value={gamma} min={90} max={99} step={1} onChange={setGamma} format={(v) => (v / 100).toFixed(2)} />
            </div>
          </div>
        </Card>
      </Reveal>
    );
  }

  function EpochTable() {
    const [sortKey, setSortKey] = useState('epoch');
    const [dir, setDir] = useState('desc');
    const headers = [['epoch', 'Epoch'], ['meanReward', 'Mean Reward'], ['policyLoss', 'Policy Loss'], ['valueLoss', 'Value Loss'], ['entropy', 'Entropy'], ['klDiv', 'KL Div'], ['duration', 'Duration'], ['status', 'Status']];
    const sorted = [...AIU.epochs].sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      if (typeof x === 'number') return dir === 'asc' ? x - y : y - x;
      return dir === 'asc' ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
    });
    const sc = { improved: 'var(--accent-cyan)', stable: 'var(--status-training)', degraded: 'var(--status-offline)' };
    const click = (k) => { if (sortKey === k) setDir(dir === 'asc' ? 'desc' : 'asc'); else { setSortKey(k); setDir('desc'); } };
    return (
      <Reveal delay={200} style={{ gridColumn: '1 / -1' }}>
        <Card style={{ padding: 18 }}>
          <SectionTitle right={<span className="t-caption">{AIU.epochs.length} epochs</span>}>Epoch History</SectionTitle>
          <div style={{ overflow: 'auto', maxHeight: 300 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--elevated)', zIndex: 1 }}>
                <tr>{headers.map(([k, l]) => (
                  <th key={k} onClick={() => click(k)} className="t-label" style={{ textAlign: 'left', padding: '9px 12px', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}>
                    <span className="flex items-center" style={{ gap: 4 }}>{l}{sortKey === k && <span style={{ color: 'var(--accent-cyan)' }}>{dir === 'asc' ? '↑' : '↓'}</span>}</span>
                  </th>
                ))}</tr>
              </thead>
              <tbody>
                {sorted.map((e) => (
                  <tr key={e.epoch} className="epoch-row" style={{ background: e.epoch % 2 ? 'transparent' : 'rgba(255,255,255,0.012)' }}
                    onMouseEnter={(ev) => ev.currentTarget.style.background = 'rgba(0,229,204,0.04)'}
                    onMouseLeave={(ev) => ev.currentTarget.style.background = e.epoch % 2 ? 'transparent' : 'rgba(255,255,255,0.012)'}>
                    <td className="t-body mono" style={{ padding: '9px 12px', color: 'var(--text-primary)' }}>{e.epoch}</td>
                    <td className="t-body mono" style={{ padding: '9px 12px', color: 'var(--accent-cyan)' }}>+{e.meanReward.toFixed(2)}</td>
                    <td className="t-body mono" style={{ padding: '9px 12px' }}>{e.policyLoss.toFixed(4)}</td>
                    <td className="t-body mono" style={{ padding: '9px 12px' }}>{e.valueLoss.toFixed(4)}</td>
                    <td className="t-body mono" style={{ padding: '9px 12px' }}>{e.entropy.toFixed(3)}</td>
                    <td className="t-body mono" style={{ padding: '9px 12px' }}>{e.klDiv.toFixed(4)}</td>
                    <td className="t-caption mono" style={{ padding: '9px 12px' }}>{e.duration}</td>
                    <td style={{ padding: '9px 12px' }}>
                      <span className="flex items-center" style={{ gap: 7 }}>
                        <span style={{ width: 7, height: 7, borderRadius: 999, background: sc[e.status] }} />
                        <span className="t-caption capitalize">{e.status}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </Reveal>
    );
  }

  return function Training() {
    const [active, setActive] = useState(true);
    return (
      <div className="grid12">
        <Pipeline active={active} />
        <Metrics />
        <RewardChart />
        <Controls active={active} setActive={setActive} />
        <EpochTable />
      </div>
    );
  };
})();
window.Training = Training;


/* AI Utopia — Spectate tab */
const Spectate = (function () {
  const { useState, useRef, useEffect } = React;
  const { Card, Reveal, SectionTitle, Avatar, RoleBadge, StatusDot, Icon } = window;
  const AIU = window.AIU;

  function WorldViewport() {
    const ref = useRef(null);
    const posRef = useRef(AIU.agents.map((a, i) => ({
      role: a.role, name: a.name,
      x: 140 + i * 150 + a.x * 0.3, y: 120 + (i % 2) * 140 + a.z * 0.2,
      vx: (Math.sin(i * 9) ) * 0.35, vy: (Math.cos(i * 7)) * 0.35, trail: [],
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
          const c = AIU.getRoleColor(p.role);
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
    const c = AIU.getRoleColor(agent.role);
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
                {[['position', '(142, 64, -89)'], ['health', `${agent.health.toFixed(1)}/20.0`], ['hunger', `${agent.hunger.toFixed(1)}/20.0`]].map(([k, v]) => (
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

  function Chat({ agent, setAgent }) {
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
                {AIU.agents.map((a) => (
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
                    <span className="t-caption">Message <span style={{ color: AIU.getRoleColor(agent.role) }}>@{agent.name}</span> — replies stream from the planner</span>
                  </div>
                )}
                {msgs.map((m) => (
                  <div key={m.id} className="flex" style={{ justifyContent: m.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                    <div style={{ maxWidth: '72%', padding: '8px 12px', borderRadius: 9, background: m.sender === 'user' ? 'var(--elevated)' : 'var(--surface-2)', borderLeft: m.sender === 'agent' ? `2px solid ${AIU.getRoleColor(m.role)}` : 'none' }}>
                      {m.sender === 'agent' && <div className="flex items-center" style={{ gap: 8, marginBottom: 3 }}><span className="t-caption" style={{ color: AIU.getRoleColor(m.role), fontWeight: 500 }}>{m.name}</span><span className="t-caption mono">{m.t}</span></div>}
                      <div className="t-body" style={{ fontSize: 11.5, lineHeight: 1.5, color: m.sender === 'user' ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{m.content}</div>
                    </div>
                  </div>
                ))}
                {typing && (
                  <div className="flex" style={{ justifyContent: 'flex-start' }}>
                    <div style={{ padding: '9px 12px', borderRadius: 9, background: 'var(--surface-2)', borderLeft: `2px solid ${AIU.getRoleColor(agent.role)}` }}>
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

  return function Spectate({ selectedAgentId, setSelectedAgentId }) {
    const agent = AIU.agents.find((a) => a.id === selectedAgentId) || AIU.agents[0];
    return (
      <div className="grid12">
        <WorldViewport />
        <Inspector agent={agent} />
        <Chat agent={agent} setAgent={setSelectedAgentId} />
      </div>
    );
  };
})();
window.Spectate = Spectate;


/* AI Utopia — Settings tab */
const Settings = (function () {
  const { useState } = React;
  const { Card, Reveal, Toggle, Slider, Field, GhostButton, Icon } = window;

  const subtabs = [['system', 'System', 'server'], ['environment', 'Environment', 'globe'], ['memory', 'Memory', 'database'], ['advanced', 'Advanced', 'cpu']];

  function SystemTab() {
    const ro = { color: 'var(--text-tertiary)' };
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 40, rowGap: 0 }}>
        <Field label="Py4J Port" description="Port for the Py4J bridge socket"><input type="number" defaultValue={25099} className="inp" style={{ width: 110 }} /></Field>
        <Field label="Minecraft Version" description="Pinned — UnionClef baseline"><div className="inp flex items-center" style={{ width: 110, ...ro }}>1.21.1</div></Field>
        <Field label="Python Version" description="Required runtime"><div className="inp flex items-center" style={{ width: 110, ...ro }}>3.12</div></Field>
        <Field label="Log Level" description="Console verbosity"><select className="inp" style={{ width: 130 }}><option>DEBUG</option><option selected>INFO</option><option>WARNING</option><option>ERROR</option></select></Field>
        <Field label="Auto-save Interval" description="Seconds between checkpoints"><input type="number" defaultValue={300} className="inp" style={{ width: 110 }} /></Field>
        <Field label="Theme" description="Interface appearance"><select className="inp" style={{ width: 130 }}><option>Void</option><option>Aurora</option><option>Minimal</option></select></Field>
      </div>
    );
  }

  function PathInput({ defaultValue }) {
    return (
      <div className="flex items-center" style={{ gap: 8, width: 360 }}>
        <input className="inp flex-1" defaultValue={defaultValue} />
        <button className="flex items-center justify-center" style={{ width: 36, height: 36, borderRadius: 9, cursor: 'pointer', background: 'var(--surface-2)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}><Icon name="folder" size={13} /></button>
      </div>
    );
  }

  function EnvironmentTab() {
    const [docker, setDocker] = useState(false);
    return (
      <div style={{ maxWidth: 620 }}>
        <Field label="AIUTOPIA_ROOT" description="Project root directory"><PathInput defaultValue="/opt/ai-utopia" /></Field>
        <Field label="ChromaDB Path" description="Vector database storage"><PathInput defaultValue="/opt/ai-utopia/chroma" /></Field>
        <Field label="Identity DB Path" description="SQLite identity database"><PathInput defaultValue="/opt/ai-utopia/identity.db" /></Field>
        <Field label="Checkpoints Path" description="Model checkpoint storage"><PathInput defaultValue="/opt/ai-utopia/checkpoints" /></Field>
        <Field label="Docker Compose" description="Production deployment mode"><Toggle value={docker} onChange={setDocker} /></Field>
        <div style={{ padding: '12px 0' }}>
          <label className="t-label" style={{ display: 'block', marginBottom: 8 }}>JVM Arguments</label>
          <textarea className="inp full mono" rows={5} style={{ fontSize: 11 }} defaultValue={'-XX:+UseZGC\n-XX:+ZGenerational\n-Xms4G\n-Xmx8G\n-Daiutopia.py4j.port=25099'} />
        </div>
      </div>
    );
  }

  function MemoryTab() {
    const [reranker, setReranker] = useState(false);
    const [sim, setSim] = useState(0.75);
    const [ctx, setCtx] = useState(10);
    const collections = [['mem_01HABC...', '4,231 entries'], ['skill_lib_01HABC...', '892 entries'], ['mem_01HABD...', '3,104 entries'], ['skill_lib_01HABD...', '567 entries']];
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 40 }}>
        <div>
          <Field label="ChromaDB Connection" description="Database connectivity">
            <span className="flex items-center" style={{ gap: 6, height: 32, padding: '0 12px', borderRadius: 9, background: 'rgba(0,229,204,0.08)', border: '1px solid rgba(0,229,204,0.3)', color: 'var(--accent-cyan)', fontSize: 11 }}><Icon name="check" size={12} /> Connected</span>
          </Field>
          <div style={{ marginTop: 14 }}>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Memory Collections</label>
            <div className="flex flex-col" style={{ gap: 6 }}>
              {collections.map(([n, s]) => (
                <div key={n} className="soft flex items-center justify-between" style={{ padding: '9px 12px' }}>
                  <span className="t-code mono">{n}</span><span className="t-caption">{s}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div>
          <Field label="Embedding Model" description="Text embedding provider"><select className="inp" style={{ width: 200 }}><option>all-MiniLM-L6-v2</option><option>text-embedding-3</option><option>Local ONNX runtime</option></select></Field>
          <Field label="Reranker" description="Cross-encoder reranking"><Toggle value={reranker} onChange={setReranker} /></Field>
          <Field label="Similarity Threshold" description="Minimum relevance score"><div style={{ width: 150 }}><Slider value={sim} min={0} max={1} step={0.05} onChange={setSim} format={(v) => v.toFixed(2)} /></div></Field>
          <Field label="Max Context Entries" description="Entries sent to LLM context"><div style={{ width: 150 }}><Slider value={ctx} min={1} max={50} step={1} onChange={setCtx} /></div></Field>
        </div>
      </div>
    );
  }

  function AdvancedTab() {
    const [workers, setWorkers] = useState(4);
    const [bounds, setBounds] = useState(true);
    const coord = (dv) => <input type="number" defaultValue={dv} className="inp" style={{ width: 64, padding: '0 8px' }} />;
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 40 }}>
        <Field label="Determinism Seed" description="Reproducible random seed"><input type="text" defaultValue="42" className="inp" style={{ width: 110 }} /></Field>
        <Field label="Parallel Workers" description="Training worker processes"><div style={{ width: 150 }}><Slider value={workers} min={1} max={8} step={1} onChange={setWorkers} /></div></Field>
        <Field label="Arena Bounds" description="Restrict agent movement area"><Toggle value={bounds} onChange={setBounds} /></Field>
        <Field label="Schema Version" description="Current plan schema"><div className="inp flex items-center" style={{ width: 110, color: 'var(--text-tertiary)' }}>1.0.0</div></Field>
        {bounds && (
          <React.Fragment>
            <Field label="Min Coordinates" description="Arena minimum X, Y, Z"><div className="flex" style={{ gap: 8 }}>{coord(-256)}{coord(0)}{coord(-256)}</div></Field>
            <Field label="Max Coordinates" description="Arena maximum X, Y, Z"><div className="flex" style={{ gap: 8 }}>{coord(256)}{coord(128)}{coord(256)}</div></Field>
          </React.Fragment>
        )}
        <div className="flex items-center" style={{ gridColumn: '1 / -1', gap: 12, marginTop: 18, paddingTop: 18, borderTop: '1px solid var(--border-subtle)' }}>
          <GhostButton color="var(--border-active)" style={{ color: 'var(--text-primary)' }}>Export Database</GhostButton>
          <GhostButton color="var(--border-active)" style={{ color: 'var(--text-primary)' }}>Import Database</GhostButton>
          <div className="flex-1" />
          <GhostButton color="var(--status-offline)" icon="alert">Factory Reset</GhostButton>
        </div>
      </div>
    );
  }

  const content = { system: SystemTab, environment: EnvironmentTab, memory: MemoryTab, advanced: AdvancedTab };

  return function Settings() {
    const [tab, setTab] = useState('system');
    const Body = content[tab];
    return (
      <div className="grid12">
        <Reveal style={{ gridColumn: '1 / -1' }}>
          <Card style={{ padding: 20 }}>
            <div className="flex" style={{ borderBottom: '1px solid var(--border-subtle)', marginBottom: 22, gap: 2 }}>
              {subtabs.map(([k, l, ic]) => (
                <button key={k} onClick={() => setTab(k)} className="flex items-center relative" style={{ gap: 8, padding: '10px 16px', cursor: 'pointer', background: 'transparent', border: 'none', fontSize: 12, color: tab === k ? 'var(--accent-cyan)' : 'var(--text-secondary)', fontWeight: tab === k ? 500 : 400 }}>
                  <Icon name={ic} size={14} />{l}
                  {tab === k && <span style={{ position: 'absolute', bottom: -1, left: 0, right: 0, height: 2, background: 'var(--accent-cyan)' }} />}
                </button>
              ))}
            </div>
            <div className="tab-fade" key={tab}><Body /></div>
          </Card>
        </Reveal>
      </div>
    );
  };
})();
window.Settings = Settings;


