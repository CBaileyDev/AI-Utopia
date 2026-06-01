/* AI Utopia — Dashboard tab */
import { useState, useEffect } from 'react';
import { Card, Reveal, SectionTitle, StatusDot, Avatar, AnimatedNumber, EmptyState, OfflinePill } from '../components/primitives.jsx';
import { AreaStack, Donut } from '../lib/charts.jsx';
import { Icon } from '../lib/icons.jsx';
import { activity, logs as MOCK_LOGS, roleColors, roleMeta, getRoleColor } from '../mockData.js';
import { useResource } from '../useApi.js';
import { getLogs } from '../api.js';
import { adaptLogs } from '../lib/transforms.js';

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
  const colors = roleColors;
  return (
    <Reveal delay={120} style={{ gridColumn: 'span 8' }}>
      <Card style={{ padding: 18 }}>
        <SectionTitle right={<span className="t-caption">sample · last 7 hrs</span>}>Agent Activity Timeline</SectionTitle>
        <AreaStack data={activity} keys={['gatherer', 'builder', 'farmer', 'defender']} colors={colors} height={196} />
        <div className="flex items-center" style={{ gap: 18, marginTop: 12 }}>
          {Object.keys(roleMeta).map((r) => (
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

function RewardDistribution({ agents }) {
  const total = agents.reduce((s, a) => s + a.rewards, 0);
  const data = agents.map((a) => ({ name: a.name, value: a.rewards, color: getRoleColor(a.role) }));
  const sorted = [...agents].sort((a, b) => b.rewards - a.rewards);
  if (!agents.length) {
    return (
      <Reveal delay={150} style={{ gridColumn: 'span 4' }}>
        <Card style={{ padding: 18, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <SectionTitle>Reward Distribution</SectionTitle>
          <EmptyState icon="trophy" title="No rewards yet" hint="Reward totals appear once agents are spawned and earning." />
        </Card>
      </Reveal>
    );
  }
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
            {sorted.map((a) => (
              <div key={a.id} className="flex items-center justify-between">
                <div className="flex items-center" style={{ gap: 8 }}>
                  <span style={{ width: 7, height: 7, borderRadius: 999, background: getRoleColor(a.role) }} />
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
  const { data: feedLogs, online } = useResource(getLogs, {
    fallback: MOCK_LOGS, transform: adaptLogs, pollMs: 3000,
  });
  const indicator = online
    ? <span className="flex items-center t-caption" style={{ gap: 6, color: 'var(--accent-cyan)' }}><StatusDot status="alive" size={6} /> LIVE</span>
    : <OfflinePill />;
  return (
    <Reveal delay={180} style={{ gridColumn: 'span 7' }}>
      <Card style={{ padding: 18, display: 'flex', flexDirection: 'column', height: 360 }}>
        <SectionTitle right={indicator}>Live Activity Feed</SectionTitle>
        <div className="flex-1" style={{ overflowY: 'auto', marginRight: -6, paddingRight: 6 }}>
          {feedLogs.length === 0 ? (
            <EmptyState icon="activity" title="No log activity yet" hint="Bridge events and training output stream here as they happen." />
          ) : feedLogs.map((log, i) => (
            <div key={log.id} className="flex items-start" style={{ gap: 10, padding: '6px 4px', borderRadius: 6, background: i % 2 ? 'transparent' : 'rgba(255,255,255,0.012)' }}>
              <span className="t-caption mono" style={{ minWidth: 52 }}>{log.timestamp}</span>
              <span className="t-caption mono" style={{ color: typeColors[log.type] || 'var(--text-secondary)', minWidth: 50, fontWeight: 500 }}>[{log.type}]</span>
              <span className="t-body" style={{ lineHeight: 1.5 }}>{log.message}</span>
            </div>
          ))}
        </div>
      </Card>
    </Reveal>
  );
}

function ActiveAgents({ onOpen, agents, agentsOnline }) {
  const right = agentsOnline
    ? <span className="t-caption">{agents.length} online</span>
    : <OfflinePill />;
  return (
    <Reveal delay={210} style={{ gridColumn: 'span 5' }}>
      <Card style={{ padding: 18, height: 360, display: 'flex', flexDirection: 'column' }}>
        <SectionTitle right={right}>Active Agents</SectionTitle>
        <div className="flex flex-col flex-1" style={{ gap: 9 }}>
          {agents.length === 0 && (
            <EmptyState icon="users" title="No agents spawned" hint="Spawn agents from the Bot Config tab to populate the village." />
          )}
          {agents.map((a) => (
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

export default function Dashboard({ onOpenAgent, agents = [], agentsOnline = false }) {
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
      <Metric icon="users" value={agents.length} animated label="Active Agents" sub={agentsOnline ? 'live roster' : 'sample data'} trend={agents.length ? 'up' : 'neutral'} delay={0} />
      <Metric icon="brain" value={epochs} animated label="Epochs Completed" sub="sample · Batch 64 / 128" trend="neutral" delay={40} />
      <Metric icon="database" value={8492} animated label="Memory Entries" sub="sample · ChromaDB" trend="neutral" delay={80} />
      <Metric icon="clock" value={timeStr} label="World Time" sub="sample · Day 1,247" trend="neutral" delay={120} />
      <ActivityTimeline />
      <RewardDistribution agents={agents} />
      <ActivityFeed />
      <ActiveAgents onOpen={onOpenAgent} agents={agents} agentsOnline={agentsOnline} />
    </div>
  );
}
