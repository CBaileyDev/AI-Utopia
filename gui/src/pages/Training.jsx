/* AI Utopia — Training tab */
import { useState } from 'react';
import { Card, Reveal, SectionTitle, Slider, Segmented, GhostButton } from '../components/primitives.jsx';
import { Sparkline, LineChart } from '../lib/charts.jsx';
import { Icon } from '../lib/icons.jsx';
import { epochs as epochData, rewardCurve } from '../mockData.js';

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
        <LineChart data={rewardCurve} baseline={1.8} height={272} />
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
  const sorted = [...epochData].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (typeof x === 'number') return dir === 'asc' ? x - y : y - x;
    return dir === 'asc' ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
  });
  const sc = { improved: 'var(--accent-cyan)', stable: 'var(--status-training)', degraded: 'var(--status-offline)' };
  const click = (k) => { if (sortKey === k) setDir(dir === 'asc' ? 'desc' : 'asc'); else { setSortKey(k); setDir('desc'); } };
  return (
    <Reveal delay={200} style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 18 }}>
        <SectionTitle right={<span className="t-caption">{epochData.length} epochs</span>}>Epoch History</SectionTitle>
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

export default function Training() {
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
}
