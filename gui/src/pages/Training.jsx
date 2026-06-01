/* AI Utopia — Training tab */
import { useState } from 'react';
import { Card, Reveal, SectionTitle, Slider, Segmented, GhostButton, EmptyState, OfflinePill, Toasts } from '../components/primitives.jsx';
import { Sparkline, LineChart } from '../lib/charts.jsx';
import { Icon } from '../lib/icons.jsx';
import { epochs as MOCK_EPOCHS, rewardCurve as MOCK_REWARD_CURVE } from '../mockData.js';
import { useResource } from '../useApi.js';
import { getTrainingStatus, getTrainingRuns, startTraining, stopTraining } from '../api.js';
import { adaptEpochs, adaptRewardCurve, fmt } from '../lib/transforms.js';

function Pipeline({ status, online }) {
  const running = !!status?.running;
  const iter = status?.iter ?? 0;
  const maxIters = status?.max_iters ?? 0;
  const pct = maxIters ? iter / maxIters : 0;
  // Derive a 4-stage pipeline view from real progress.
  const stages = [
    { name: 'Environment Setup', status: running || iter > 0 ? 'complete' : 'pending' },
    { name: 'Observation Feed', status: running ? 'active' : iter > 0 ? 'complete' : 'pending' },
    { name: 'Policy Update', status: running && pct > 0.05 ? 'active' : iter > 0 && !running ? 'complete' : 'pending' },
    { name: 'Weight Promotion', status: pct >= 1 ? 'complete' : 'pending' },
  ];
  return (
    <Reveal style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 20 }}>
        <SectionTitle right={online ? <span className="t-caption">PPO + LSTM{status?.backend ? ` · ${status.backend}` : ''}</span> : <OfflinePill />}>Training Pipeline</SectionTitle>
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
          <span className="t-body">
            {running ? `Pipeline active — iter ${iter}/${maxIters}` : iter > 0 ? `Run complete — ${iter} iters` : 'Pipeline idle'}
          </span>
          {running && <span className="t-data mono" style={{ fontSize: 16, color: 'var(--accent-cyan)' }}>{status?.sps != null ? `${status.sps.toFixed(0)} sps` : '—'}</span>}
        </div>
      </Card>
    </Reveal>
  );
}

function Metrics({ status, history }) {
  const m = status?.metrics || {};
  // Spark from real recent return history when available, else a gentle mock.
  const spark = history && history.length > 1
    ? history.slice(-28).map((h) => h.return_mean)
    : Array.from({ length: 28 }, (_, i) => 2 + Math.sin(i * 0.4) * 0.3 + Math.sin(i * 1.3) * 0.08);
  const items = [
    { label: 'Mean Reward', value: fmt(m.return_mean, 3, '+'), trend: 'up', spark: true },
    { label: 'Entropy', value: fmt(m.entropy, 3), trend: 'up' },
    { label: 'KL Divergence', value: fmt(m.kl, 4), trend: 'down' },
    { label: 'Clip Fraction', value: m.clipfrac == null ? 'n/a' : fmt(m.clipfrac, 4), trend: 'neutral' },
    { label: 'Term Rate', value: m.term_rate == null ? 'n/a' : fmt(m.term_rate, 3), trend: 'neutral' },
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

function RewardChart({ curve, online }) {
  const hasData = Array.isArray(curve) && curve.length >= 2;
  // baseline: mean of the series, so the dashed line sits sensibly for any scale.
  const baseline = hasData ? curve.reduce((s, d) => s + d.reward, 0) / curve.length : null;
  return (
    <Reveal delay={120} style={{ gridColumn: 'span 5' }}>
      <Card style={{ padding: 18, height: '100%', display: 'flex', flexDirection: 'column' }}>
        <SectionTitle right={online
          ? (hasData ? <span className="t-caption flex items-center" style={{ gap: 6 }}><span style={{ width: 14, height: 0, borderTop: '1px dashed var(--text-tertiary)', display: 'inline-block' }} />mean {baseline.toFixed(2)}</span> : <span className="t-caption">live</span>)
          : <OfflinePill />}>Reward Over Time</SectionTitle>
        {hasData
          ? <LineChart data={curve} baseline={baseline} height={272} />
          : <EmptyState icon="pulse" title="No reward history" hint="The reward curve appears once a run has logged iterations." />}
      </Card>
    </Reveal>
  );
}

function FlagToggle({ label, value, onChange }) {
  return (
    <button onClick={() => onChange(!value)} className="flex items-center justify-between" style={{
      width: '100%', padding: '8px 10px', borderRadius: 8, cursor: 'pointer', textAlign: 'left',
      background: value ? 'rgba(0,229,204,0.06)' : 'var(--surface-2)',
      border: `1px solid ${value ? 'rgba(0,229,204,0.3)' : 'var(--border-subtle)'}`,
    }}>
      <span className="t-body" style={{ fontSize: 11 }}>{label}</span>
      <span style={{ width: 7, height: 7, borderRadius: 999, background: value ? 'var(--accent-cyan)' : 'var(--border-active)' }} />
    </button>
  );
}

function Controls({ status, onStart, onStop, busy }) {
  const running = !!status?.running;
  const [backend, setBackend] = useState('sim');
  const [iters, setIters] = useState(200);
  const [numEnvs, setNumEnvs] = useState(4);
  const [entropy, setEntropy] = useState(0); // ×1e-3, 0 = default
  const [flags, setFlags] = useState({ spawn_jitter: false, approach_shaping: false, force_masked_spawn: false });

  const start = () => {
    const opts = { backend, iters, num_envs: numEnvs, ...flags };
    if (entropy > 0) opts.entropy_coeff = entropy / 1000;
    onStart(opts);
  };

  return (
    <Reveal delay={160} style={{ gridColumn: 'span 3' }}>
      <Card style={{ padding: 18, height: '100%' }}>
        <SectionTitle>Controls</SectionTitle>
        <div className="flex flex-col" style={{ gap: 10 }}>
          {running ? (
            <GhostButton color="var(--status-offline)" icon={busy ? 'loader' : 'stop'} full onClick={onStop}>
              {busy ? 'Stopping…' : 'Stop Training'}
            </GhostButton>
          ) : (
            <GhostButton color="var(--status-training)" icon={busy ? 'loader' : 'play'} full onClick={start}>
              {busy ? 'Starting…' : 'Start Training'}
            </GhostButton>
          )}
        </div>
        <div className="flex flex-col" style={{ gap: 16, marginTop: 20 }}>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Backend</label>
            <Segmented options={[{ value: 'sim', label: 'Sim' }, { value: 'real', label: 'Real MC' }]} value={backend} onChange={setBackend} />
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Iterations</label>
            <Slider value={iters} min={10} max={400} step={10} onChange={setIters} />
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Num Envs</label>
            <Segmented options={[1, 2, 4, 8]} value={numEnvs} onChange={setNumEnvs} />
          </div>
          <div>
            <label className="t-label" style={{ display: 'block', marginBottom: 9 }}>Entropy Coeff</label>
            <Slider value={entropy} min={0} max={10} step={1} onChange={setEntropy} format={(v) => (v === 0 ? 'default' : `${v}e-3`)} />
          </div>
          <div className="flex flex-col" style={{ gap: 7 }}>
            <FlagToggle label="Spawn jitter" value={flags.spawn_jitter} onChange={(v) => setFlags((f) => ({ ...f, spawn_jitter: v }))} />
            <FlagToggle label="Approach shaping" value={flags.approach_shaping} onChange={(v) => setFlags((f) => ({ ...f, approach_shaping: v }))} />
            <FlagToggle label="Force masked spawn" value={flags.force_masked_spawn} onChange={(v) => setFlags((f) => ({ ...f, force_masked_spawn: v }))} />
          </div>
        </div>
      </Card>
    </Reveal>
  );
}

function RunSelector({ runs, online, selected, onSelect }) {
  return (
    <Reveal delay={40} style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 16 }}>
        <div className="flex items-center wrap" style={{ gap: 12 }}>
          <span className="t-label">Run</span>
          {online ? (
            runs.length === 0 ? <span className="t-caption">no runs found</span> : runs.map((r) => {
              const on = r.run_id === selected;
              return (
                <button key={r.run_id} onClick={() => onSelect(r.run_id)} className="flex items-center" style={{
                  gap: 8, height: 32, padding: '0 12px', borderRadius: 8, cursor: 'pointer',
                  background: on ? 'rgba(0,229,204,0.08)' : 'var(--surface-2)',
                  border: `1px solid ${on ? 'rgba(0,229,204,0.35)' : 'var(--border-subtle)'}`,
                  color: on ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                }}>
                  <span className="t-body mono" style={{ fontSize: 11 }}>{r.run_id}</span>
                  <span className="t-caption mono">{r.backend} · {r.iters}it{r.last_return != null ? ` · ${r.last_return.toFixed(1)}` : ''}</span>
                </button>
              );
            })
          ) : <OfflinePill />}
        </div>
      </Card>
    </Reveal>
  );
}

function EpochTable({ rows, online }) {
  const [sortKey, setSortKey] = useState('epoch');
  const [dir, setDir] = useState('desc');
  const headers = [['epoch', 'Iter'], ['meanReward', 'Return Mean'], ['entropy', 'Entropy'], ['klDiv', 'KL Div'], ['policyLoss', 'Policy Loss'], ['valueLoss', 'Value Loss'], ['duration', 'Duration'], ['status', 'Status']];
  const sorted = [...rows].sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    if (typeof x === 'number') return dir === 'asc' ? x - y : y - x;
    return dir === 'asc' ? String(x).localeCompare(String(y)) : String(y).localeCompare(String(x));
  });
  const sc = { improved: 'var(--accent-cyan)', stable: 'var(--status-training)', degraded: 'var(--status-offline)' };
  const click = (k) => { if (sortKey === k) setDir(dir === 'asc' ? 'desc' : 'asc'); else { setSortKey(k); setDir('desc'); } };
  const num = (v, d) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(d));
  return (
    <Reveal delay={200} style={{ gridColumn: '1 / -1' }}>
      <Card style={{ padding: 18 }}>
        <SectionTitle right={online ? <span className="t-caption">{rows.length} iterations</span> : <OfflinePill />}>Iteration History</SectionTitle>
        {rows.length === 0 ? (
          <EmptyState icon="pulse" title="No iteration history" hint="Per-iteration return / entropy / KL appear once a run logs progress." />
        ) : (
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
                  <td className="t-body mono" style={{ padding: '9px 12px', color: 'var(--accent-cyan)' }}>{e.meanReward == null ? '—' : `+${e.meanReward.toFixed(2)}`}</td>
                  <td className="t-body mono" style={{ padding: '9px 12px' }}>{num(e.entropy, 3)}</td>
                  <td className="t-body mono" style={{ padding: '9px 12px' }}>{num(e.klDiv, 4)}</td>
                  <td className="t-body mono" style={{ padding: '9px 12px', color: 'var(--text-tertiary)' }}>{num(e.policyLoss, 4)}</td>
                  <td className="t-body mono" style={{ padding: '9px 12px', color: 'var(--text-tertiary)' }}>{num(e.valueLoss, 4)}</td>
                  <td className="t-caption mono" style={{ padding: '9px 12px' }}>{e.duration ?? '—'}</td>
                  <td style={{ padding: '9px 12px' }}>
                    {e.status ? (
                      <span className="flex items-center" style={{ gap: 7 }}>
                        <span style={{ width: 7, height: 7, borderRadius: 999, background: sc[e.status] || 'var(--text-tertiary)' }} />
                        <span className="t-caption capitalize">{e.status}</span>
                      </span>
                    ) : <span className="t-caption">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        )}
      </Card>
    </Reveal>
  );
}

const STATUS_FALLBACK = { running: false, iter: 0, max_iters: 0, metrics: {}, history: [] };

let _tToastId = 0;

export default function Training() {
  // Poll status fast (2s) while the tab is mounted; runs slower (8s).
  const { data: status, online, refetch: refetchStatus } = useResource(getTrainingStatus, {
    fallback: STATUS_FALLBACK, pollMs: 2000,
  });
  const { data: runs, online: runsOnline } = useResource(getTrainingRuns, { fallback: [], pollMs: 8000 });
  const [selectedRun, setSelectedRun] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toasts, setToasts] = useState([]);

  const pushToast = (kind, message) => {
    const id = ++_tToastId;
    setToasts((t) => [...t, { id, kind, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  };
  const dismiss = (id) => setToasts((t) => t.filter((x) => x.id !== id));

  // History/metrics come from /status (live run, or the latest done run the
  // backend reports). When offline, fall back to the mock shapes.
  const history = online ? (status.history || []) : null;
  const epochRows = online ? adaptEpochs(history) : MOCK_EPOCHS;
  const rewardCurve = online ? adaptRewardCurve(history) : MOCK_REWARD_CURVE;

  const handleStart = async (opts) => {
    setBusy(true);
    try {
      const res = await startTraining(opts);
      if (res && res.ok === false) pushToast('error', `Start failed: ${res.error || 'unknown'}`);
      else pushToast('success', `Training started${res?.run_id ? ` · ${res.run_id}` : ''}`);
    } catch (e) {
      pushToast('error', e.detail ? `Start failed: ${e.detail}` : 'Start failed — backend unreachable');
    } finally {
      setBusy(false);
      refetchStatus();
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await stopTraining();
      pushToast('success', 'Training stop requested');
    } catch (e) {
      pushToast('error', e.detail ? `Stop failed: ${e.detail}` : 'Stop failed — backend unreachable');
    } finally {
      setBusy(false);
      refetchStatus();
    }
  };

  return (
    <div className="grid12">
      <RunSelector runs={runs} online={runsOnline} selected={selectedRun ?? status?.run_id} onSelect={setSelectedRun} />
      <Pipeline status={status} online={online} />
      <Metrics status={status} history={history} />
      <RewardChart curve={rewardCurve} online={online} />
      <Controls status={status} onStart={handleStart} onStop={handleStop} busy={busy} />
      <EpochTable rows={epochRows} online={online} />
      <Toasts items={toasts} onDismiss={dismiss} />
    </div>
  );
}
