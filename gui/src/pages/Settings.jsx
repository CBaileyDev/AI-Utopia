/* AI Utopia — Settings tab */
import { useState } from 'react';
import { Card, Reveal, Toggle, Slider, Field, GhostButton } from '../components/primitives.jsx';
import { Icon } from '../lib/icons.jsx';

const subtabs = [['system', 'System', 'server'], ['environment', 'Environment', 'globe'], ['memory', 'Memory', 'database'], ['advanced', 'Advanced', 'cpu']];

function SystemTab() {
  const ro = { color: 'var(--text-tertiary)' };
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 40, rowGap: 0 }}>
      <Field label="Py4J Port" description="Port for the Py4J bridge socket"><input type="number" defaultValue={25099} className="inp" style={{ width: 110 }} /></Field>
      <Field label="Minecraft Version" description="Pinned — UnionClef baseline"><div className="inp flex items-center" style={{ width: 110, ...ro }}>1.21.1</div></Field>
      <Field label="Python Version" description="Required runtime"><div className="inp flex items-center" style={{ width: 110, ...ro }}>3.12</div></Field>
      <Field label="Log Level" description="Console verbosity"><select className="inp" style={{ width: 130 }} defaultValue="INFO"><option>DEBUG</option><option>INFO</option><option>WARNING</option><option>ERROR</option></select></Field>
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
        <>
          <Field label="Min Coordinates" description="Arena minimum X, Y, Z"><div className="flex" style={{ gap: 8 }}>{coord(-256)}{coord(0)}{coord(-256)}</div></Field>
          <Field label="Max Coordinates" description="Arena maximum X, Y, Z"><div className="flex" style={{ gap: 8 }}>{coord(256)}{coord(128)}{coord(256)}</div></Field>
        </>
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

export default function Settings() {
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
}
