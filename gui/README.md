# AI Utopia — Control Center (GUI)

A dark, premium desktop-style control center for the AI Utopia multi-agent
Minecraft village. Vite + React 18, hand-written CSS (no Tailwind), plain SVG
charts. This is a **faithful recreation of the design prototype**
(`.design/aiutopia/project/AI Utopia.html`) restructured into idiomatic ES
modules, and is packaged as a **standalone Windows desktop app via Tauri v2**
(`src-tauri/`) — the in-app titlebar drives the real OS window.

> **Status:** front end only. Every value on screen comes from `src/mockData.js`.
> Nothing is wired to the Python backend yet — that is a deliberate later step
> (see **BACKEND WIRING — TODO** below).

## Run it

### As a desktop app (Tauri — native window)

```bash
cd gui
npm install
npm run tauri:dev     # launches the native desktop window (Vite + Rust webview)
npm run tauri:build   # compiles the standalone Windows .exe + NSIS installer
```

`tauri:build` outputs:

- **Standalone exe:** `src-tauri/target/release/aiutopia-control-center.exe`
  (the binary name follows the Cargo package name; ~3 MB, double-click to run)
- **NSIS installer:** `src-tauri/target/release/bundle/nsis/AI Utopia_0.1.0_x64-setup.exe`
  (~1.2 MB; installs per-user, Start-menu shortcut as "AI Utopia")

Requires the Rust toolchain (cargo) and the WebView2 runtime (preinstalled on
Windows 11). First build is a cold Rust compile and takes several minutes;
later builds are incremental.

The titlebar (minimize / maximize / close + the drag region) is wired to the
**native OS window** via `@tauri-apps/api/window` — the window is borderless
(`decorations: false`) and the React app draws all its own chrome. The window
controls feature-detect Tauri (`window.__TAURI_INTERNALS__`), so the exact same
code still runs as a plain browser app (`npm run dev`) where the buttons are
no-ops. The window-control permissions live in
`src-tauri/capabilities/default.json` (`core:window:allow-minimize` /
`allow-toggle-maximize` / `allow-close` / `allow-start-dragging`).

### As a plain web app (browser)

```bash
cd gui
npm install
npm run dev       # dev server → http://localhost:5173
npm run build     # production static build → dist/  (verified passing)
npm run preview   # serve the production build
```

Requires Node 18+ (developed on Node 24, npm 11). No backend, no env vars.

## Fonts

Geist + Geist Mono are **bundled locally** (latin-subset variable `woff2` in
`src/assets/fonts/`, `@font-face`'d in `src/styles.css`) so the standalone
desktop `.exe` renders identically with **no network**. The font stacks keep
system monospace/sans fallbacks (`'Geist Mono','SF Mono',ui-monospace,monospace`
and `'Geist',sans-serif`) if a file is ever missing. There is no longer any
Google Fonts `<link>` in `index.html`.

## Structure

```
gui/
├─ index.html                # #root + module entry (fonts bundled, no CDN link)
├─ vite.config.js            # @vitejs/plugin-react, nothing exotic
├─ src-tauri/                # Tauri v2 desktop shell (Rust)
│  ├─ tauri.conf.json        # borderless dark window, 1280×820, identifier, NSIS bundle
│  ├─ Cargo.toml             # tauri 2 deps; release profile (lto, strip)
│  ├─ build.rs               # tauri_build::build()
│  ├─ src/{main,lib}.rs      # builder entry (no custom commands — window API is core)
│  ├─ capabilities/default.json  # window-control permissions for the "main" window
│  ├─ icons/                 # generated icon set (icon.ico/.png/.icns + source.png)
│  └─ gen-icon.py            # one-shot Pillow script that drew the source logo PNG
├─ src/
│  ├─ main.jsx               # ReactDOM.createRoot → <App/>, imports styles.css
│  ├─ assets/fonts/          # bundled Geist + Geist Mono woff2 (offline-resilient)
│  ├─ lib/tauri.js           # native-window helpers (minimize/maximize/close), Tauri-guarded
│  ├─ App.jsx                # window shell: titlebar, TabBar (animated indicator),
│  │                         #   content header, statusbar. useState tab-switching.
│  ├─ styles.css             # the prototype <style> block, ported VERBATIM
│  │                         #   (pixel-perfect source of truth — color tokens,
│  │                         #   layout, animations). Do not "tidy".
│  ├─ mockData.js            # ★ the single data seam (see below)
│  ├─ lib/
│  │  ├─ icons.jsx           # <Icon name=.../> — lucide paths (MIT)
│  │  └─ charts.jsx          # AreaStack / Donut / LineChart / Sparkline (pure SVG)
│  ├─ components/
│  │  └─ primitives.jsx      # Reveal, Card, SectionTitle, StatusDot, RoleBadge,
│  │                         #   Avatar, Toggle, Slider, Segmented, GhostButton,
│  │                         #   AnimatedNumber, Field
│  └─ pages/
│     ├─ Dashboard.jsx       # metrics, activity timeline, reward donut, live feed,
│     │                      #   active-agents list (opens Spectate on click)
│     ├─ BotConfig.jsx       # spawn controller, agent params form, identity preview
│     ├─ Training.jsx        # pipeline stepper, metric cards, reward chart,
│     │                      #   controls, sortable epoch table
│     ├─ Spectate.jsx        # live canvas world viewport, agent inspector
│     │                      #   (State/Memory/Plan), chat panel
│     └─ Settings.jsx        # System / Environment / Memory / Advanced subtabs
```

### How it differs from the prototype (intentional)

The prototype was a single self-contained HTML file using React UMD, Babel
standalone, and `window.X` globals to share code between inlined script blocks.
This app reproduces the **visual output exactly** but uses proper ES module
imports instead of globals — components in `src/components/`, charts/icons in
`src/lib/`, pages in `src/pages/`, data in `src/mockData.js`. No router and no
state library were added: the prototype is simple `useState` tab-switching and
this app keeps it that way.

## All five tabs are fully built

| Tab | Status | Notes |
|-----|--------|-------|
| **Dashboard** | Complete | 4 animated metric cards, stacked-area activity timeline, reward donut, scrolling live feed, active-agents list (click → Spectate). |
| **Bot Config** | Complete | Role-select spawn bar, parameters form (selects, temperature slider, system-prompt textarea, memory toggles), live identity-preview card. |
| **Training** | Complete | Animated pipeline stepper (ping ring + spinner), 5 metric cards w/ sparkline, reward line chart (gradient fill + dashed baseline), controls (ghost buttons, LR/gamma sliders, batch segmented), sortable epoch table. |
| **Spectate** | Complete | Animated `<canvas>` world viewport (agent dots + motion trails + village ring), inspector w/ State/Memory/Plan tabs, working chat panel (simulated planner reply). |
| **Settings** | Complete | System / Environment / Memory / Advanced subtabs, all fields/toggles/sliders. |

Polish carried over: window-in animation, per-tab `Reveal` stagger, the sliding
glowing tab indicator, `AnimatedNumber` eases, `tab-fade` replay on tab switch
(via `key={active}` remount), hover transitions on cards/rows.

## Backend wiring — TODO

Everything renders from `src/mockData.js`. That module is the **single seam**: a
future step replaces the hard-coded constants with real data (e.g. a small
FastAPI service over the existing Py4J bridge), and **no component needs to
change** as long as the data shapes stay the same. The recommended path is a
thin `src/api.js` that returns the same shapes (today: the mock constants;
later: `fetch()` calls), so pages import from one place.

| Mock export (`src/mockData.js`) | Used by | Real source it should map to |
|---|---|---|
| `agents` | Dashboard (active agents, reward donut), BotConfig (identity preview), Spectate (viewport, inspector, chat list) | `identity.db` (ULID, role, skin, born) + live bridge state (`observationsAll()`: position, health, hunger, status) via the CLI `agent list` / a FastAPI `/agents` endpoint |
| `logs` | Dashboard live activity feed | Tailed training/agent log stream (`Research/train-*.log`) or a `/logs` SSE/websocket endpoint |
| `epochs` | Training epoch-history table | Training `progress.csv` / Ray `result.json` per-iteration metrics (mean reward, policy/value loss, entropy, KL) |
| `rewardCurve` | Training reward-over-time chart | Same training results history (reward per iteration) + the eval-gate baseline |
| `activity` | Dashboard activity timeline | Per-role action counts aggregated over time from the log/metrics stream |
| `roleColors` / `roleMeta` / `getRoleColor` / `statusColor` | Everywhere (theming) | Static config — stays client-side; no backend needed |

Interactive controls that are currently **inert mock UI** and would later issue
commands:

- **BotConfig → Spawn / Apply / Terminate** → `aiutopia agent spawn --role …`,
  identity update, terminate (CLI: `aiutopia.cli.app agent`).
- **Training → Pause / Save Checkpoint / Promote Weights / Emergency Stop** →
  `scripts/train.py` lifecycle + `promote-weights` (the §5.10 checklist).
- **Spectate → chat send** → the `@<agent_name>` chat router on the Java side /
  planner; the reply is currently a canned per-role string.
- **Settings** fields → write to `src/aiutopia/common/config.py`-backed config
  (Py4J port, paths, seed, workers, arena bounds, JVM args). Treat the Python
  src as read-only from here; route through an API, do not edit configs directly
  from the browser without a backend mediator.

## Notes

- `node_modules/` and `dist/` are git-ignored (`gui/.gitignore`).
- Two `npm audit` moderate advisories come from transitive dev-only deps in the
  Vite toolchain; they don't affect the shipped static bundle.
