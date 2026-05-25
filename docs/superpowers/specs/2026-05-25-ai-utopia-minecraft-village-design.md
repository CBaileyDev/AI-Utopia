# AI Utopia — Multi-Agent Minecraft AI Village Design Spec

**Date:** 2026-05-25
**Status:** Design approved; ready for implementation planning
**Owner:** Carte (`barker_carter@icloud.com`)
**Successor doc:** `IMPLEMENTATION_PLAN.md` (produced by `writing-plans` skill)
**Research basis:** `GeminiResearch.md`; `Kimi_Agent_Minecraft AI Village Research/` (final report + 10 research-dimension files + cross-verification + 10 cross-dimensional insights)

---

## Executive Summary

A persistent multi-agent reinforcement-learning system in which four specialized AI agents — **gatherer, builder, farmer, defender** — cooperatively grow and operate a Minecraft Java village. Agents are coordinated by an event-driven LLM planner that emits typed JSON subgoal DAGs; per-role PPO policies execute parameterized skill primitives at 20 Hz. Real human friends can join the world, are recognized as friendly entities, and can converse with agents via in-chat `@<agent_name>` messages. Agents have persistent identity across server restarts including per-agent skill libraries, episodic memory, and social memory of individual players.

### Locked scope (from Q1–Q5)

| Dimension | Decision |
|---|---|
| Society scope | A (cooperative village runs indefinitely) + F (persistent identity, Level D — social) |
| Stretch | C (agent↔agent economy) post-M6 |
| Player interaction | B (recognized as world citizens) + C (NL chat via `@name`) |
| Commitment | ~10–15 hr/wk, ~22–28 weeks to M6 |
| Hardware | Ryzen 9950X3D (16c/32t), 64 GB DDR5, RTX 4080 16 GB |
| Runtime LLM budget | $40–80 / month (Claude Haiku), local Qwen 14B fallback path |
| Dev LLM tooling | $0 incremental (Claude Max 20x, Codex, Cursor Pro, Gemini Pro, Kimi Code) |
| Bedrock support | **No.** Java-only, client-side Fabric mods OK. Server is private + whitelist friends. |

### Locked stack

- **MARL framework:** PettingZoo Parallel API + Ray RLlib direct (MARLlib rejected — unmaintained since late 2023)
- **Algorithm:** IPPO start → MAPPO at first cooperative milestone *if IPPO+CTDE baseline shows instability* (HeMAC suggests IPPO often suffices in heterogeneous scenarios) → HAPPO only if heterogeneity causes policy collapse (sec07 trigger)
- **LLM planner:** Claude Haiku API, event-driven (5–20 calls/hr); auto-fallback to local Qwen 14B on budget cap; halt + alert on Qwen unavailability
- **Bridge:** Fork of UnionClef Py4J pattern (MC 1.21.x Fabric)
- **Identity bridge in production:** Carpet `/player <name> spawn` fake players driven via Py4J
- **Observations:** Hybrid — symbolic primary (universal), pixel patches role-specific (builder 64×64 from M2; defender 4× 32×32 from M4 only on plateau)
- **Action space:** Per-role Dict schemas, parameterized skill primitives, hard `-inf` logit masking
- **Embedding (goal spec):** Frozen pre-trained BGE-small (384-d); hard role dispatch (no gating network)
- **Memory:** Chroma vector DB; importance × recency_decay retrieval; per-agent collections
- **Identity DB:** SQLite (`identity.db`, `planner_state.db`)
- **Dashboard:** Tauri client + FastAPI SSE sidecar
- **Server stack:** Fabric 1.21.x + Carpet + Lithium + FerriteCore + Krypton, **Generational ZGC** (not G1GC), Aikar JVM flags adapted for ZGC

### Two-world topology

- **Training world** — 4 disposable Fabric instances, tick-warped, resettable (`/tick sprint`), 8 cores, Ports 25001–25004. RLlib EnvRunners (`num_env_runners=4 × num_envs_per_env_runner=2 = 8` concurrent envs).
- **Production world** — 1 persistent Fabric instance, 20 TPS, never resets. Port 25100. 4 living agents with frozen policy weights but growing skill libraries and episodic memory. Friends join here.
- **Promotion** — manual TensorBoard + held-out scenario review in v1; CLI `aiutopia promote-weights`; hot-reload at next subgoal boundary preserving LSTM hidden state.

### Milestone summary

| M | Weeks | Phase | Deliverable | Eval gate |
|---|---|---|---|---|
| M1 | 4–6 | Solo pretraining | Gatherer alone collects wood | 80% success "64 oak logs" in 1000 steps, 3 consecutive evals |
| M2 | 4–6 | Solo→Coop | + Builder; stub LLM planner emits paired subgoals | 50% cooperative `structure_built_correctly` in 2000-step episodes |
| M3 | 3–4 | Coop scaling | + Farmer (3 roles, CTDE) | `food_security > 0.8` over 5 days with structure work proceeding |
| M4 | 3–4 | Coop scaling | + Defender (4 roles); seed strategy mix-in (3 fixed + 1 random) | 70% mob kill rate inside perimeter; zero villager deaths to hostile mobs in 7-day eval |
| M5 | 6–8 | LLM integration + Level-D identity | Real Claude Haiku planner replaces stub; identity layer folded in | End-to-end "build a village" completed without manual subgoal injection over 14 in-game days |
| M6 | 4–6 | Production deployment | 6 parallel training instances; live production server with friends invited | 30 in-game days, no major instability; plan-cache & memory healthy |

Total: 24–34 weeks at 10–15 hr/wk; with 9950X3D upgrade vs research's 9800X3D baseline, expect ~22–28 weeks.

### Showstopper risks

1. **LLM→RL interface layer is unimplemented in prior art.** Highest engineering risk; covered by §2.1 (frozen BGE + hard role dispatch) + §5.1 (typed Pydantic v2 schemas) + §5.7 (DAG state machine). Budget 2–3× engineering time for M5.
2. **Multi-agent coordination in Minecraft is effectively unresearched.** No validated baselines; first 3 months of cooperative milestones are research, not engineering.
3. **Reward hacking** — see §4.5 (six-type exploit catalog) + §3.5 (hard action masking).
4. **Carpet fake players visible in `/list` and kickable by op** — mitigated by required guard mod (mixin on `PlayerListS2CPacket` + `KickPlayerCommand`).

---

## 1. Scope, Constraints &amp; Non-Goals

### 1.1 In scope (locked)

- Cooperative 4-role village that runs indefinitely on a persistent Java Fabric server.
- Level-D persistent identity: name, skin, inventory, position, role policy weights (per-role, not per-agent), skill library and episodic memory (per-agent), and per-player relationship memory (same episodic schema, queried by `participants` filter).
- Player interaction: friends join as Java players, are recognized as friendly entities, can NL-chat with agents via `@<agent_name>` and receive contextual replies.
- Permadeath with role-policy inheritance and delayed (next in-game morning) succession; funeral events written to surviving agents' episodic memory.
- Friend whitelist with op admin powers.

### 1.2 Stretch (post-M6)

- Agent↔agent internal economy: trade skill primitives, cross-role goal-embedding sharing, scarcity-driven specialization. Requires policy retraining on extended action space.

### 1.3 Out of scope (explicit non-goals)

- Bedrock client support (architecturally permanent — eliminates server-side-only constraint).
- Public hosted server / live ops at scale.
- Vanilla villager-style trade GUI for players ↔ agents (NL chat covers the interaction need).
- Player-issued work tickets / job board.
- Emergent role specialization (roles are fixed at agent spawn).
- Continual learning on the production server (training is two-world; production gets frozen weights).

### 1.4 Hardware envelope

| Component | Allocation |
|---|---|
| 9950X3D cores 0–7 | 4 Fabric training instances + Carpet tick-warp (~12 GB RAM) |
| 9950X3D cores 8–11 | Ray RLlib EnvRunners (4 workers × 2 envs) |
| 9950X3D cores 12–13 | Production Fabric server (~2–3 GB RAM, 25% one core idle, headroom for friends) |
| 9950X3D core 14 | Production LLM planner process |
| 9950X3D core 15 | Chroma + dashboard SSE sidecar |
| RTX 4080 16 GB | RLlib learner 4–6 GB during training; 6–10 GB headroom for local Qwen 14B fallback |
| 64 GB DDR5 | Sums to ~38–46 GB in use, ~18–26 GB headroom |

### 1.5 Budget envelope

- **Runtime LLM:** $40–80 / month. Hard cap enforced in planner; on cap, switch to local Qwen 14B + alert; on Qwen unavailability, halt planner (pause active plans) + page operator. RL policies continue current subgoals; no new subgoals dispatched.
- **Dev LLM:** $0 incremental — Claude Max 20x, Codex, Cursor Pro, Gemini Pro, Kimi Code already covered.

---

## 2. System Topology

### 2.1 Physical layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING WORLD (disposable, tick-warped, resettable)                        │
│                                                                              │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│   │ Fabric srv 1 │ │ Fabric srv 2 │ │ Fabric srv 3 │ │ Fabric srv 4 │        │
│   │ Carpet+tick  │ │ Carpet+tick  │ │ Carpet+tick  │ │ Carpet+tick  │        │
│   │ Lithium/Ferr │ │ Lithium/Ferr │ │ Lithium/Ferr │ │ Lithium/Ferr │        │
│   │ Py4J:25001   │ │ Py4J:25002   │ │ Py4J:25003   │ │ Py4J:25004   │        │
│   │ cores 0-1    │ │ cores 2-3    │ │ cores 4-5    │ │ cores 6-7    │        │
│   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘        │
│          │                │                │                │                │
│          └────────────────┴────────────────┴────────────────┘                │
│                                  │                                           │
│   PettingZoo Parallel API + Ray RLlib EnvRunners (cores 8-11)                │
│                                  │                                           │
│            ┌─────────────────────▼─────────────────────┐                     │
│            │  Ray RLlib Learner (PPO, MultiRLModule)   │                     │
│            │  GPU: 4-6 GB during training              │                     │
│            └─────────────────────┬─────────────────────┘                     │
│                                  │ manual checkpoint review (TB + scenarios) │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │ promote frozen weights
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRODUCTION WORLD (persistent, 20 TPS, never resets)                         │
│                                                                              │
│   ┌──────────────┐                              ┌────────────────────────┐   │
│   │ Fabric srv P │  ← whitelist friends join   │ LLM Planner (loops)    │   │
│   │ Carpet (no   │                              │ - Claude Haiku API     │   │
│   │  tick sprint)│   4 living agents (Carpet   │   $40-80/mo runtime    │   │
│   │ Py4J:25100   │   fake players via Py4J):   │ - 3-tier degradation:  │   │
│   │ cores 12-13  │   gatherer/builder/         │     Haiku → local Qwen │   │
│   │              │   farmer/defender           │     14B → halt+alert   │   │
│   │              │   frozen RL weights         │ - event-driven only    │   │
│   │              │   growing skill lib         └────────────┬───────────┘   │
│   │              │   growing episodic mem                   │               │
│   └──────┬───────┘                                          │               │
│          │                                                  │               │
│          ▼                                                  ▼               │
│      ChatBridge ◄─── @<agent_name> player chat ─── Planner EventQueue       │
│                                                                              │
│   ┌────────────────────────┐    ┌─────────────────────┐                      │
│   │ Chroma vector DB       │    │ SQLite              │                      │
│   │ - skill_lib_{uuid}     │    │ - identity.db       │                      │
│   │ - mem_{uuid}           │    │ - planner_state.db  │                      │
│   │ - importance×recency   │    │ - chat_failures     │                      │
│   └────────────────────────┘    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Per-agent logical brain (three tiers)

```
┌─────────────────────────┐
│  Tier 1 — LLM Planner   │   event-driven (~5–20 calls/hr)
│  Claude Haiku           │   emits typed JSON DAG of subgoals per role
└────────────┬────────────┘
             │ LlmPlanOutput JSON (§5.1)
             ▼
┌─────────────────────────┐
│ Tier 2 — Goal Spec      │   deterministic Python adapter
│ Adapter (§2.1)          │   frozen BGE-small NL embedding + 128-d
│ NO learned routing      │   structured features → 512-d goal_embedding;
│                         │   hard role dispatch via string lookup
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Tier 3 — RL Policy     │   per-role (shared backbone, role head)
│  PPO at 20 Hz           │   parameterized skill primitives
│  hybrid obs (sym +      │   CTDE during training, decentralized at
│  optional pixel)        │   inference
│  Voyager-style skill    │
│  library augments       │
│  actions                │
└─────────────────────────┘
```

### 2.3 Seed strategy (training world)

- **M1–M3:** all-fixed seeds across 4 EnvRunners — `(forest/day/water)`, repeated four ways for variance.
- **M4+:** mix `3 fixed + 1 randomized`:
  - Seed A (easy): forest biome, day start, near water
  - Seed B (medium): mixed terrain, dusk start
  - Seed C (hard): rough terrain, night start with mobs spawning
  - Seed D (random): regenerated per-episode, uniform over biome generation space
- `per_worker_seed_offset: True` in `env_config` so the random rollout truly diverges across workers.

### 2.4 Resource budget summary

```
TRAINING (cores 0–7)         12 GB RAM, 50–150 TPS sustained per instance
ROLLOUT (cores 8–11)         Ray EnvRunners, ~1000–1300 agent-steps/sec effective
                              (after Py4J batching fix in §3.7)
LEARNER (RTX 4080)           4–6 GB VRAM (PPO multi-policy)
PRODUCTION (cores 12–14)     ~2–3 GB Fabric + ~2 GB planner Python
DASHBOARD + Chroma (core 15) ~2.5 GB total
QWEN 14B FALLBACK (RTX 4080) up to 10 GB VRAM when active (separate container)

Total ~38–46 GB RAM in use, ~18–26 GB headroom.
```

---

## 3. Agent Components &amp; Identity Layer

### 3.1 Tier 2 — Goal Spec Adapter (the load-bearing custom component)

The Tier 2 adapter turns LLM-emitted JSON subgoals into the conditioning signal the RL policy consumes. **Frozen pre-trained embedding + hard role dispatch — no learned routing.**

```python
# Flow:
LLM Planner emits Subgoal JSON (§5.2)
        ↓
GoalSpecAdapter (deterministic):
  1. Pydantic v2 validate
  2. Build NL summary string from subgoal_id + nl_summary
     e.g. "gatherer: collect 32 oak_log within 5 min, avoid combat,
           prefer nearest forest biome"
  3. Frozen BGE-small-en-v1.5 encode → 384-d (CPU, ~5 ms)
  4. Build 128-d structured feature vector:
     [one_hot_role(4) | inventory_delta_normalized(64) |
      timeout_normalized(1) | priority(1) | flags(58)]
  5. Concat → 512-d goal_embedding (no projection)
  6. Hard string dispatch: lookup role_policy in registry by
     subgoal.role ∈ {gatherer, builder, farmer, defender}
  7. Inject as `goal_embedding` into rl_policy_input
        ↓
role-specific PPO policy (shared backbone + role head)
```

**Why BGE-small (384-d), not BGE-base (768-d):** structured features carry the precision; NL embedding only needs flavor. Doubling dimension doubles input layer of every role-head and per-step inference cost. Upgrade only if M2 reward curves show goal-embedding bottleneck.

**Why frozen, not learned end-to-end:** avoids stacking two non-stationary training targets (embedder + policy). Embedder is interpretable (you can read the string sent in). Stable across LLM swaps (Haiku → Qwen 14B doesn't invalidate cached embeddings).

**Why hard role dispatch, not gating network:** the planner already decided the role. Gating networks introduce training instability under heterogeneous MARL. Hard dispatch is testable (bad role string raises immediately). Migration path to gating is open if stretch C economy makes cross-role transfer important.

### 3.2 Production agent embodiment — Carpet fake players

For v1, each of the 4 production agents is a Carpet fake player created via `/player <name> spawn` and `/player <name> loadProfile <name>` driven through Py4J. Position, inventory, look direction, health, hunger, attack, use, jump, sneak — all set via Py4J → Java method calls. Carpet + UnionClef-fork bridge provides this for free.

**Cosmetic / safety gap — required guard mod:** A Fabric mixin on `PlayerListS2CPacket` and `KickPlayerCommand` filters Carpet fake players for non-admin users. Friends who type `/list` should not see "Bjorn, Gunnar, Bram, Lisa, Carte" with no AI/human indicator (breaks the world-citizen experience). Accidental `/kick` of a fake player would trigger a permadeath event by misclick; mixin disallows non-admin kicks of fake players.

**Custom Fabric entity type** (`agentmod:villager_ai`) is the Phase 5+ alternative if `/list` pollution or chat-list semantics become limiting. Defer.

**Headless Java client per agent** is rejected — ~600 MB RAM each (~2.4 GB total), full networking stack, no benefit over Carpet fake players in a server-trusted setup.

### 3.3 ChatBridge — Q2 player↔agent NL chat path

ChatBridge is a separate component, not bundled into the planner. Chat does NOT interrupt the RL policy's current action; behavioral changes happen via new subgoal emission, not chat interrupt.

```
Player types in chat:   "@Bjorn where is the iron?"
                              │
                              ▼
┌──────────────────────────────────────────────────────┐
│ Fabric mixin on                                      │
│   ServerPlayNetworkHandler.onChatMessage(packet)     │
│ - Regex extract leading @<agent_name>                │
│ - Lookup agent_uuid via SQLite agents.agent_name     │
│ - If no match → fall through (vanilla chat)          │
│ - If match → emit ChatEvent (§5.4) to planner queue  │
│ - Suppress vanilla chat broadcast (default true)     │
└──────────────────────────────────────────────────────┘
                              │ Py4J callback
                              ▼
┌──────────────────────────────────────────────────────┐
│ LLM Planner — ChatEvent handler                      │
│ 1. Retrieve agent's episodic memory filtered by      │
│    participants CONTAINS player_uuid (§5.6)          │
│ 2. Retrieve current task / DAG state                 │
│ 3. Compose prompt with agent persona + memory +      │
│    current activity + incoming message               │
│ 4. Haiku call (5 s timeout, exp backoff)             │
│    - if &gt; 2 s: emit "Bjorn is thinking..."           │
│    - if &gt; 5 s or failure: deterministic fallback    │
│      "Bjorn doesn't seem to hear you" +             │
│      log to chat_failures table                      │
│    - 60-s same-player-same-text cache (in-memory)    │
│ 5. Write ChatEvent + Response → agent's episodic     │
│    memory (importance scored by length / keywords)   │
│ 6. Emit /tellraw response via Py4J                   │
│    server.getCommandManager().executeWithPrefix(     │
│      source, "/tellraw @a {...}")                    │
└──────────────────────────────────────────────────────┘
```

`expected_reply_type` selection heuristic in ChatBridge (v1):
- Contains `?` → `text` (question)
- First 3 words contain imperative verb (come, bring, stop, attack, defend, gather, build, move) → `action_ack`
- Otherwise → `none`
- Upgrade to LLM classifier deferred to Phase 5+ if heuristic miscategorizes &gt;10% of messages.

### 3.4 Planner state persistence (production restart survival)

```sql
CREATE TABLE planner_state (
  plan_id              TEXT PRIMARY KEY,          -- ULID
  agent_uuid           TEXT NOT NULL,
  status               TEXT NOT NULL,             -- active|completed|failed|paused|failed_migration
  dag_json             TEXT NOT NULL,             -- full DAG of subgoals
  current_subgoal_id   TEXT,
  pending_events_jsonl TEXT,                      -- queued events not yet processed
  llm_call_log_jsonl   TEXT,                      -- cost accounting + debugging
  schema_version       TEXT NOT NULL,             -- §5.5
  created_at           INTEGER NOT NULL,
  last_updated         INTEGER NOT NULL,
  FOREIGN KEY (agent_uuid) REFERENCES agents(agent_uuid)
);

CREATE INDEX idx_planner_active ON planner_state(status, agent_uuid)
  WHERE status IN ('active', 'paused');
```

- 30-s autosave + on-event-write durability tradeoff.
- **Startup protocol:** load all `status='active'` plans → run schema migration if `schema_version < CURRENT` → replay `pending_events_jsonl` → mark each plan `'paused'` until the agent's PPO policy reports healthy.
- **"Healthy" definition (operational):** PPO policy has consumed ≥1 observation AND emitted ≥1 action without exception within 30 s of restart. If timeout, log to `planner_failures` and remain paused for manual operator intervention.

### 3.5 Identity schema (Q5b death + succession compliant)

Five tables. `role_id` is decoupled from `agent_uuid`. Policy weights belong to the role; memory and skill library belong to the agent.

```sql
CREATE TABLE roles (
  role_id                    TEXT PRIMARY KEY,    -- 'gatherer'|'builder'|'farmer'|'defender'
  display_name               TEXT NOT NULL,
  policy_weights_path        TEXT NOT NULL,
  policy_version             INTEGER NOT NULL,
  observation_schema_version INTEGER NOT NULL,
  action_schema_version      INTEGER NOT NULL,
  max_lives                  INTEGER DEFAULT 1,   -- Q5b: 1 = permadeath
  default_skin_pool          TEXT                 -- JSON list of MC profile names
);

CREATE TABLE agents (
  agent_uuid          TEXT PRIMARY KEY,           -- new ULID per life
  role_id             TEXT NOT NULL REFERENCES roles(role_id),
  agent_name          TEXT NOT NULL,              -- UNIQUE among living
  skill_library_id    TEXT NOT NULL,              -- 'skill_lib_{agent_uuid}'
  memory_id           TEXT NOT NULL,              -- 'mem_{agent_uuid}'
  status              TEXT NOT NULL,              -- alive|dead
  born_at             INTEGER NOT NULL,
  died_at             INTEGER,
  spawn_position_json TEXT,
  current_skin        TEXT                        -- seeded by agent_uuid (NOT agent_name)
);
CREATE UNIQUE INDEX idx_agent_name_alive
  ON agents(agent_name) WHERE status = 'alive';

CREATE TABLE agent_lives (
  life_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
  role_id        TEXT NOT NULL REFERENCES roles(role_id),
  born_at        INTEGER NOT NULL,
  died_at        INTEGER,
  cause_of_death TEXT
);

CREATE TABLE funerals (
  funeral_id              INTEGER PRIMARY KEY AUTOINCREMENT,
  deceased_agent_uuid     TEXT NOT NULL REFERENCES agents(agent_uuid),
  witness_agent_uuids_json TEXT NOT NULL,        -- JSON array of UUIDs
  event_summary           TEXT NOT NULL,         -- LLM-generated 1-line obit
  written_to_memory_at    INTEGER NOT NULL
);

CREATE TABLE policy_deployments (              -- audit log of weight promotions
  deployment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id          TEXT NOT NULL REFERENCES roles(role_id),
  from_version     INTEGER,
  to_version       INTEGER NOT NULL,
  deployed_at      INTEGER NOT NULL,
  deployed_by      TEXT NOT NULL,
  notes            TEXT
);

CREATE TABLE chat_failures (                   -- §3.3 chat timeout/error log
  failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_uuid    TEXT NOT NULL REFERENCES agents(agent_uuid),
  player_uuid   TEXT NOT NULL,
  text          TEXT NOT NULL,
  error_type    TEXT NOT NULL,                  -- timeout|api_error|qwen_unavail
  occurred_at   INTEGER NOT NULL
);

CREATE TABLE planner_failures (                -- §3.4 startup-recovery log
  failure_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  plan_id       TEXT NOT NULL REFERENCES planner_state(plan_id),
  failure_type  TEXT NOT NULL,
  detail_json   TEXT,
  occurred_at   INTEGER NOT NULL
);
```

**Conventions:**
- All identifiers in this system use **ULID Crockford base32** from `python-ulid`. No UUID-v4, no UUID-v7. Single regex `^[0-9A-HJKMNP-TV-Z]{26}$`.
- `skill_library_id` ≡ `f"skill_lib_{agent_uuid}"`, `memory_id` ≡ `f"mem_{agent_uuid}"`. Auto-generated at agent creation, never reused after retirement.
- Skin selector seeded by `agent_uuid`, not `agent_name`. Two different lives both named "Bjorn" appear visually distinct.
- Skin pool: 12 names per role (configurable JSON list). Procedural fallback after exhaustion combines first-name pool + procedural surname (e.g., "Bjorn Ironwood"). Never `defender_42`-style names.

### 3.6 Death + succession sequence

```
T0: Creeper kills Bjorn (defender, agent_uuid=01J...)

T0:  Fabric onDeath event → Py4J → IdentityService
     - UPDATE agents SET status='dead', died_at=T0
     - UPDATE agent_lives SET died_at=T0, cause_of_death='creeper'
     - INSERT funerals (LLM generates 1-line obit)
     - Funeral event written to episodic_memory of all agents within
       proximity radius (~32 blocks)
     - Bjorn's Carpet body removed (`/player Bjorn kill`)
     - Defender slot status: 'vacant — succession pending'

T0..Tdawn:  Village runs without a defender. Survivors' next LLM
            replan sees "defender unavailable" in world state and
            adjusts behavior (builder/farmer stay closer to center).

Tdawn: IdentityService.spawn_successor(role_id='defender')
       - new agent_uuid (ULID)
       - role_id='defender' (unchanged; reuses role policy weights)
       - new agent_name (e.g. 'Gunnar')
       - new empty skill_library_id, new empty memory_id
       - INSERT agents, agent_lives
       - Carpet /player Gunnar spawn ... at village_center
       - Other agents' first interaction with Gunnar gets normal
         episodic memory write — "new defender arrived today"
```

---

## 4. Data Flow, Schemas &amp; Loops

### 4.1 Per-role observation Dict schemas

Disjoint key namespaces (`g_*`, `b_*`, `f_*`, `d_*`) make role schemas composable without collision. RLlib's Dict spaces tolerate the prefixes at zero cost.

**Universal core (every role):**

```python
core = Dict({
  "agent_uuid_embed":   Box(-1, 1,  (384,), f32),    # frozen BGE-small over uuid
  "role_one_hot":       MultiBinary(4),
  "tick_in_episode":    Box(0, 24000, (1,), i32),

  "position":           Box(-3e7, 3e7, (3,), f32),
  "velocity":           Box(-10, 10,  (3,), f32),
  "yaw_pitch":          Box(-180, 180, (2,), f32),
  "health":             Box(0, 20, (1,), f32),
  "hunger":             Box(0, 20, (1,), f32),
  "saturation":         Box(0, 20, (1,), f32),
  "armor_value":        Box(0, 20, (1,), f32),

  "inv_slot_item_ids":  MultiDiscrete([N_ITEMS]*36),
  "inv_slot_counts":    Box(0, 64, (36,), i32),
  "main_hand_item_id":  Discrete(N_ITEMS),
  "off_hand_item_id":   Discrete(N_ITEMS),

  "goal_embedding":     Box(-3, 3, (512,), f32),     # §3.1 — 384 BGE + 128 structured
  "goal_ticks_left":    Box(0, 12000, (1,), i32),    # 10 min cap (matches timeout_ticks)

  "time_of_day":        Box(0, 24000, (1,), i32),
  "weather":            Discrete(3),
  "biome_id":           Discrete(N_BIOMES),
  "light_level":        Box(0, 15, (1,), i32),

  # comms ring buffer (32 slots, 1.6 s history at 20 TPS)
  "comm_payloads":      Box(-1, 1, (32, 128), f32),
  "comm_metadata":      Box(0, 1, (32, 8), f32),     # [sender_role_4hot, age_norm,
                                                     #  priority, type_2hot]
  "action_mask":        Dict({...})                  # see §4.5
})
```

**Role overlays** (each role's obs Dict adds only its own overlay; no overlap):

```python
# gatherer_overlay (M1+)
"g_resource_grid":     Box(0, 1, (32, 32, 6), f32),   # top-down 32-radius, 6 channels
"g_nearest_resources": Box(-1, 1, (8, 6), f32),       # top-8 nearest
"g_richness_score":    Box(0, 1, (1,), f32),
"g_hostiles_nearby":   Box(0, 1, (4, 4), f32),

# builder_overlay (M2+)
"b_voxel_grid":        MultiDiscrete([N_BLOCKS]*(11*11*11)),
"b_blueprint_grid":    MultiDiscrete([N_BLOCKS]*(11*11*11)),
"b_blueprint_status":  MultiDiscrete([4]*(11*11*11)),  # 0=void,1=correct,2=missing,3=wrong
"b_materials_needed":  Box(0, 512, (N_BLOCK_TYPES,)),
"b_progress":          Box(0, 1, (1,), f32),
"b_pixel_patch":       Box(0, 255, (64, 64, 3), u8),  # task-area RGB (M2+)

# farmer_overlay (M3+)
"f_crop_plots":        Box(0, 1, (16, 8), f32),
"f_livestock":         Box(0, 1, (8, 6), f32),
"f_food_security":     Box(0, 1, (1,), f32),
"f_day_number":        Box(0, 10000, (1,), i32),
"f_seasonal_phase":    Discrete(4),

# defender_overlay (M4+)
"d_threats":           Box(-1, 1, (16, 9), f32),
"d_perimeter":         Box(0, 1, (8, 4), f32),
"d_friendlies":        Box(-1, 1, (4, 4), f32),
"d_threat_level":      Box(0, 1, (1,), f32),
"d_panoramic_patches": Box(0, 255, (4, 32, 32, 3), u8),  # M4+ conditional on plateau
```

### 4.2 Per-role action Dict schemas

Universal action header; `skill_type` enum is role-specific.

```python
action = Dict({
  "skill_type":      Discrete(N_SKILLS_FOR_ROLE),
  "target_class":    Discrete(N_TARGET_CLASSES),
  "spatial_param":   Box(-1, 1, (3,), f32),          # normalized offset
  "scalar_param":    Box(0, 1, (1,), f32),           # quantity / urgency
  "comm_payload":    Box(-1, 1, (128,), f32),        # learned message embedding
  "should_broadcast":Discrete(2),
  "comm_target_mask":MultiBinary(4),                  # excludes self;
                                                      # [0,0,0,0] + should_broadcast=1
                                                      # → env rewrites to should_broadcast=0
                                                      #   and logs 'comm_misroute'
})
```

**Role skill enums** (parameterized; multi-tick when motor module dispatches):

- Gatherer: `navigate, harvest, deposit_chest, search, wait, noop_broadcast`
- Builder: `navigate, place_block, fetch_materials, follow_blueprint, level_terrain, repair_damaged_block, wait, noop_broadcast`
- Farmer: `navigate, till_soil, plant, water, harvest_crop, breed_livestock, feed_livestock, butcher, wait, noop_broadcast`
- Defender: `patrol, intercept, defend_agent, build_defense, retreat_heal, wait, noop_broadcast`

### 4.3 Shared backbone architecture

```
INPUT (per agent, per timestep): obs_dict[role]
                                  (disjoint keys: core + role_overlay + action_mask)

┌──────────────────────────────────────────────────────────────────┐
│ CoreEncoder (parameter-shared across all 4 roles)                │
│   inputs:  agent_uuid_embed(384) + role_one_hot(4) + body(11)    │
│            + inventory(36+72) + goal(513) + world(4) + comms     │
│            (32×136 → flattened/mean-pooled)                      │
│   layers:  Linear(big→512) → ReLU → Linear(512→256) → ReLU       │
│   output:  core_feat ∈ R^256                                     │
└──────────────────────────────────────────────────────────────────┘
                          │
                          │  +  RoleSpecificEncoder (per role)
                          │     gatherer:  CNN(g_resource_grid) + MLP(nearest)
                          │     builder:   3D-CNN(voxel+blueprint) + MLP(materials)
                          │     farmer:    MLP(crop_plots + livestock + scalars)
                          │     defender:  MLP(threats + perimeter + friendlies)
                          │     → role_feat ∈ R^128
                          │
                          │  +  OptionalPixelEncoder
                          │     builder M2+:  CNN(64×64×3) → R^64
                          │     defender M4+: shared-weights CNN(32×32×3) ×4 dirs → R^64
                          │     others / pre-milestone: zeros(64)
                          │
                          ▼
            concat: [core_feat(256) | role_feat(128) | pixel_feat(64)] = 448
                          │
┌──────────────────────────────────────────────────────────────────┐
│ SharedBackbone (parameter-shared across all 4 roles)             │
│   layers: Linear(448→384) → ReLU                                 │
│           → LSTM(384, hidden=256, num_layers=1)                  │
│              # recurrent state PER agent_uuid, persisted across  │
│              # ticks within an episode                           │
│   output: hidden ∈ R^256                                         │
└──────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌────────────────────────┐       ┌──────────────────────────────┐
│ ActorHead[role]        │       │ CTDE CriticHead (shared)     │
│ (4 heads, separate)    │       │ TWO-STAGE encoder            │
│ input: hidden(256) +   │       │  per-agent shared encoder    │
│        goal_embed(512) │       │  compresses each agent's     │
│ output: action_dict    │       │  obs → 128-d, then           │
│ logits (per §4.2)      │       │  MLP(4·128 + village_inv_enc │
│ + apply action_mask    │       │      → 256) → V(s)           │
│   before softmax       │       │ ONLY used in training        │
└────────────────────────┘       └──────────────────────────────┘
```

**Backbone sharing rationale:** roles benefit from common world dynamics (light, gravity, mob behavior). If a role's gradients hurt others, peel onto its own backbone instance via `algorithm_config_overrides_per_module`.

**Implementation requirement** (§7.3 — RLModule code): shared components are declared as `additional_module_specs` on `MultiRLModuleSpec` so RLlib's weight-sync mechanism propagates updates from learner to all rollout workers. **Module-level Python globals are forbidden** — Ray workers fork processes; Python module globals don't share, gradients silently never propagate to rollouts.

### 4.4 Pixel patches

**Builder (M2+)** — 64×64 task-area RGB:
- Only rendered when `current_skill ∈ {place_block, follow_blueprint, repair_damaged_block}`.
- Camera positioned 5 blocks back from `target_pos`, yaw/pitch oriented toward target.
- Iris/Sodium-compatible offscreen framebuffer when available; **software-rendered fallback** (raycast + texture lookup, ~50 ms vs ~5 ms) when not. Mod-availability detected at startup; fallback selected automatically.
- 16 KB framebuffer per builder, reused.

**Defender (M4+, conditional)** — 4× 32×32 cardinal patches:
- Built only if symbolic defender plateaus on eval scenarios in M4.
- Camera at `agent_pos + Vec(0, 1.5, 0)`, 90° FOV per cardinal, `pitch=0`.
- Stacked `(4, 32, 32, 3)`; shared-weights CNN per patch, concat → 64-d.

**Gatherer / farmer** — no pixels in v1.

### 4.5 Action masking

Hard `-inf` logit mask, computed in env wrapper, applied pre-softmax. **No reward penalty for illegal actions** (one documented exception: §5.6 γ_clip for silent continuous clipping).

```python
"action_mask": Dict({
  "skill_type":         MultiBinary(N_SKILLS_FOR_ROLE),
  "target_per_skill":   MultiBinary((N_SKILLS_FOR_ROLE, N_TARGET_CLASSES)),
  "comm_payload":       MultiBinary(1),
  "should_broadcast":   MultiBinary(2),
})

# Example cascade:
if inv_slot_counts.sum() == 36*64:    skill_type[HARVEST] = 0
if health[0] < 4 and role == DEF:     skill_type[INTERCEPT] = 0  # forces retreat_heal
if not any(b_blueprint_status == 2):  skill_type[PLACE_BLOCK] = 0
if b_materials_needed.sum() > 0:      skill_type[PLACE_BLOCK] = 0   # must fetch first
if no_target_chest_in_range:          target_per_skill[DEPOSIT, *] = 0
if all_target_per_skill[s] == 0:      skill_type[s] = 0             # cascade

# Oscillation requires intermediate non-adjacent tile visit between revisits;
# stay-in-place is NOT oscillation (per §5 carry-forward).
```

**Fail-safe:** if mask filters all `skill_type` (impossible state), env wrapper emits `skill_type=WAIT` and logs `mask_collapse`. Should never fire in practice; alert if it does.

### 4.6 Per-tick RL loop

```
1. Java-side Fabric: read world state via Carpet+Lithium APIs
   → JSON serialized → Py4J → Python env wrapper
   (single BATCHED call per env: fabric.observationsAll() returns
    JSON blob containing all agents on that env; NOT per-agent
    fabric.observation(agent) — 4× Py4J roundtrips would cap throughput
    at ~300 agent-steps/sec)
2. Python env wrapper:
   - parse JSON into role-specific Dict (universal core + overlay)
   - compute action_mask sub-dict from symbolic obs (§4.5)
   - emit obs to PettingZoo Parallel API
3. RLlib EnvRunner:
   - batch obs across N envs (8 envs × 4 agents = 32 obs per batch)
   - send to learner GPU
4. Per-role policy forward (§4.3 backbone):
   - CoreEncoder + RoleEncoder + (optional) PixelEncoder → 448-d
   - SharedBackbone (LSTM with persisted hidden state) → 256-d
   - ActorHead[role] → action logits → apply action_mask → masked softmax
5. MID-TICK COMM FLUSH: comm bus flushes between policy forward (4) and
   action emission (5), so within-tick broadcasts reach receivers on the
   same tick. (NOT end-of-tick — would introduce 1-tick latency per hop.)
6. Action_dict → Py4J → Java motor module:
   - decode (skill_type, target_class, spatial, scalar)
   - execute via Baritone / Fabric API
   - skill may be multi-tick (5–30 s); ack-based completion via
     SkillCompletionEvent emitted by motor module
   - Idempotency: per-agent skill_invocation_id counter dedupes
     retried events
   - 30-s wait cap on advanceTickAwaitEvents; on timeout emit
     SkillTimeoutEvent, reset current_skill, log motor_hang
7. Env wrapper computes reward (§5 reward stack: PBRS + scarcity + village)
8. Episodic memory: if importance > threshold, batch buffer (§4.9)
9. RLlib trajectory accumulation → learner update every train_batch_size=4096
```

### 4.7 Event-driven LLM planner loop

**Triggers** (any of):
- Phase boundary (dawn / dusk / new in-game day)
- Subgoal completion (success or `FailureReport`)
- World event (mob attack, resource exhaustion, player join/leave, weather→thunder)
- ChatEvent (§3.3)
- Manual operator command

**EventQueue priority** (smaller integer = higher precedence, standard heap semantics):

| Priority | Event |
|---|---|
| 0 | `ChatEvent` (highest, interactive) |
| 1 | `FailureReport` |
| 2 | `WorldEvent` |
| 3 | `PhaseTransition` (lowest) |

**Per-trigger flow** (asyncio in planner process):

1. EventQueue dequeue
2. Compose prompt: system + retrieved episodic memory (§5.6, query templates per intent) + retrieved skill library + current world state + plan DAG + event payload
3. **Three-tier LLM degradation:**
   - Claude Haiku (5 s timeout, exp backoff on rate-limit) — primary
   - On budget cap → switch to local Qwen 14B in `planner-llm-local` container + alert
   - On Qwen unavailable → halt planner (paused on all active plans) + page operator. RL policies continue current subgoals; no new subgoals dispatched.
4. Parse JSON → Pydantic validate → write to `planner_state` (§3.4)
5. For each new subgoal: `GoalSpecAdapter` (§3.1) → `goal_embedding` → role policy via Py4J
6. Write `LLMCallEvent` to per-agent episodic memory (importance = mid)

### 4.8 Inter-agent communication channel

```python
CommMessage = {
  "sender_uuid":      str,
  "sender_role":      str,
  "timestamp":        int,
  "message_type":     "broadcast"|"directed"|"alert",
  "target_role_mask": MultiBinary(4),     # excludes self; [0,0,0,0]+broadcast=1 invalid
  "payload_128d":     ndarray(128, f32),  # learned by policy
  "spatial_origin":   (x, y, z),
  "spatial_range":    32                  # blocks (sec03 default)
}
```

**Routing (CommBus, Python-side):**
- Filter receivers by role mask + euclidean distance ≤ spatial_range
- Insert into 32-slot ring buffer (oldest evicted by recency)
- Receiver sees in next tick's `comm_payloads` + `comm_metadata`

**Why 128-d learned payload:** cheap (128 floats), differentiable for policy gradient.
**Risk mitigation:** sec07 communication-degeneration mitigation — sparse engineered messages first (broadcast / directed / alert with `comm_target_mask` does most coordination), learned 128-d payload gated to M5+ only if MAPPO eval shows tasks unsolvable without it.
**Player-readable summaries** generated post-hoc by LLM for chat overlay (debug view) — not in RL loop.

### 4.9 Episodic memory write path

```python
# Per agent, each tick:
importance = weighted_sum({
  abs(reward) / max_reward_scale: 0.30,
  novel_state_indicator:           0.15,  # KL vs running mean
  comm_received_count_norm:        0.10,
  player_within_proximity:         0.15,
  threat_level:                    0.15,
  planner_event_this_tick:         0.15,
})
# v1 weights are starting guesses. Exploitation hunt scheduled
# for end of M3 (§5 carry-forward): log retrieved top-K per
# planner prompt vs cited in DAG. Target ≥30% citation rate.
# Bias check: if 80%+ of HIGH writes come from same 3-4 event
# types, redistribute weight toward novel_state_indicator.

if importance ≥ 0.7:
    # HIGH: write immediately
    summary = LLM_template(...)        # or Haiku for ≥0.9
    embedding = BGE_encode(summary)
    record = {
      "agent_uuid": uuid, "timestamp": tick,
      "event_type": event_type, "participants": [...],
      "importance_score": importance,
      "embedding": embedding, "summary": summary,
    }
    chroma_client.collection(f"mem_{uuid}").add(record)
    sqlite.episodic_event_log.insert(record_meta)   # mirror for query/audit

elif importance ≥ 0.3:
    # MEDIUM: batch buffer
    batch_buffer[uuid].append(record_meta)

# else: ignored (sub-threshold majority)

# Flush every 200 ticks OR 50 buffered records (whichever first).

# Retention (background daily):
#   - DELETE records with importance < 0.4 AND age > 30 in-game days
#   - Compact low-importance records into LLM-generated "season summaries"
```

`participants` is stored in two forms — the structured list for JSON queries, and `participants_csv` for Chroma `$contains` filtering. CSV format: `",player-{uuid},agent-{uuid},..."` with leading comma + prefix to disambiguate player UUID space from agent UUID space. Substring match always includes leading comma to avoid prefix-collision false positives.

---

## 5. Reward Architecture &amp; Training Curriculum

### 5.1 Three stages

Reward function is stage-dispatched. One implementation (`compute_reward()` in §5.2) selects per-stage components.

**Stage 1 (M1–M2 first half, solo per-role pretraining):** per-role primary signal + PBRS shaping + universal penalties. No village component, no team term.

```python
r_stage_1 = (
    + r_primary_signal      # role-specific (§5.4 table)
    + r_pbrs                # γ=0.99, tech-tree potential
    - r_death_penalty       # 10.0 on death
    - r_time_penalty        # 0.001/tick
    - r_exploit_penalty     # §5.3
    - r_clip_penalty        # γ_clip=0.05 per clipped axis
)
```

**Stage 2 (M2 second half – M4, cooperative CTDE):** per-role multi-objective + team progress with curriculum decay.

```python
# Curriculum factor — decays role-specific signal influence, raises team term
curriculum_step  = train_iter * train_batch_size   # ENV STEPS, not ticks
curriculum_decay = max(0.7, 1.0 - 0.3 * curriculum_step / 10_000_000)
                   # 1.0 at step 0; 0.7 at step 10M (~3h wall-clock at 1000 steps/sec)

r_role_objectives = sum(w_k * r_k for k in role.objective_set)
# w_k from role.stage2_weights; normalized to sum=1.0; defaults in §5.4

r_stage_2 = (
      curriculum_decay     * r_role_objectives
    + (1 - curriculum_decay) * 0.5 * r_team_progress
    + r_pbrs - r_death_penalty - r_time_penalty
    - r_exploit_penalty - r_clip_penalty
)

# r_team_progress (during cooperative training):
#   M2: structure_built_correctly         +5.0 on completion
#   M3: + food_security ≥ 0.8 milestone   +3.0
#   M4: + no agent died in last day       +2.0 daily survival
```

**Curriculum denominator is 10M env steps** (not 100K). At 1000 agent-steps/sec the 100K value would decay curriculum to floor in ~80 s, neutralizing the schedule.

**Stage 3 (M5+, LLM-integrated, village-scarcity-aware):** village inventory targets drive per-role scarcity weights; +50% deposit bonus.

```python
# Village inventory targets emitted by LLM planner per subgoal
# (stored in plan.village_targets; null in stages 1-2 and late-M4 stub)

def scarcity(r):
    return max(0, 1.0 - village_inventory[r] / village_targets[r])

w_k(scarcity_vector) = 0.2 + 0.8 * scarcity[k_resource]
weights = normalize(w_k for k in role.objective_set)  # sum = 1.0

r_role_objectives_dynamic = sum(weights[k] * r_k for k in role.objective_set)
r_deposit_bonus = 0.5 * r_primary_signal if action == deposit_chest and success else 0

r_stage_3 = (
      0.7 * r_role_objectives_dynamic
    + 0.3 * r_team_progress
    + r_deposit_bonus
    + r_pbrs - r_death_penalty - r_time_penalty
    - r_exploit_penalty - r_clip_penalty
)

# Decentralized fallback when village_targets unavailable (LLM offline):
#   weights revert to fixed Stage-2 defaults (curriculum-decayed)
#   r_deposit_bonus stays active (local-observable)
```

**Stage-2 → Stage-3 transition:** hybrid pre-exposure + 5M env step ramp.
- Late M4: stub planner emits `village_targets=null` sentinel. Env wrapper sees field; `scarcity()` returns 0. Policy gets schema exposure without behavioral change.
- M5 start: real LLM activates. Interpolate `α: 0→1` over 5M env steps (~80 min wall-clock at full throughput). Effective reward: `α·r_stage_3 + (1-α)·r_stage_2`.

### 5.2 Unified `compute_reward()`

```python
def compute_reward(role, stage, obs_prev, obs_curr, action, env_meta) -> float:
    r_primary  = primary_signals[role](obs_prev, obs_curr, action)
    phi_prev   = tech_tree_potential(obs_prev.inventory, role)
    phi_curr   = tech_tree_potential(obs_curr.inventory, role)
    r_pbrs     = 0.99 * phi_curr - phi_prev

    r_death    = 10.0 if env_meta.died_this_tick else 0.0
    r_time     = 0.001
    r_exploits = sum(detect_exploits(role, obs_prev, obs_curr, action))   # §5.3
    r_clip     = 0.05 * env_meta.n_clipped_param_axes                     # §5.5

    if stage == 1:
        return (r_primary + r_pbrs - r_death - r_time
                - r_exploits - r_clip)

    if stage == 2:
        curriculum = max(0.7, 1.0 - 0.3 * env_meta.global_step / 10_000_000)
        weights    = role.stage2_weights
        r_role_obj = sum(weights[k] * role_objectives[k](obs_prev, obs_curr)
                         for k in role.objective_set)
        r_team     = team_progress_signal(env_meta.global_state)
        return (curriculum * r_role_obj
                + (1 - curriculum) * 0.5 * r_team
                + r_pbrs - r_death - r_time - r_exploits - r_clip)

    # stage 3 uses fixed 0.7/0.3 split, no curriculum decay
    weights    = scarcity_weights(env_meta.village_inventory,
                                  env_meta.village_targets, role)
    r_role_obj = sum(weights[k] * role_objectives[k](obs_prev, obs_curr)
                     for k in role.objective_set)
    r_team     = team_progress_signal(env_meta.global_state)
    r_deposit  = (0.5 * r_primary
                  if action.skill_type == DEPOSIT_CHEST
                     and env_meta.deposit_succeeded
                  else 0.0)
    return (0.7 * r_role_obj + 0.3 * r_team + r_deposit
            + r_pbrs - r_death - r_time - r_exploits - r_clip)
```

### 5.3 Exploit catalog (6 types, sec06-aligned, hardened)

```python
class ExploitDetector:
    """Per-agent stateful detector. Returns list of (name, penalty) tuples."""
    def __init__(self):
        self.recent_drops    = deque(maxlen=20)   # window=20 ticks
        self.position_buf    = deque(maxlen=40)
        self.inv_snapshots   = deque(maxlen=10)
        self.bulk_log        = deque(maxlen=200)
        self.last_action     = None
        self.idle_ticks      = 0

    def step(self, role, obs_prev, obs_curr, action, env_meta):
        penalties = []

        # 1. ITEM_DROP_SPAM (MAX_DROPS_PER_WINDOW = 2, tightened from 3)
        dropped = inv_drop_set(obs_prev.inv, obs_curr.inv)
        if dropped:
            self.recent_drops.extend(dropped)
        if len(self.recent_drops) > 2:
            penalties.append(("drop_spam", 0.5 * len(self.recent_drops)))

        # 2. OSCILLATION: revisits same 3×3 tile > 5 times in 40 ticks AND
        #    at least one non-adjacent tile visited between revisits
        #    (stay-in-place is not oscillation)
        tile = quantize_pos(obs_curr.position, grid=3)
        self.position_buf.append(tile)
        if (self.position_buf.count(tile) > 5
            and has_non_adjacent_intermediate(self.position_buf, tile)):
            penalties.append(("oscillation", 0.3))

        # 3. INVENTORY_REPEAT: same inventory snapshot > 2 in last 10
        snap = inv_hash(obs_curr.inv)
        if list(self.inv_snapshots).count(snap) >= 2:
            penalties.append(("inv_repeat", 0.1))
        self.inv_snapshots.append(snap)

        # 4. BULK_FARMING (multi-agent collusion) — implemented at env-wrapper
        #    level: if A drops X and B picks X within 5 ticks, increment
        #    collusion_counter[A,B,X]; on counter > 3 write penalty -1.0 split.

        # 5. SAFE_INACTION / LAZY: same WAIT/NOOP_BROADCAST > 80 of last 100
        if action.skill_type == self.last_action:
            self.idle_ticks += 1
        else:
            self.idle_ticks = 0
        if action.skill_type in {WAIT, NOOP_BROADCAST} and self.idle_ticks > 80:
            penalties.append(("lazy_inaction", 0.2))
        self.last_action = action.skill_type

        # 6. MEANINGLESS_TOOL_CALL: IMMEDIATE_FAILURE > 3 for same
        #    (skill, target) in 50 ticks
        key = (action.skill_type, action.target_class)
        self.bulk_log.append((env_meta.global_step, key,
                              env_meta.skill_result_code))
        recent_fails = sum(1 for ts, k, r in self.bulk_log
                            if k == key and r == "IMMEDIATE_FAILURE"
                            and env_meta.global_step - ts < 50)
        if recent_fails > 3:
            penalties.append(("noop_skill_spam", 0.2 * recent_fails))

        return penalties
```

### 5.4 Per-role primary signal table

| Role | `r_primary_signal` | `objective_set` (stage-2 weights, sum=1.0) |
|---|---|---|
| gatherer | `Σ_r delta_inv[r] * potential[r]` (VPT-normalized potentials); +50% on deposit (stage 3) | `wood: 0.30 stone: 0.35 food: 0.35` (curriculum-decayed) |
| builder | `Σ_cell delta(b_blueprint_status)` weights: `{void→correct:+1.0, missing→correct:+1.0, wrong→correct:+0.5, correct→wrong:-2.0}`; +5.0 on complete blueprint | `blueprint_match: 0.60 material_efficiency: 0.20 structure_integrity: 0.20` |
| farmer | `Σ_crop delta(growth_stage)*0.5 + harvested_food_value*1.0 + livestock_breed_event*0.5 − crop_died*1.0` | `crop_progress: 0.40 food_security: 0.40 livestock_health: 0.20` |
| defender | `damage_dealt_to_hostile*0.1 − damage_taken*0.1 + hostile_killed*1.0 + ally_protected_from_threat*0.5 − ally_died_in_proximity*5.0` (PBRS over `threat_level`, NOT raw kills) | `threat_reduction: 0.50 ally_protection: 0.30 perimeter_integrity: 0.20` |

### 5.5 γ_clip — silent continuous-param clipping penalty

```python
# Motor module returns env_meta.clipped_param_log[axis] = bool for each
# continuous param axis it had to clamp.
#   spatial_param: 3 axes (dx, dy, dz)
#   scalar_param:  1 axis
# Total 4 axes.

r_clip = 0.05 * sum(env_meta.clipped_param_log[axis] for axis in 4_axes)
```

This is the documented exception to "no reward penalty for illegal actions" because clipping is silent (policy can't observe it via obs Dict). Committed for v1 (no "expose in obs" alternative — would create a gaming surface).

### 5.6 Memory retrieval query shapes (read-side; write-side in §4.9)

```python
class MemoryRetriever:
    def __init__(self, chroma, bge):
        self.chroma = chroma
        self.bge    = bge

    def retrieve(self, agent_uuid, query_text, top_k=10,
                 min_importance=0.3, recency_lambda=0.04,
                 participant_filter=None, event_type_filter=None,
                 now_tick=None):
        """
        Hybrid: ANN with 5× overfetch, then rerank by
          score = (cos_sim * importance) * exp(-λ * age_days)
        """
        collection = self.chroma.get_collection(f"mem_{agent_uuid}")
        query_emb  = self.bge.encode(query_text).tolist()
        where      = {"importance_score": {"$gte": min_importance}}
        if event_type_filter:
            where["event_type"] = {"$in": event_type_filter}
        if participant_filter:
            # CSV format: ",player-{uuid},agent-{uuid},..."
            where["participants_csv"] = {"$contains": f",{participant_filter},"}

        hits = collection.query(query_embeddings=[query_emb],
                                 n_results=top_k * 5,
                                 where=where,
                                 include=["metadatas","documents","distances"])

        scored = []
        for doc, meta, dist in zip(hits["documents"][0],
                                    hits["metadatas"][0],
                                    hits["distances"][0]):
            cos_sim  = 1.0 - dist
            imp      = meta["importance_score"]
            age_days = max(0, (now_tick - meta["timestamp"]) / 24_000)
            scored.append({
              "score": (cos_sim * imp) * exp(-recency_lambda * age_days),
              "doc":   doc, "meta": meta,
            })
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]
```

**Static query templates for v1** (LLM-composed queries deferred to Phase 5+):
- General context: `f"general context for planning: goal={goal}, role={role}"`
- Player history: `f"interactions with player {player_name}"`
- Combat: `f"combat danger threat involving {threat_types}"`
- Funeral context: `f"death funeral memorial of {predecessor_agent_name}"`

**Tiered default `recency_lambda` per query intent:**
- 0.02 for long-term identity memories (funerals, first-meetings, succession) — decay at ~50 days
- 0.04 for general context (default)
- 0.05 for time-sensitive events (combat, weather, player interactions) — decay at ~20 days

### 5.7 `tech_tree_potential()` and `threat_level()` (referenced from §5.1)

```python
LOG_VALUE = {
    "oak_log": 1.0, "oak_planks": 0.05, "stick": 0.0625,
    "cobblestone": 1/11, "stone": 0.5, "coal": 0.4,
    "iron_ore": 4.0, "iron_ingot": 5.0,
    "gold_ingot": 3.0, "diamond": 8.0,
    "wheat": 0.1875, "bread": 0.375,
    "porkchop": 0.375, "cooked_porkchop": 0.5,
    "beef": 0.375, "cooked_beef": 0.5,
    "chicken": 0.375, "cooked_chicken": 0.5,
    "carrot": 0.1875, "apple": 0.25,
    "crafting_table": 1.0, "furnace": 1.0,
    "wooden_pickaxe": 1.0, "stone_pickaxe": 1.5, "iron_pickaxe": 4.0,
    "wooden_sword": 0.5, "stone_sword": 1.0, "iron_sword": 4.0,
    "wooden_axe": 0.5, "stone_axe": 1.0, "iron_axe": 4.0,
    "wooden_hoe": 0.5, "stone_hoe": 1.0, "iron_hoe": 4.0,
    "leather_helmet": 0.5, "iron_helmet": 2.0,
    "leather_chestplate": 1.0, "iron_chestplate": 4.0,
    "leather_leggings": 0.875, "iron_leggings": 3.5,
    "leather_boots": 0.5, "iron_boots": 2.0,
    "torch": 0.125, "oak_door": 0.5, "glass_pane": 0.5,
    "ladder": 0.25, "fence": 0.125, "chest": 1.0,
}

ROLE_INVENTORY_CAPS = {
    "gatherer":  {"oak_log":256, "cobblestone":256, "stone":128,
                  "coal":128, "iron_ore":128, "iron_ingot":64,
                  "diamond":16, "wheat":64, "bread":32,
                  "stick":64, "oak_planks":128,
                  "wooden_pickaxe":4, "stone_pickaxe":4, "iron_pickaxe":2,
                  "_default":64},
    "builder":   {"oak_log":128, "oak_planks":512, "cobblestone":512,
                  "stone":256, "torch":128, "oak_door":16,
                  "glass_pane":64, "ladder":32, "fence":64,
                  "chest":8, "iron_ingot":16, "_default":32},
    "farmer":    {"wheat":256, "bread":128, "carrot":64,
                  "porkchop":32, "beef":32, "chicken":32,
                  "cooked_porkchop":32, "cooked_beef":32, "cooked_chicken":32,
                  "_default":16},
    "defender":  {"iron_sword":4, "iron_pickaxe":2,
                  "iron_helmet":2, "iron_chestplate":2,
                  "iron_leggings":2, "iron_boots":2,
                  "bread":16, "cooked_beef":16, "_default":8},
}

def tech_tree_potential(inventory: dict[str, int], role: str) -> float:
    caps = ROLE_INVENTORY_CAPS[role]
    default_cap = caps.get("_default", 32)
    total = 0.0
    for item, qty in inventory.items():
        if item not in LOG_VALUE:
            continue
        cap = caps.get(item, default_cap)
        total += min(qty, cap) * LOG_VALUE[item]
    return total

def threat_level(d_threats, agent_pos, village_center) -> float:
    import numpy as np
    active = [t for t in d_threats if not _is_zero_threat(t)]
    if not active:
        return 0.0
    priorities       = np.array([t.priority for t in active])
    times_to_village = np.array([max(1.0, t.time_to_village) for t in active])
    weighted         = priorities * (1.0 / times_to_village)
    N_THREATS_MAX, MAX_PRIORITY = 16, 1.0
    return min(1.0, weighted.sum() / (N_THREATS_MAX * MAX_PRIORITY))
```

### 5.8 Three-phase pipeline mapped to milestones

```
PHASE 1 — Solo per-role pretraining (Stage-1 reward, IPPO single-agent)
  M1: gatherer solo, fixed seed 1 (forest/day/water).
      Eval gate: 80% success on "collect 64 oak logs" in 1000 env steps,
                 3 consecutive evals.
  M2 (first half): builder solo. Stub LLM "planner" emits hand-coded
      blueprints (5 standard structures: wall, gate, watchtower, well,
      storehouse). Eval gate: 70% blueprint_match on standard set, no
      place/break cycling.

PHASE 2 — Cooperative CTDE training (Stage-2 reward, IPPO → MAPPO,
                                       curriculum 2→3→4 agents)
  M2 (second half): gatherer + builder, 2-agent CTDE.
      Stub planner emits paired subgoals. Eval gate: cooperative
      structure_built_correctly > 50% in 2000-step episodes.
      Failure-mode watch: lazy gatherer; collision deadlock at chest.
  M3: + farmer (3 agents). Eval gate: food_security > 0.8 over 5 days
      while structure work proceeds. Per-role PBRS validation.
  M4: + defender (4 agents, full roster). Switch to mixed_3fixed_1random
      seed strategy. Eval gate: 70% mob kill rate inside perimeter,
      zero villager deaths to hostile mobs in 7-day eval.
      Defender pixel-patch decision: train symbolic-only first;
      gate adding panoramic patches on plateau evidence.

PHASE 3 — LLM planner integration + identity (Stage-3 reward; MAPPO
          frozen in early M5, optionally fine-tuned in late M5)
  M5: replace stub planner with real Claude Haiku planner emitting
      LlmPlanOutput JSON DAG. GoalSpecAdapter routes via §3.1.
      Identity layer (level D — social) folded in alongside.
      Eval gate: end-to-end "build a village" completed without
      manual subgoal injection over 14 in-game days.

PHASE 4 — Production deployment (M6)
  M6: 6 parallel training instances (CPU upgrade enables vs research's 4).
      Manual TensorBoard + held-out scenario review for promotion.
      Weight promotion via policy_deployments. Run for 30 in-game days
      with friends invited (Q2 B+C). Monitor for distribution shift,
      plan-cache health, episodic memory growth.

POST-M6 (stretch C): agent-agent economy.
```

### 5.9 MARL failure mitigation wiring

| Failure | Mitigation | Wired at milestone |
|---|---|---|
| Credit assignment | COMA counterfactual baseline; CTDE critic marginalizes over each agent's action while holding others fixed | M2 (first cooperative scenario). Trigger: per-agent reward variance ratio > 5× between active/idle agents over 1K episodes |
| Non-stationarity | CTDE + CADP extension. Multi-timescale LR: gatherer 1.0×, builder 0.7×, farmer 0.5×, defender 0.5× | M2 baseline; multi-timescale LR active from M3. MAPPO epoch ceiling: 5–10. |
| Lazy / free-rider | LAIES intrinsic (IDI + CDI). Initial weight 0.1× extrinsic | Earlier trigger: per-role contribution to team reward < 20% of fair-share (1/N_roles) over 500 episodes (NOT completion-rate at 2K — too late) |
| Equilibrium / role collapse | ROMA role-learning + R3DM mutual info bonus; curriculum 2→3→4 agents | M3 (where similarity first becomes risky). Detection: trajectory cosine similarity > 0.8 between roles |
| Reward hacking | Architectural: PBRS, delta-inventory, KL-reg to VPT prior, ExploitDetector | All stages. ExploitDetector active from M1. KL-reg activates in MAPPO phase (M3+) |
| Agent collision | Spatial role assignment: gatherer perimeter, builder center, farmer field, defender boundary | M2 onward. Hybrid MAPF only if collision deadlock > 3× per episode in M4 eval |
| Communication degeneration | Engineered protocols first (broadcast / directed / alert via `comm_target_mask`); learned 128-d payload gated | M2–M4 engineered only. Learned payload activated in M5 IF MAPPO eval shows tasks unsolvable without it |

Monitoring instrumented from M1: per-role action entropy, per-role Q-value variance, trajectory cosine similarity, reward-vs-eval gap. Intervention pause when 2+ thresholds breach simultaneously.

### 5.10 Promotion criteria (training → production)

```
Manual promotion checklist (per Q5A, v1):
  1. TensorBoard curves: episodic return trending up, no collapse in
     last 50K env steps, variance bounded.
  2. Held-out scenario battery (3 fixed seeds + 1 randomized):
     - per-role eval scenarios pass at threshold (M-specific)
     - cooperative scenario passes at threshold
  3. Exploit detector: total exploit penalty per episode < 5% of
     positive reward magnitude.
  4. Failure-mode dashboard: all sec07 high-priority signals green
     (per-role entropy > 1.5 bits, Q-value variance ratio < 5×,
     trajectory cosine < 0.8 between roles).
  5. Determinism check: action_argmax_divergence < 0.05 AND
     continuous_param_L2 < 0.1 over 1000-tick replay window.

On pass:
  aiutopia promote-weights --role <role> \
    --from-checkpoint <ray-run-id>/checkpoint_N --bump-version \
    --notes "<reason>"
  - Updates roles.policy_weights_path, bumps roles.policy_version
  - Writes policy_deployments audit row
  - Production planner hot-reloads weights at next subgoal boundary
    (per-agent LSTM hidden state PRESERVED across reload — agent has
     continuity of mind even though weights changed; reload happens
     at subgoal boundary not mid-skill)

Rollback:
  aiutopia promote-weights --role <role> --rollback
  - Restores prior version from policy_deployments history
  - Reverse hot-reload at next subgoal boundary, LSTM state preserved
```

---

## 6. LLM Planner Schemas, Memory &amp; DAG Lifecycle

### 6.1 Schema versioning

All four planner schemas (`LlmPlanOutput`, `Subgoal`, `FailureReport`, `ChatEvent`) share a unified `SCHEMA_VERSION_LLM_PLAN = "1.0.0"` and evolve as a single artifact. Any breaking change to any of them bumps the version.

SemVer mapping:
- **PATCH** — bug fix in validator, no field changes; auto-loaded.
- **MINOR** — additive (new optional fields, new enum literals); auto-loaded, missing defaults to None/empty.
- **MAJOR** — breaking (rename, drop, type change, new required); migration required.

Migrations live at `/var/lib/aiutopia/schema_migrations/llm_plan/X.Y.Z_to_X'.Y'.Z'.py`, exporting `def migrate(data: dict) -> dict`. Loader walks migrations forward, then re-validates. On migration failure, plan marked `status='failed_migration'`, original `dag_json` preserved for offline repair, alert raised. Migration runs **before** paused-recovery (§3.4).

### 6.2 `LlmPlanOutput` (Pydantic v2)

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator
import ulid

SCHEMA_VERSION_LLM_PLAN = "1.0.0"

RoleId = Literal["gatherer", "builder", "farmer", "defender"]
FailureType = Literal["timeout", "health_critical", "tool_broken",
                      "inventory_full", "path_blocked", "resource_unavailable",
                      "attacked", "unknown"]
PlannerSource = Literal["claude-haiku", "local-qwen-14b",
                        "stub-planner", "manual-cli"]

class Dependency(BaseModel):
    before: str
    after:  str

class LlmPlanOutput(BaseModel):
    plan_id:                     str = Field(default_factory=lambda: str(ulid.ULID()))
    schema_version:              str = SCHEMA_VERSION_LLM_PLAN
    high_level_goal:             str = Field(..., min_length=1, max_length=400)
    high_level_goal_template_id: Optional[str] = Field(
        None,
        description="references /var/lib/aiutopia/goal_templates/{id}.yaml; "
                    "None or 'freeform' = free-form goal (rarely cache-hits)"
    )
    village_targets:             Optional[dict[str, int]] = Field(
        default=None,
        description="Stage-3 inventory targets; null in stages 1-2 and "
                    "late-M4 stub pre-exposure."
    )
    subgoals:                    list["Subgoal"] = Field(..., min_length=1, max_length=32)
    dependencies:                list[Dependency] = Field(default_factory=list)
    max_fallback_chain_depth:    int = Field(default=3, ge=1, le=5)
    created_at:                  int = Field(..., description="unix epoch seconds")
    created_by:                  PlannerSource
    notes:                       Optional[str] = None

    @model_validator(mode="after")
    def _validate_dag(self) -> "LlmPlanOutput":
        ids = {s.subgoal_id for s in self.subgoals}
        for dep in self.dependencies:
            if dep.before not in ids:
                raise ValueError(f"dep.before {dep.before!r} not in subgoals")
            if dep.after not in ids:
                raise ValueError(f"dep.after {dep.after!r} not in subgoals")
            if dep.before == dep.after:
                raise ValueError("self-dependency forbidden")
        for sg in self.subgoals:
            for fb in sg.fallback_subgoals:
                if fb not in ids:
                    raise ValueError(f"fallback {fb!r} not in subgoals")
        # Kahn's topo sort for cycle detection
        in_deg = {sid: 0 for sid in ids}
        adj    = {sid: [] for sid in ids}
        for dep in self.dependencies:
            in_deg[dep.after] += 1
            adj[dep.before].append(dep.after)
        roots = [n for n, d in in_deg.items() if d == 0]
        seen  = 0
        while roots:
            n = roots.pop()
            seen += 1
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    roots.append(m)
        if seen != len(ids):
            raise ValueError("DAG cycle detected in dependencies")
        return self
```

### 6.3 `Subgoal` / `GoalSpecification` / `TerminationConditions` / `Constraints`

```python
class TargetState(BaseModel):
    inventory_delta:    dict[str, int] = Field(default_factory=dict)
    spatial_target:     Optional[tuple[float, float, float]] = None
    blueprint_target:   Optional[str] = None
    threat_neutralized: Optional[bool] = None    # defender-only

    @model_validator(mode="after")
    def _at_least_one(self):
        # Distinguish "not set" (None / empty) from "set to False" — a
        # defender goal `threat_neutralized=False` ("ensure threat is NOT
        # in the neutralized state yet") is valid. Truthiness alone treats
        # False as falsy and silently rejects it.
        if (not self.inventory_delta
            and self.spatial_target is None
            and self.blueprint_target is None
            and self.threat_neutralized is None):
            raise ValueError("TargetState requires at least one target field")
        return self

class TerminationConditions(BaseModel):
    success_criteria: list[str]      = Field(..., min_length=1)
    timeout_ticks:    int            = Field(..., gt=0, le=12000)
    failure_events:   list[FailureType] = Field(default_factory=list)

class Constraints(BaseModel):
    preserve_items:    list[str]     = Field(default_factory=list)
    avoid_biomes:      list[str]     = Field(default_factory=list)
    max_health_cost:   Optional[int] = Field(None, ge=0, le=20)
    tool_requirements: list[str]     = Field(default_factory=list)
    no_combat:         bool          = False

class GoalSpecification(BaseModel):
    target_state:           TargetState
    termination_conditions: TerminationConditions

class Subgoal(BaseModel):
    subgoal_id:         str  = Field(default_factory=lambda: str(ulid.ULID()))
    role:               RoleId
    priority:           int  = Field(default=5, ge=0, le=10)
    goal_specification: GoalSpecification
    constraints:        Constraints = Field(default_factory=Constraints)
    fallback_subgoals:  list[str]   = Field(default_factory=list,
        description="subgoal_ids tried in order if this subgoal fails")
    nl_summary:         str  = Field(..., min_length=1, max_length=1500,
        description="natural-language description; BGE-encoded by GoalSpecAdapter")
```

**`fallback_subgoals` semantics:** fallbacks are alternates **within the current plan** (already-defined subgoals tried in order on primary failure). If `fallback_chain_depth` exhausts without success, `FailureReport` triggers full LLM replan (new plan, not new fallback). Distinction: fallbacks = "plan considered alternatives"; replan = "plan didn't account for this failure mode."

### 6.4 `FailureReport`

```python
class ExecutionTraceEntry(BaseModel):
    tick:                int
    action_summary:      str  = Field(..., max_length=200)
    observation_summary: str  = Field(..., max_length=400)
    reward:              float

class PartialProgress(BaseModel):
    inventory_delta_achieved: dict[str, int] = Field(default_factory=dict)
    success_criteria_met:     list[str]      = Field(default_factory=list)
    progress_fraction:        float          = Field(..., ge=0.0, le=1.0)
    blueprint_status_summary: Optional[dict] = None
    crops_progressed:         Optional[int]  = None
    threats_neutralized:      Optional[int]  = None

class FailureDetails(BaseModel):
    failure_type:        FailureType
    failure_tick:        int
    final_state_summary: dict
    descriptor_summary:  str = Field(..., max_length=400)
    execution_trace:     list[ExecutionTraceEntry] = Field(
        default_factory=list, max_length=200
    )

class FailureReport(BaseModel):
    report_id:        str               = Field(default_factory=lambda: str(ulid.ULID()))
    schema_version:   str               = SCHEMA_VERSION_LLM_PLAN
    plan_id:          str
    subgoal_id:       str
    role:             RoleId
    agent_uuid:       str
    status:           Literal["failed"] = "failed"
    failure_details:  FailureDetails
    partial_progress: PartialProgress
    reported_at:      int               = Field(...)
```

**Execution trace downsampling** (`max_length=200`; full trace at 20 TPS over 12000-tick subgoal = 12000 entries; ~3% sample):

```python
def downsample_trace(full_trace, max_entries=200):
    if len(full_trace) <= max_entries:
        return full_trace
    first_50, last_100 = full_trace[:50], full_trace[-100:]
    middle             = full_trace[50:-100]
    change_ticks       = [e for e in middle if e.is_action_change]
    remaining          = 50 - len(change_ticks)
    non_change         = [e for e in middle if not e.is_action_change]
    evenly             = non_change[::max(1, len(non_change) // remaining)][:remaining]
    return first_50 + sorted(change_ticks + evenly, key=lambda e: e.tick) + last_100
```

### 6.5 `ChatEvent`

```python
class ChatEvent(BaseModel):
    event_id:             str  = Field(default_factory=lambda: str(ulid.ULID()))
    schema_version:       str  = SCHEMA_VERSION_LLM_PLAN
    sender_player_uuid:   str  = Field(..., description="Mojang UUID of player")
    sender_player_name:   str  = Field(..., min_length=1, max_length=16)
    addressed_agent_uuid: str
    addressed_agent_name: str  = Field(..., min_length=1, max_length=16)
    text:                 str  = Field(..., min_length=1, max_length=1000,
        description="raw chat text WITHOUT the leading @<agent_name> prefix")
    timestamp:            int  = Field(...)
    expected_reply_type:  Literal["text", "action_ack", "none"] = "text"
    suppressed_in_chat:   bool = True
```

Chat reply emitted via `/tellraw` is a one-shot string + optional hover JSON; not modeled as a planner artifact.

### 6.6 Subgoal DAG state machine

```
pending  ──dispatch──►  active    (scheduler clears all Dependency.before)
active   ──success───►  completed (motor: all success_criteria met)
active   ──failure───►  failed    (failure_event OR timeout)
active   ──pause─────►  paused    (production restart ONLY; never normal op)
paused   ──resume────►  active    (PPO healthy per §3.4 AND deps still satisfied)
failed   ──fallback──►  pending   (new instance from fallback_subgoals[0];
                                    original retained for audit;
                                    cap fallback_chain_depth)

completed │ terminal
failed    │ terminal-after-fallback-exhaustion
```

On-state-change hooks (Python observer pattern):

```
on_pending_to_active:
  - GoalSpecAdapter → goal_embedding (§3.1)
  - Py4J motor.start_subgoal(subgoal, embedding)
  - UPDATE planner_state (current_subgoal_id, last_updated)
  - Episodic memory write ('subgoal_started', importance=0.4)

on_active_to_completed:
  - Release credit assignment to RLlib
  - Episodic memory write ('subgoal_completed', importance=0.6)
  - Notify scheduler to dispatch downstream pending subgoals
  - On all subgoals completed → mark plan completed, archive

on_active_to_failed:
  - Generate FailureReport (§6.4)
  - EventQueue.put(FailureReport, priority=1)
  - Episodic memory write ('subgoal_failed', importance=0.7)

on_active_to_paused:
  - Serialize current_subgoal state to planner_state
  - Halt motor module skill execution (clean abort)
  - No memory write (operational, not narrative)

on_failed_to_pending (fallback):
  - Instantiate fallback subgoal as new pending record
  - Inherit parent dependencies
  - Increment fallback_chain_depth; refuse if > max_fallback_chain_depth
```

### 6.7 Plan cache schema

`plan_cache` lives in `/var/lib/aiutopia/planner_state.db` alongside the `planner_state` table from §3.4.

```sql
CREATE TABLE plan_cache (
  cache_key            TEXT PRIMARY KEY,        -- SHA256 over canonical bucketed context
  context_json         TEXT NOT NULL,
  prompt_text          TEXT NOT NULL,
  plan_json            TEXT NOT NULL,
  llm_model            TEXT NOT NULL,
  llm_call_latency_ms  INTEGER NOT NULL,
  llm_call_cost_usd    REAL,
  hit_count            INTEGER NOT NULL DEFAULT 1,
  created_at           INTEGER NOT NULL,
  last_hit_at          INTEGER NOT NULL,
  ttl_seconds          INTEGER NOT NULL DEFAULT 3600,
  invalidated          INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_plan_cache_last_hit ON plan_cache(last_hit_at)
  WHERE invalidated = 0;
```

Cache key construction uses **bucketed** canonical context (exact-context strings never hit — free-text goals and float-precision inventories never repeat). Buckets capture "situation type" — morning, gatherer healthy, builder mid-task, village low on wood.

**Invalidation triggers** (subscribed to §4.7 EventQueue):
1. Mob attack start
2. Whitelist join / leave
3. Weather → thunder transition
4. Agent death or successor spawn
5. Day-phase transition
6. Schema MAJOR bump (bulk `invalidated=1 WHERE schema_version < CURRENT`)
7. Manual `aiutopia cache-invalidate`

**Size cap:** 5000 entries (~25 MB JSON). LRU eviction by `last_hit_at`.

---

## 7. RLlib Configuration, Training Infrastructure &amp; Operational Tooling

### 7.1 `PPOConfig` — single-agent (M1) and multi-agent (M2+)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env

register_env("aiutopia_minecraft", lambda cfg: AiUtopiaPettingZooEnv(cfg))

# ─── M1 single-agent gatherer ───
m1_config = (
    PPOConfig()
    .framework("torch")
    .environment(
        env="aiutopia_minecraft",
        env_config={
            "stage":          1,
            "active_roles":   ["gatherer"],
            "seed_strategy":  "fixed_easy",
            "py4j_ports":     [25001, 25002, 25003, 25004],
            "tick_warp":      True,
            "max_episode_ticks": 6_000,
            "per_worker_seed_offset": True,
        },
    )
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=2,
        rollout_fragment_length="auto",
        sample_timeout_s=120.0,
        placement_strategy="STRICT_PACK",
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=0.5,
    )
    .training(
        train_batch_size=4096,
        minibatch_size=512,
        num_epochs=5,                      # cap 5-10 (sec07 MAPPO ceiling)
        gamma=0.99,
        lr=3e-4,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        kl_coeff=0.2,
        grad_clip=1.0,
        model={"use_lstm": True, "lstm_cell_size": 256, "max_seq_len": 32},
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "gatherer_policy": RLModuleSpec(
                    module_class=AiUtopiaRoleRLModule,
                    observation_space=role_obs_space("gatherer"),
                    action_space=role_action_space("gatherer"),
                    model_config={
                        "role":          "gatherer",
                        "core_hidden":   [512, 256],
                        "role_hidden":   [128],
                        "pixel_hidden":  None,
                        "lstm_size":     256,
                        "actor_hidden":  [256],
                        "critic_hidden": [256],
                        "use_action_masking": True,
                    },
                ),
            },
            # SHARED COMPONENTS as additional_module_specs — see §7.2
            additional_module_specs={
                "core_encoder":    RLModuleSpec(module_class=CoreEncoderModule, ...),
                "shared_backbone": RLModuleSpec(module_class=SharedBackboneModule, ...),
                "ctde_critic":     RLModuleSpec(module_class=CTDECriticModule, ...),
            },
        )
    )
    .multi_agent(
        policies={"gatherer_policy"},
        policy_mapping_fn=lambda agent_id, *_: "gatherer_policy",
    )
    .resources(num_cpus_for_main_process=2)
    .reporting(
        metrics_num_episodes_for_smoothing=200,
        keep_per_episode_custom_metrics=False,    # PRODUCTION-safe; True only for short evals
    )
    .checkpointing(
        export_native_model_files=True,
        checkpoint_trainable_policies_only=True,
    )
    .debugging(seed=1)
)

# ─── M4+ full 4-role MAPPO (deltas) ───
m4_config = (
    m1_config
    .environment(env_config={
        "stage":         2,
        "active_roles":  ["gatherer", "builder", "farmer", "defender"],
        "seed_strategy": "mixed_3fixed_1random",
        "py4j_ports":    [25001, 25002, 25003, 25004],
        "max_episode_ticks": 24_000,
        "per_worker_seed_offset": True,
    })
    .rl_module(rl_module_spec=MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": _role_module_spec("gatherer", lr_mult=1.0),
            "builder_policy":  _role_module_spec("builder",  lr_mult=0.7),
            "farmer_policy":   _role_module_spec("farmer",   lr_mult=0.5),
            "defender_policy": _role_module_spec("defender", lr_mult=0.5),
        },
        additional_module_specs={
            "core_encoder":    RLModuleSpec(module_class=CoreEncoderModule, ...),
            "shared_backbone": RLModuleSpec(module_class=SharedBackboneModule, ...),
            "ctde_critic":     RLModuleSpec(module_class=CTDECriticModule, ...),
        },
    ))
    .multi_agent(
        policies={"gatherer_policy", "builder_policy",
                  "farmer_policy", "defender_policy"},
        policy_mapping_fn=lambda agent_id, *_: f"{agent_id.split('_')[0]}_policy",
    )
    .training(
        learner_config_dict={
            "algorithm_config_overrides_per_module": {
                "gatherer_policy": {"lr": 3.0e-4},
                "builder_policy":  {"lr": 2.1e-4},
                "farmer_policy":   {"lr": 1.5e-4},
                "defender_policy": {"lr": 1.5e-4},
            }
        }
    )
)
```

### 7.2 `AiUtopiaRoleRLModule` and shared submodules

Shared components (`CoreEncoder`, `SharedBackbone`, `CTDECritic`) are **declared as RLModules in `additional_module_specs`** on `MultiRLModuleSpec` so RLlib's weight-sync mechanism handles propagation from learner to all rollout workers. **Module-level Python globals (`_CORE_ENCODER`, etc.) are forbidden** — Ray workers fork processes; Python globals don't share, and gradients silently never propagate to rollouts (training appears to work but never improves).

```python
import torch, torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

class CoreEncoderModule(TorchRLModule):
    """§4.3 universal core obs → 256-d feature. Parameter-shared via
       additional_module_specs reference from each role policy."""
    def setup(self):
        cfg = self.model_config
        in_dim = core_obs_flat_dim()
        layers, prev = [], in_dim
        for h in cfg["core_hidden"]:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
    def _forward_train(self, batch):     return self._fwd(batch)
    def _forward_inference(self, batch): return self._fwd(batch)
    def _forward_exploration(self, batch): return self._fwd(batch)
    def _fwd(self, batch):
        return {"core_feat": self.net(flatten_core(batch["obs"]))}

class SharedBackboneModule(TorchRLModule):
    """§4.3 projection + LSTM(256). Parameter-shared. Stateful per agent_uuid."""
    def setup(self):
        cfg = self.model_config
        self.proj = nn.Linear(448, 384)
        self.lstm = nn.LSTM(384, cfg["lstm_size"], num_layers=1, batch_first=True)
        self.lstm_size = cfg["lstm_size"]
    def get_initial_state(self):
        return {"h": torch.zeros(1, self.lstm_size),
                "c": torch.zeros(1, self.lstm_size)}
    def _forward_train(self, batch):     return self._fwd(batch)
    def _forward_inference(self, batch): return self._fwd(batch)
    def _forward_exploration(self, batch): return self._fwd(batch)
    def _fwd(self, batch):
        x = batch["fused_features"]                    # (B, 448)
        state = (batch["state_in"]["h"].unsqueeze(0),
                 batch["state_in"]["c"].unsqueeze(0))
        proj = torch.relu(self.proj(x))
        out, new_state = self.lstm(proj.unsqueeze(1), state)
        return {"hidden": out.squeeze(1),
                "state_out": {"h": new_state[0].squeeze(0),
                              "c": new_state[1].squeeze(0)}}

class CTDECriticModule(TorchRLModule):
    """Two-stage encoder per §4.3."""
    def setup(self):
        cfg = self.model_config
        self.per_agent_encoder = nn.Sequential(
            nn.Linear(per_agent_obs_dim(), 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * 128 + village_inv_dim(), cfg["critic_hidden"]),
            nn.ReLU(),
            nn.Linear(cfg["critic_hidden"], 1),
        )
    def _forward_train(self, batch):
        compressed = self.per_agent_encoder(batch["privileged_state"]["all_agents_obs"])
        flat       = compressed.flatten(start_dim=1)
        x          = torch.cat([flat, batch["privileged_state"]["village_inv"]], dim=-1)
        return {"vf_preds": self.head(x).squeeze(-1)}
    def _forward_inference(self, batch):  return {}
    def _forward_exploration(self, batch): return {}

class AiUtopiaRoleRLModule(TorchRLModule):
    """Per-role module: role encoder + (optional) pixel encoder + actor head."""
    def setup(self):
        cfg              = self.model_config
        self.role        = cfg["role"]
        self.role_encoder = _build_role_encoder(self.role,
                                                self.observation_space,
                                                cfg["role_hidden"])
        self.pixel_encoder = (_build_pixel_encoder(self.role,
                                                    self.observation_space,
                                                    cfg["pixel_hidden"])
                              if cfg.get("pixel_hidden") else None)
        self.actor = MultiHeadActor(
            input_dim=cfg["lstm_size"] + 512,
            action_space=self.action_space,
            hidden=cfg["actor_hidden"],
            use_action_masking=cfg["use_action_masking"],
        )
    def get_initial_state(self):
        # delegates to SharedBackboneModule's state
        return self.multi_rl_module["shared_backbone"].get_initial_state()
    def _forward_train(self, batch):
        return self._fwd(batch, training=True)
    def _forward_inference(self, batch):
        return self._fwd(batch, training=False)
    def _forward_exploration(self, batch):
        return self._fwd(batch, training=False)
    def _fwd(self, batch, training):
        core_out  = self.multi_rl_module["core_encoder"]._fwd(batch)["core_feat"]
        role_f    = self.role_encoder(batch["obs"])
        pixel_f   = (self.pixel_encoder(batch["obs"]) if self.pixel_encoder
                     else torch.zeros(core_out.shape[0], 64, device=core_out.device))
        fused     = torch.cat([core_out, role_f, pixel_f], dim=-1)
        batch["fused_features"] = fused
        bb_out    = self.multi_rl_module["shared_backbone"]._fwd(batch)
        hidden    = bb_out["hidden"]
        actor_in  = torch.cat([hidden, batch["obs"]["goal_embedding"]], dim=-1)
        action_dist_inputs = self.actor(actor_in, batch["obs"].get("action_mask"))
        result = {
            "action_dist_inputs": action_dist_inputs,
            "state_out":          bb_out["state_out"],
        }
        if training:
            critic_out = self.multi_rl_module["ctde_critic"]._forward_train(batch)
            result["vf_preds"] = critic_out["vf_preds"]
        return result
```

### 7.3 `AiUtopiaPettingZooEnv`

```python
from pettingzoo import ParallelEnv
from py4j.java_gateway import JavaGateway, GatewayParameters

class AiUtopiaPettingZooEnv(ParallelEnv):
    metadata = {"name": "aiutopia_minecraft_v0", "render_modes": []}

    def __init__(self, config: dict):
        self.cfg          = config
        self.active_roles = config["active_roles"]
        self.agents_init  = [f"{r}_0" for r in self.active_roles]
        self.possible_agents = list(self.agents_init)
        self.agents       = []
        self.stage        = config["stage"]
        self.tick_warp    = config["tick_warp"]

        port              = config["py4j_ports"][config.worker_index % 4]
        self.gw           = JavaGateway(GatewayParameters(port=port, auto_field=True))
        self.fabric       = self.gw.entry_point        # UnionClef fork
        self.motor        = self.fabric.motorBridge()

        self.skill_counters   = {}
        self.exploit_detectors = {a: ExploitDetector() for a in self.agents_init}
        self.epi_writer        = EpisodicMemoryWriter(self.cfg["chroma_path"])

    def observation_space(self, agent: str):
        return build_role_observation_space(role_of(agent), stage=self.stage)
    def action_space(self, agent: str):
        return build_role_action_space(role_of(agent))

    def reset(self, seed=None, options=None):
        # per_worker_seed_offset randomizes only the random-strategy rollout
        if seed is None:
            seed = self._next_seed_for_strategy()
        self.fabric.resetWorld(seed)
        self.agents = list(self.agents_init)
        self.skill_counters = {a: 0 for a in self.agents}
        self._tick = 0
        obs = self._read_all_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, dict]):
        # 1. Dispatch all agent actions (mid-tick comm flush)
        comm_msgs = []
        for agent, act in actions.items():
            self.skill_counters[agent] += 1
            invocation_id = f"{agent}-{self.skill_counters[agent]}"
            self.motor.dispatchSkill(agent, encode_action(act), invocation_id)
            if act["should_broadcast"] and act["comm_target_mask"].any():
                comm_msgs.append(build_comm_msg(agent, act))
        self.fabric.commBus().flushBatch(comm_msgs)   # mid-tick flush

        # 2. Advance world; collect SkillCompletionEvents (30 s cap)
        completion_events = self.fabric.advanceTickAwaitEvents(timeout_ms=30_000)

        # 3. BATCHED observation read — one Py4J call returns all agents' obs
        new_obs = self._read_all_obs()

        # 4. Compute rewards, terminations
        rew, term, trunc, info = {}, {}, {}, {}
        for agent in list(self.agents):
            rew[agent]  = compute_reward(role_of(agent), self.stage,
                                          self._prev_obs[agent], new_obs[agent],
                                          actions[agent], self._env_meta(agent))
            term[agent] = self._is_terminal(agent)
            trunc[agent]= self._tick >= self.cfg["max_episode_ticks"]
            self.exploit_detectors[agent].step(role_of(agent),
                                                self._prev_obs[agent], new_obs[agent],
                                                actions[agent], self._env_meta(agent))
            self.epi_writer.maybe_write(agent, self._tick,
                                         self._prev_obs[agent], new_obs[agent],
                                         actions[agent], rew[agent])
            info[agent] = {"skill_completed": agent in completion_events,
                           "exploit_log":    self.exploit_detectors[agent].last_step}
        self._prev_obs = new_obs
        self._tick += 1
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return new_obs, rew, term, trunc, info

    def _read_all_obs(self) -> dict[str, dict]:
        """Single BATCHED Py4J call — fabric.observationsAll() returns one
           JSON blob per env containing every agent's obs. Avoids
           N-roundtrips-per-tick that would cap throughput at ~300 agent-steps/sec."""
        raw_all = self.fabric.observationsAll()         # JSON string
        parsed  = json.loads(raw_all)
        return {agent: decode_obs(parsed[agent], role_of(agent),
                                   stage=self.stage,
                                   action_mask=compute_action_mask(
                                                  parsed[agent], role_of(agent)))
                for agent in self.agents}

    def close(self):
        """PettingZoo lifecycle requirement. Without this, Ray worker shutdown
           leaks Java processes that hold Py4J ports; after dozens of runs,
           can't spawn new JVMs."""
        self.epi_writer.flush()
        try:    self.fabric.shutdown()
        finally: self.gw.shutdown()
```

### 7.4 Training driver

```python
# scripts/train.py — NOT containerized; runs on host machine
# (Ray + RLlib fight Docker process isolation. Production runtime is
#  containerized via docker-compose.production.yml. Training is not.)
import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig

ray.init(num_cpus=16, num_gpus=1,
         object_store_memory=8 * 1024**3,
         _system_config={"object_spilling_threshold": 0.95})

tuner = tune.Tuner(
    trainable="PPO",
    param_space=m_config_for(milestone).to_dict(),
    run_config=RunConfig(
        name=f"aiutopia_{milestone}_{run_id}",
        storage_path="/var/lib/aiutopia/runs",
        checkpoint_config=CheckpointConfig(
            checkpoint_frequency=50,
            num_to_keep=10,
            checkpoint_at_end=True,
            checkpoint_score_attribute="env_runners/episode_return_mean",
            checkpoint_score_order="max",
        ),
        verbose=1,
        log_to_file=True,
        callbacks=[
            tune.logger.TBXLoggerCallback(),
            AiUtopiaMetricsCallback(),          # per-role entropy, Q-var, traj cos
            ExploitHuntCallback(every_n_iters=200),
            EvalGateStopCallback(milestone),    # early-stop on §5.8 gate pass
        ],
        stop={"training_iteration": milestone_iter_cap[milestone]},
    ),
)
results = tuner.fit()
# TensorBoard: tensorboard --logdir /var/lib/aiutopia/runs --bind_all --port 6006

milestone_iter_cap = {
    "M1": 2_000, "M2": 4_000, "M3": 3_500,
    "M4": 5_000, "M5": 6_000, "M6": 8_000,
}
```

`milestone_iter_cap` is a **budget ceiling**; `EvalGateStopCallback` triggers early stop when the §5.8 milestone-specific eval gate passes.

### 7.5 CLI tooling

Typer-based, single entry point `aiutopia = "aiutopia.cli:app"` in `pyproject.toml`.

```
aiutopia promote-weights
  --role {gatherer|builder|farmer|defender}
  --from-checkpoint <ray-run-id>/checkpoint_N
  [--bump-version | --rollback]
  --notes "<reason>"
  - Verifies §5.10 promotion checklist (fail-fast)
  - Copies weights → /var/lib/aiutopia/weights/{role_id}/v{n}.ckpt
  - UPDATE roles, INSERT policy_deployments
  - POSTs /admin/hot-reload to production planner; weights reload at next
    subgoal boundary. Per-agent LSTM hidden state PRESERVED across reload —
    agent has continuity of mind even though weights changed. Reload does
    NOT happen mid-skill.

aiutopia cache-invalidate
  [--cache-key <hash> | --all | --by-event <event-type>]

aiutopia planner trigger-replan --agent <uuid_or_name> [--reason "<str>"]

aiutopia agent kill   --agent <agent_uuid> --cause "<str>"
aiutopia agent spawn  --role <role> [--name <str>]

aiutopia chat send    --as-player <name> --to <agent_name> --text "<msg>"

aiutopia eval scenario --scenario <held_out_id> --role <role> [--seeds N]

aiutopia memory inspect --agent <agent_uuid> [--query "<str>" --top-k 10]

aiutopia world snapshot [--out <path>]
  - Saves Fabric world + planner_state + identity DB + Chroma collections
    to single tarball; idempotent reset point.

aiutopia determinism check --weights <ckpt> --episodes N
  - Runs §7.8 determinism harness across N seeds; reports pass/fail counts.
```

### 7.6 Production deployment — Docker Compose

```yaml
services:
  fabric-prod:
    image: aiutopia/fabric-server:1.21.5
    container_name: aiutopia-fabric-prod
    cpuset: "12-13"
    mem_limit: 4g
    volumes:
      - ./world:/server/world
      - ./mods:/server/mods:ro
      - ./logs/fabric:/server/logs
    ports:
      - "25565:25565"
      - "127.0.0.1:25100:25100"     # Py4J — localhost only
    environment:
      # Generational ZGC — NOT G1GC (G1's 200ms pauses stall the tick loop)
      - JAVA_OPTS=-Xms3g -Xmx3g
                  -XX:+UseZGC -XX:+ZGenerational
                  -XX:+UnlockExperimentalVMOptions
                  -XX:+UseTransparentHugePages
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:25100/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  planner:
    image: aiutopia/planner:latest
    cpuset: "14"
    mem_limit: 4g
    depends_on: [fabric-prod, chroma]
    volumes:
      - ./var/lib/aiutopia:/var/lib/aiutopia
      - ./goal_templates:/var/lib/aiutopia/goal_templates:ro
    environment:
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
      - PY4J_HOST=fabric-prod
      - PY4J_PORT=25100
      - CHROMA_URL=http://chroma:8000
      - LLM_MODEL=claude-haiku
      - LLM_BUDGET_HARD_CAP_USD_MONTH=80
      - QWEN_LOCAL_URL=http://planner-llm-local:8001
    secrets: [anthropic_key]
    restart: unless-stopped

  # Local fallback LLM. Profile-gated: only started when budget cap triggers.
  planner-llm-local:
    image: aiutopia/llm-local:qwen14b-int4
    cpuset: "14"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "127.0.0.1:8001:8001"
    profiles: [llm-fallback]

  chroma:
    image: chromadb/chroma:0.5.20
    cpuset: "15"
    mem_limit: 2g
    volumes:
      - ./var/lib/aiutopia/chroma:/chroma/chroma
    ports:
      - "127.0.0.1:8000:8000"

  dashboard-feed:
    image: aiutopia/dashboard-feed:latest
    cpuset: "15"
    mem_limit: 512m
    volumes:
      - ./var/lib/aiutopia:/var/lib/aiutopia:ro
    ports:
      - "127.0.0.1:8080:8080"        # localhost only — never expose publicly

secrets:
  anthropic_key:
    file: ./secrets/anthropic_key
```

Systemd unit `/etc/systemd/system/aiutopia.service` calls `docker compose -f docker-compose.production.yml up`; `Restart=on-failure`, `RestartSec=10`. Compose chosen over native systemd-per-service because Fabric mods directory benefits from container-isolated immutable mounts.

**Backup policy:**
- Daily: incremental `rsync` of `/var/lib/aiutopia` to date-stamped directory on NAS/USB.
- Weekly: full tarball of `/var/lib/aiutopia`.
- Retention: 7 dailies + 4 weeklies.

### 7.7 Tauri dashboard data feed

Localhost-only FastAPI SSE sidecar. Bound to `127.0.0.1` in both Compose port mapping and uvicorn host arg; documented as containing sensitive data and must never be exposed publicly.

```python
# services/dashboard-feed/server.py
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio, sqlite3, json

app = FastAPI(title="aiutopia-dashboard-feed")

@app.get("/snapshot")
def snapshot():
    """Full state: alive agents, current plans, village inventory,
       last 50 events. Tauri loads on connect."""
    return {"agents":   _query_alive_agents(),
            "plans":    _query_active_plans(),
            "village":  _query_village_inventory(),
            "events":   _query_recent_events(limit=50)}

@app.get("/events")
async def events():
    """SSE stream: every relevant state change. Tauri subscribes on app start."""
    async def gen():
        last_tick = 0
        while True:
            new = _query_events_since(last_tick)
            for e in new:
                yield {"event": e["type"], "data": json.dumps(e)}
                last_tick = max(last_tick, e["tick"])
            await asyncio.sleep(0.5)        # 2 Hz SQLite poll
    return EventSourceResponse(gen())

@app.get("/agent/{uuid}/memory")
def agent_memory(uuid: str, query: str = "", limit: int = 20):
    return MemoryRetriever().retrieve(uuid, query, top_k=limit)

@app.get("/metrics/training")
def training_metrics():
    return _read_ray_run_summary("/var/lib/aiutopia/runs/latest")

@app.get("/llm/cost")
def llm_cost():
    return _aggregate_llm_costs(window_days=30)

# Run with: uvicorn server:app --host 127.0.0.1 --port 8080
```

**Tauri panels** (six per Gemini's design): Agent List, World Map (top-down agent positions + village outline), Current Plan DAG (per agent), Reward Curves (live TB pull), LLM Cost (rolling 30-day spend vs $80 cap), Memory Inspector (per agent, query).

### 7.8 Determinism harness

```python
# tests/determinism/test_seeded_replay.py
import os, numpy as np, pytest, torch
from aiutopia.env import AiUtopiaPettingZooEnv

DET_CFG = {"stage": 1, "active_roles": ["gatherer"],
           "seed_strategy": "fixed_easy", "tick_warp": True,
           "py4j_ports": [25099], "max_episode_ticks": 1_000,
           "per_worker_seed_offset": False}
EPS_ARGMAX = 0.05
EPS_L2     = 0.10

@pytest.fixture(autouse=True)
def _cuda_determinism():
    """Without this, cuDNN autotuner randomizes and the gate is flaky."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    yield

@pytest.mark.determinism
def test_seeded_run_reproducible(snapshot_weights_path):
    """§5.10 gate #5. Same seed + same weights → same trajectory within ε."""
    torch.manual_seed(1); np.random.seed(1)
    m1, env1 = _load(snapshot_weights_path), AiUtopiaPettingZooEnv(DET_CFG)
    trace1   = _run_episode(env1, m1, seed=1)
    try: env1.close()
    except: pass

    torch.manual_seed(1); np.random.seed(1)
    m2, env2 = _load(snapshot_weights_path), AiUtopiaPettingZooEnv(DET_CFG)
    trace2   = _run_episode(env2, m2, seed=1)
    try: env2.close()
    except: pass

    assert len(trace1) == len(trace2), "trajectory length divergence"
    argmax_div = np.mean([t1.action_argmax != t2.action_argmax
                          for t1, t2 in zip(trace1, trace2)])
    l2_div     = np.mean([np.linalg.norm(t1.continuous_params - t2.continuous_params)
                          for t1, t2 in zip(trace1, trace2)])
    assert argmax_div < EPS_ARGMAX, f"action_argmax_divergence={argmax_div:.3f}"
    assert l2_div     < EPS_L2,     f"continuous_param_L2={l2_div:.3f}"
```

---

## 8. Open Decision Points

These remain open after the design review; each is small, defer-friendly, and called out for the implementation plan to either lock or revisit:

| ID | Decision | Default if not addressed |
|---|---|---|
| OD-1 | LSTM `hidden_size`: 256 vs 128 | 256 initial; try 128 if GPU pressure or sample-efficiency issues |
| OD-2 | Curriculum decay slope (linear vs cosine vs step) over 10M steps | Linear; replan after first M2 return curve |
| OD-3 | Stage-2 weight sensitivity sweep (±20%) | Run at end of M3 via fine-tune from M3 checkpoint, 500K steps each |
| OD-4 | Plan cache event-driven invalidation triggers complete? | List in §6.7 is initial set; add as new event types observed |
| OD-5 | LLM-composed memory retrieval queries | Static templates only for v1; escalate Phase 5+ if static miss > 20% |
| OD-6 | LLM classifier for `expected_reply_type` | Regex/keyword heuristic in v1; escalate if miscat > 10% |
| OD-7 | Off-machine backup target (NAS/USB) | Hardware-dependent; pick before M5 production deploy |

---

## 9. Glossary

| Term | Meaning |
|---|---|
| Agent | One of 4 specialized RL-driven entities in the village (gatherer, builder, farmer, defender). Has `agent_uuid`, persistent identity Level D. |
| `agent_uuid` | ULID identifying a specific *life* of an agent. Rotates on death; new successor gets new UUID. |
| `role_id` | Stable identifier `{gatherer\|builder\|farmer\|defender}`. Survives death; policy weights belong to role, not to agent. |
| Backbone | Shared neural network feature extractor (`LSTM(256)`) parameter-shared across all 4 roles. |
| BGE-small | Frozen pre-trained sentence embedder (`BAAI/bge-small-en-v1.5`, 384-d). Used for goal embeddings and memory embeddings. |
| Carpet fake player | Server-side simulated player created via Carpet mod's `/player <name> spawn`. The production-server embodiment of each agent. |
| ChatBridge | Component intercepting `@<agent_name>` chat and routing to the LLM planner. |
| Chroma | Embedded vector DB. Stores `mem_{uuid}` and `skill_lib_{uuid}` collections. |
| CTDE | Centralized Training, Decentralized Execution. CTDE critic has access to all agents' obs + village inventory during training; actors see only their own obs. |
| `compute_reward()` | Single stage-dispatched reward function (§5.2). |
| `EventQueue` | Priority queue feeding the LLM planner. Priorities: 0=ChatEvent, 1=FailureReport, 2=WorldEvent, 3=PhaseTransition. |
| `ExploitDetector` | Per-agent stateful detector for the 6 reward-hacking exploit types (§5.3). |
| Fabric | Modern Minecraft Java modding framework. Selected over Forge for performance. |
| `goal_embedding` | 512-d concat of 384-d frozen BGE NL embedding + 128-d structured features. Conditions per-role policy. |
| `GoalSpecAdapter` | Deterministic Python module that converts JSON `Subgoal` to `goal_embedding`. The Tier 2 of the 3-tier brain. |
| HAPPO | Heterogeneous-agent PPO. Phase 5+ escalation if MAPPO collapses on heterogeneity. |
| IPPO | Independent PPO (no centralized critic). M1 start algorithm. |
| LAIES | Lazy-Agent Intrinsic Exploration of States. IDI + CDI intrinsic rewards from External States Transition Model. |
| `LlmPlanOutput` | Typed Pydantic v2 schema for the JSON DAG of subgoals the planner emits (§6.2). |
| MAPPO | Multi-Agent PPO with shared centralized critic. M2+ escalation from IPPO. |
| MAPF | Multi-Agent Path Finding. Conditional add at M4 if collision deadlock detected. |
| PBRS | Potential-Based Reward Shaping. `F(s,s') = γ·Φ(s') − Φ(s)`. Policy-invariant per Ng/Harada/Russell 1999. |
| PettingZoo | Multi-agent Gym-style env API. Parallel variant used here. |
| Production world | The persistent 20-TPS Fabric server agents live on. Friends join here. |
| Py4J | Python ↔ Java bridge. UnionClef-fork is the working production implementation. |
| ROMA | Role-Oriented Multi-Agent. Role-collapse mitigation. |
| Stage 1 / 2 / 3 | Reward stage (§5.1); selected by `compute_reward()` per training milestone. |
| Subgoal | One node in the LLM-emitted DAG; assigned to one `role` with a `goal_specification` and `termination_conditions`. |
| Training world | The 4 disposable, tick-warped, resettable Fabric instances. |
| ULID | Universally Unique Lexicographically Sortable Identifier (Crockford base32, time-ordered). The system's sole ID format. |

---

## 10. References

### Project research

- `GeminiResearch.md` — independent Gemini deep-research pass (47 KB)
- `Kimi_Agent_Minecraft AI Village Research/minecraft_village_research.agent.final.md` — Kimi consolidated report (300 KB)
- `Kimi_Agent_Minecraft AI Village Research/minecraft_village_research.agent.outline.md` — section outline (16 KB)
- `Kimi_Agent_Minecraft AI Village Research/minecraft_village_research_sec00.md` … `sec11.md` — 12-section split
- `Kimi_Agent_Minecraft AI Village Research/research/minecraft_village_dim01.md` … `dim10.md` — 10 research dimensions
- `Kimi_Agent_Minecraft AI Village Research/research/minecraft_village_cross_verification.md` — confidence tiers + conflict zones
- `Kimi_Agent_Minecraft AI Village Research/research/minecraft_village_insight.md` — 10 cross-dimensional insights

### Top-priority external repos / papers (per `sec01:1.4` and `sec10:10.1`)

1. `mindcraft-bots/mindcraft` — actively maintained multi-agent LLM framework (clone as scaffold)
2. `3ndetz/unionclef` — production Py4J two-way bridge for Fabric 1.21.x (fork as bridge)
3. `MineDojo/Voyager` — skill-library architecture (read-only; archived)
4. `cocacola-lab/MineLand` — 48-agent simulator (study three-module architecture)
5. HeMAC paper (ECAI 2025) — IPPO-over-MAPPO in highly heterogeneous scenarios
6. COMA (Foerster et al., AAAI 2018) — counterfactual baseline formula
7. Ray RLlib multi-agent docs — `policy_mapping_fn`, `MultiRLModuleSpec`, fractional GPU
8. `zju-vipa/odyssey` — 40 primitive + 183 compositional skills, multi-agent DAG
9. LAIES paper — IDI/CDI intrinsic rewards for lazy-agent mitigation
10. `JiuTian-VL/Optimus-3` — MoE dual-router; reference for LLM planner design (not for RL training)

### Standards / specifications

- Pydantic v2 — schema validation: <https://docs.pydantic.dev/latest/>
- Ray RLlib RLModule API — <https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html>
- PettingZoo Parallel API — <https://pettingzoo.farama.org/api/parallel/>
- Chroma documentation — <https://docs.trychroma.com/>
- Generational ZGC — <https://openjdk.org/jeps/439>
- ULID specification — <https://github.com/ulid/spec>

---

*End of spec. Next step: invoke `superpowers:writing-plans` skill to produce `IMPLEMENTATION_PLAN.md` for Milestone 0 (infrastructure foundation).*
