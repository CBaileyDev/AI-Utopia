# Phase D — Decision-Core: Java (Fabric mod) Scoping

**Date:** 2026-05-29
**Status:** READ-ONLY scoping. No code written. Build/deploy is a USER-RESERVED step.
**Goal:** mirror the proven Python-sim "M2 decision-core" mechanism into the real
Fabric mod (MC 1.21.1, Java 21), flag-gated so the default-OFF survival-forest
HARVEST path is byte-for-byte untouched.

## 0. The mechanism in one paragraph (what we are mirroring)

In the sim, when `decision_core=True`, HARVEST (`skill_type=1`) is DEMOTED: instead
of `findNearest` + chaining through the whole field, it mines ONLY the single trunk
the policy points at. The policy's `target_class` is REINTERPRETED as an **instance
pointer** `k` = the k-th entry of the gatherer's "nearest resources" list. That list
MUST be the SAME ordered list the policy saw in `g_nearest_resources`. Two obs-side
companions (both flag-gated): (a) a PERCEPTION-based HARVEST mask (valid whenever a
trunk is *visible*, not only in reach), and (b) a ground-truth "bearing cue" written
into `g_hostiles_nearby[0]` toward the nearest alive log. Plus a "clusters"
blind-explore arena variant in WorldOps.

The authoritative Python contract is `src/aiutopia/sim/sim_env.py`
(`_dispatch_decision_core`, lines 206-223) + `src/aiutopia/sim/skills.py`
(`mine_instance`, lines 216-239) + `src/aiutopia/sim/obs_adapter.py`
(`gatherer_nearest_columns`, lines 97-142). Java must match these.

---

## 1. Current Java logic (file:line references)

### 1.1 `fabric_mod/src/main/java/dev/aiutopia/mod/bridge/skill/HarvestSkill.java`

The do-everything HARVEST. Key constants and flow:

- `MAX_SEARCH_RADIUS = 48.0` (HarvestSkill.java:54) — DELIBERATELY wider than the
  obs `SCAN_RADIUS=16` so chaining can reach logs the policy can't see (header
  comment lines 29-36).
- `REACH_RADIUS = 4.5` / `REACH_RADIUS_SQ` (lines 66-67); `WALK_PER_TICK = 4.3/20.0`
  (line 68); `MAX_QUANTITY = 64` (line 69); `BREAK_TICKS_PER_LOG = 15` (line 83).
- `start(...)` (lines 97-118): `target_class` → index into `TARGET_CLASS_TABLE`
  (lines 104-105, 120-123); `scalar_param` → `cap = round(scalar*64)` (lines 108-113).
- `tick(...)` (lines 125-246): the **chaining loop** —
  - line 135-155: if no `currentTarget`, `findNearest(...)`; empty → COMPLETED if
    `brokenCount>0` else IMMEDIATE_FAILURE.
  - line 156-193: walk toward target at full `WALK_PER_TICK` (N16b), stall watchdog.
  - line 205-244: in reach → mine for `BREAK_TICKS_PER_LOG` ticks, then
    `Block.getDroppedStacks` + `offerOrDrop` + `setBlockState(AIR)` (N16c direct
    inventory insert), `brokenCount++`, `currentTarget=null`, loop.
- `findNearest(...)` (lines 248-257): two-pass — ground band `dy∈[-2,+1]` first,
  then full `[-radius, radius]`.
- `scanShell(...)` (lines 259-281): triple `dx/dy/dz` loop over the 48-radius cube,
  Euclidean nearest (`Math.sqrt(dx²+dy²+dz²)`, strict `<`), returns single best.

> **This is a COMPLETELY different algorithm from `g_nearest_resources`.** It is
> Euclidean-nearest over a 97×N×97 cube with a two-pass `dy` preference, NOT the
> per-column-topmost-within-±3 over a 32×32 grid the obs uses. See §3.

### 1.2 `fabric_mod/.../obs/GathererOverlayBuilder.java` (PARITY-CRITICAL)

Builds `g_nearest_resources`. The exact ordering the policy sees:

- `GRID_RADIUS=16`, `SCAN_RADIUS=16`, `TOP_K_NEAREST=8` (lines 28-31).
- **Per-column topmost scan** (lines 55-74): outer `dx ∈ [-16,16)`, inner
  `dz ∈ [-16,16)`; inner-inner `dy = 3 → -3` (top-down). First non-air block
  hits `break` (line 71) — i.e. **topmost non-air per column**. If that block
  matches a resource channel and `sqrt(dx²+dy²+dz²) ≤ 16`, it is appended to
  `nearby` (line 67) and `grid[...][channel]=1` (line 65).
- `g_nearest_resources` pack (lines 84-102): `nearby.sort(Comparator.comparingDouble(distSq))`
  (line 85, **distSq only — see §3**), then top-8 rows
  `[dx/16, dy/8, dz/16, channel, 1.0, 1.0]` (lines 91-96), zero-padded (line 98).
- `nearest_resource_distance` (lines 136-139): `sqrt(nearby.get(0).distSq())` or
  `SENTINEL_NO_TARGET=999.0` → Python derives `target_resource_in_range`.
- `g_hostiles_nearby` (lines 107-127): real hostiles, sorted by squared distance,
  rows `[dx/16, dy/16, dz/16, typeId]`. In the peaceful training arena
  (`/difficulty peaceful` + `doMobSpawning false`, WorldOps:189,192) this is
  ALWAYS all-zeros — which is exactly why the bearing cue can safely overwrite row 0.
- `NearbyResource.distSq()` = `dx²+dy²+dz²` (lines 178-180).

### 1.3 `fabric_mod/.../bridge/WorldOps.java`

- `resetEpisode(playerName, seed)` (lines 109-173): tp to `64 66 -48`, `/clear`,
  equip stone axe, `/fill 48 65 -64 80 70 -32 air`, then 16 trunks on a 4×4 grid
  (`baseX=52+7*col`, `baseZ=-61+7*row`), per-trunk `epRand.nextInt(3)-1` jitter
  (x THEN z), clamp to `[48,80]×[-64,-32]`, dedup-nudge off spawn tile, 4-log
  stack `Y=65..68`. MUST match `sim/world.py::_tree_grid_bases` byte-for-byte.
- `setupTrainingScene()` (lines 184-212): difficulty peaceful, no daylight, no mob
  spawning, **`/forceload add 32 -64 96 -16`** (line 194), grass floor
  **`/fill 48 60 -64 80 64 -32 grass_block`** (line 196), air clear
  `/fill 48 65 -64 80 80 -32 air` (line 197), `/tick rate 60.0` (line 204).
- `key(x,z)` (lines 175-178); RNG `epRand` (line 80).

### 1.4 Dispatch path & how `target_class` reaches the skill

- `Py4JEntryPoint` (Py4JEntryPoint.java): `observationsAll()` (line 45-47),
  `motorBridge()` (line 49), `resetEpisode(playerName, seed)` (lines 90-93),
  `setupTrainingScene()` (lines 95-98), `getItemIdNameTable()` (lines 113-115),
  `runCommand(...)` (lines 78-88). This is the entire Python-facing surface.
- `MotorBridge.dispatchSkill(agentId, encodedAction, skillInvocationId)`
  (MotorBridge.java:87-91) → marshals to server thread →
  `dispatchOnServerThread` (lines 93-148): parses JSON, reads `skill_type`
  (line 129), constructs executor via `newExecutorForSkillType` (lines 152-162;
  `1 → new HarvestSkill()`), calls `exec.start(agent, action, server)` (line 139).
- `target_class` lives in the action JSON; `HarvestSkill.start` reads it
  (HarvestSkill.java:104). **The full `action` JsonObject is available to the skill**
  — so a per-dispatch `decision_core` flag could ride in the action JSON, OR be read
  from a server-side config holder (recommended — see §2.0).
- **No action mask is computed on the Java side.** The mask is built entirely in
  Python (`src/aiutopia/env/action_mask.py`) from raw obs facts Java emits
  (`nearest_resource_distance`, `inv_slot_counts`, etc.). Java exposes the facts;
  Python decides legality. (Important for §2.2.)

---

## 2. Exact edit points (all FLAG-GATED, default OFF)

### 2.0 Flag mechanism (recommended)

Add a tiny server-side **config holder** read by the obs builder, HarvestSkill, and
WorldOps. Set it via one new Py4J method. This keeps the default-OFF path byte-for-byte.

- New class `fabric_mod/.../bridge/DecisionCoreConfig.java` (or static fields on
  `Py4JEntryPoint`/`WorldOps`) holding `volatile boolean`s:
  `decisionCore`, `harvestPerceptionMask` (NOT needed on Java side — see §2.2),
  `resourceBearingCue`, plus `String arenaMode` (`"trees"`/`"clusters"`).
  All default to the OFF/`"trees"` values.
- New `Py4JEntryPoint` method, e.g.:
  `void setDecisionCoreConfig(boolean decisionCore, boolean resourceBearingCue, String arenaMode)`
  marshalled onto the server thread (or plain volatile writes — reads happen on the
  server tick thread; volatile is sufficient). Python sets it once after spawn,
  before the first reset (consistent with how `setupTrainingScene()` is called once).
- **Backward-compatible `resetEpisode` overload.** Keep the existing
  `resetEpisode(String, long)` (bridge.py:136 calls the 2-arg form). Add
  `resetEpisode(String playerName, long seed, String arenaMode)` that branches on
  `arenaMode`; the 2-arg form delegates with `"trees"`. This avoids touching the
  Python 2-arg call site for the default path.

Rationale (template already in repo): `arena_mode` already flows config → reset in
the sim (`config.py:136`, `sim_env.py:153/194`). Mirror that shape.

### 2.1 Decision-core HARVEST = mine the k-th nearest column only

**Central edit — extract the ordered `nearby` list into a shared method so the obs
builder and HARVEST index the SAME list (parity by construction, not by hand).**

1. In `GathererOverlayBuilder.java`, extract the per-column scan + sort (lines
   55-74 + line 85) into a **public static** method, e.g.:
   `public static List<int[]> orderedNearbyColumns(ServerWorld world, BlockPos origin)`
   returning entries `[worldX, worldZ, dy]` (or `[dx, dy, dz]`) **already sorted**
   exactly as today (`distSq`, stable insertion order). Have `populate(...)` call it
   so the obs pack is unchanged. (See §3 for why the existing distSq-only sort is
   already correct and must NOT be "fixed".)

2. In `HarvestSkill.tick`, add a `decisionCore` branch (gated by the §2.0 flag, read
   in `start`): when ON,
   - DO NOT call `findNearest`/`scanShell` and DO NOT chain.
   - On `start`, resolve the instance ONCE:
     `List<int[]> nearby = GathererOverlayBuilder.orderedNearbyColumns(world, agent.getBlockPos())`.
     - if `nearby.isEmpty()` → IMMEDIATE_FAILURE, reason
       `"no resource in perception (NAVIGATE to explore)"` (mirror
       `sim_env.py:214-219`).
     - `int k = targetClassRaw; k = Math.max(0, Math.min(k, nearby.size()-1));`
       (**CLAMP**, mirror `sim_env.py:221` — NOT failure on out-of-range).
     - record the chosen column world `(tx, tz)`.
   - During `tick`, harvest ONLY the alive logs in column `(tx, tz)`, **bottom-up**
     (mirror `mine_instance` sort by Y, skills.py:233): walk into reach of each log
     (reuse the existing N16b walk + `BREAK_TICKS_PER_LOG` + N16c direct insert),
     break it, advance to the next alive log in the same column. When the column has
     no alive logs left → COMPLETED if `brokenCount>0` else IMMEDIATE_FAILURE.
   - **`cap` semantics:** the sim's `mine_instance` mines the WHOLE column regardless
     of `scalar_param` (skills.py:234-238). Match that — in decision-core mode ignore
     `cap` and clear the pointed column. (If you keep `cap`, the sim's "4 logs per
     column" episode-length parity breaks — see §3.)

> Re-resolve vs resolve-once: the sim resolves `k` against the CURRENT obs each step
> (`sim_env.py:213`), and one decision-core step == one column. The Java skill spans
> many ticks for one dispatch. Resolve the column ONCE in `start` (the obs the policy
> acted on is the pre-dispatch obs) and mine that fixed column to completion. This is
> the faithful mapping of "one env step = mine one pointed column".

### 2.2 Perception-based HARVEST mask — likely ZERO Java change

The mask is Python-side (`action_mask.py:44` keys on `target_resource_in_range`,
derived from obs). "Resource visible in perception" == `g_nearest_resources` non-empty
== `nearest_resource_distance < SENTINEL` — which Java ALREADY emits
(GathererOverlayBuilder.java:136-139). The sim toggles this purely in Python
(`obs_adapter.py:205-220`: `bool(nearby)` vs `nearest_res_dist <= REACH`).

**Scope:** mirror the perception mask in the **Python wrapper / `_normalize_raw`
path** (the toggle that decides whether `target_resource_in_range` means "in reach"
or "in perception"), NOT in Java. Do not add a redundant Java mask field. (Confirm
the exact Python toggle location during implementation — it is the env-side analog of
`build_gatherer_obs(harvest_mask_on_perception=...)`. This is a
`minecraft-rl-environment-specialist` handoff, not a mod change.)

### 2.3 Bearing cue → `g_hostiles_nearby[0]` (FLAG-GATED Java edit)

Mirror `obs_adapter.py:191-200`. In `GathererOverlayBuilder.populate`, gated by the
`resourceBearingCue` flag (§2.0), AFTER building `g_hostiles_nearby` (or by replacing
its row 0):

- Scan for the **nearest alive log anywhere** — crucially, BEYOND `SCAN_RADIUS` (the
  cue's whole point is to point at a cluster the agent can't perceive). Use a wide
  search (e.g. up to the arena/cluster bounds). Compute world-space delta
  `(dx, dz)` = log − agentPos.
- Write row 0 = `[dx/norm, dz/norm, min(1.0, dist/64.0), 1.0]` where
  `norm = max(1.0, hypot(dx, dz))`. NOTE the sim uses **2D horizontal** unit vector
  (x,z only) and **3D Euclidean** distance (`dist[j]` over full delta,
  obs_adapter.py:195-200) — match exactly: unit = horizontal, dist = 3D.
- Default OFF → row 0 stays all-zeros = real/golden-trace behavior. Safe to overwrite
  row 0 because the peaceful arena has no hostiles (§1.2).

### 2.4 "clusters" arena (FLAG-GATED, in WorldOps)

Mirror `sim/world.py::_cluster_bases` (lines 173-202) **byte-for-byte** in the new
`resetEpisode(playerName, seed, "clusters")` branch (§2.0): same `_JavaRandom` draw
order — `ax = 58 + (nextInt(9)-4)`, `az = -48 + (nextInt(7)-3)`, `gap = 22 + nextInt(7)`,
`dir = dirs[nextInt(3)]`, then per-trunk `cx-7+5*col + (nextInt(3)-1)` /
`cz-3+6*row + (nextInt(3)-1)`, clamp `x∈[46,94]`, `z∈[-86,-14]`, dedup-nudge `x+=1`.
Then place the same `TRUNK_H=4` stacks at `Y=65..68`.

**MUST-NOT-MISS companion (no sim analog):** the clusters arena extends beyond the
current forceload + grass-floor bounds.
- Current forceload `X[32,96] Z[-64,-16]`, grass `X[48,80] Z[-64,-32]`
  (WorldOps.java:194,196).
- `_cluster_bases` can place cluster B at `z` down to roughly **-77** and `x` up to
  **~90** (ax±4, gap 22-28, plus offsets). Those `z<-64` / `x>80` trunks fall OUTSIDE
  both → `/setblock` into non-forceloaded chunks and/or no grass beneath them.
- **Fix:** when `arenaMode=="clusters"`, widen `setupTrainingScene` (or do a
  one-shot widen in the clusters reset) to forceload + grass-fill the cluster bounds,
  e.g. `X[40,100] Z[-90,-16]` (compute from the same constants used to place trunks;
  do NOT hardcode loosely). The sim has no chunk/forceload concept, so this gap is
  invisible from Python — it is the single most likely "works in sim, blank in MC" bug.

---

## 3. Parity risks (golden-trace discipline)

**3.1 Tie-break is ALREADY consistent — do NOT "fix" the obs builder.**
Java's `nearby.sort(comparingDouble(distSq))` (GathererOverlayBuilder.java:85) is
`List.sort` = TimSort = **stable**. Insertion order is `dx` outer `[-16,16)`, `dz`
inner (lines 55-56) — identical to the sim's reconstruction loop (obs_adapter.py:133-134).
A stable sort by `distSq` over that insertion order produces EXACTLY the sim's explicit
`(distSq, dx, dz)` key (obs_adapter.py:141). They already match (the sim docstring
obs_adapter.py:113-114 says so). **Adding an explicit tie-break to the Java comparator
would change nothing if done right and silently break parity if done wrong — leave it
stable-by-distSq.** The real risk is intra-Java: the decision-core MINE must index the
SAME ordered list, which §2.1's shared-method extraction guarantees.

**3.2 topmost-non-air (Java) vs topmost-log (sim) — preserved assumption, not a bug.**
Java takes the topmost NON-AIR block per column and `break`s (GathererOverlayBuilder.java:71),
then channel-matches. The sim takes the topmost LOG (obs_adapter.py:130-131; bare-trunk
arena has only logs). These coincide ONLY for bare trunks (no leaves). The arena is
leafless by construction (WorldOps places only `oak_log`; world.py header notes "NO
leaves"). Document this as a preserved invariant: if leaves are ever added, Java's
top-non-air would read the leaf (no channel match → column dropped) while sim reads the
log — divergence. Keep arenas leafless.

**3.3 Coordinate / dy convention (the parity gotcha to verify against a FRESH trace).**
- Agent settles at obs-y=65 (feet), `origin = agent.getBlockPos()` floors to by=65.
- A 4-tall trunk is `Y=65..68`. The ±3 window is `dy∈[+3..-3]` around by=65 → covers
  `Y=62..68`. The topmost log of a 65..68 trunk is `Y=68` → **dy=+3** → normalized
  `dy/8 = 0.375`.
- **STALE-COMMENT TRAP:** `obs_adapter.py` top docstring (lines 16-18) still says
  "logs at y=66, dy=+1, 0.125" — that is PRE-N21 flat-grid text. The live
  `gatherer_nearest_columns` produces dy=+3 → 0.375 for N21 trunks. **Verify the
  Java output against a freshly regenerated golden trace, NOT that docstring.** Do not
  "correct" code toward 0.125.
- A hypothetical Y=69 crown would be dy=+4 — ABOVE the ±3 window — never reported
  (WorldOps caps trunks at Y=68, so this never arises; do not exceed height 4).

**3.4 Distance metric: 2D unit vs 3D distance in the bearing cue (§2.3).** Easy to
mix up — sim uses horizontal (x,z) unit direction but 3D Euclidean distance. Match
exactly or the cue points slightly wrong and the blind hop drifts.

**3.5 Column-mine order: bottom-up.** `mine_instance` sorts the column by Y ascending
(skills.py:233). Java must mine `Y=65,66,67,68` in order. With the existing
walk+reach machinery this only affects which log is `currentTarget` first; functionally
irrelevant to the final inventory but matters if any per-tick obs is captured mid-dispatch
in a trace. Match it to be safe.

**3.6 `_JavaRandom` parity for clusters.** The sim already ports `java.util.Random`
byte-faithfully (world.py:55-93) specifically so seeds match. The Java side IS
`java.util.Random` (`epRand`, WorldOps:80) — so as long as the clusters branch uses
the SAME draw order/count as `_cluster_bases`, parity holds. Count the `nextInt`
calls: 4 setup draws (ax, az, gap, dir) THEN per-trunk 2 draws ×16 = exact match
required.

---

## 4. Morning execution checklist

Implement (READ-WRITE allowed in the morning), then the USER runs build/deploy.

1. **[implement]** Add the config holder + `Py4JEntryPoint.setDecisionCoreConfig(...)`
   + backward-compatible `resetEpisode(..., arenaMode)` overload (§2.0).
2. **[implement]** Extract `GathererOverlayBuilder.orderedNearbyColumns(...)` shared
   static method; rewire `populate` to use it (no behavior change) (§2.1 step 1).
3. **[implement]** Add the `decisionCore` branch to `HarvestSkill` (resolve-once,
   clamp k, mine pointed column bottom-up to completion) (§2.1 step 2).
4. **[implement]** Add the flag-gated bearing cue to `GathererOverlayBuilder` (§2.3).
5. **[implement]** Add the `"clusters"` branch to WorldOps reset + widen
   forceload/grass for cluster bounds (§2.4). **Do not forget the forceload widen.**
6. **[implement]** Perception mask: confirm it's a Python wrapper toggle (§2.2) — most
   likely NO Java change. Hand to `minecraft-rl-environment-specialist`.
7. **[implement]** Sanity: `ruff`/`mypy` clean on any Python touched (none expected
   for the mod itself); verify the default-OFF path is untouched by reading the diff.
8. **[USER-RESERVED — build]** `cd fabric_mod && export JAVA_HOME=/c/Users/Carte/jdk/jdk-21.0.11+10 && ./gradlew build`
   → `build/libs/aiutopia-mod-<mod_version>.jar`.
9. **[USER-RESERVED — deploy]** Copy the jar into **instance-1's `mods/`** (and stop
   the training run first — the n14..n18 probes contend for instance-1). Fabric does
   NOT hot-reload.
10. **[USER-RESERVED — restart]** Restart the instance-1 Java process so the new jar
    loads. (For a full run, all 4 instances; for golden-trace regen, instance-1 only.)
11. **[verify]** With decision-core flags ON, regenerate the golden trace from the
    LIVE env (the analog of `tests/fixtures/gatherer_obs_trace_seed1.json`) and
    confirm: (a) `g_nearest_resources[0]` matches the sim row-for-row; (b) the topmost
    log reads **dy=+3 → 0.375** (NOT 0.375≠0.125 stale comment); (c) a HARVEST with
    `target_class=k` mines column == `nearby[k]`; (d) bearing cue (if on) row 0 matches
    sim direction/distance.
12. **[verify]** With ALL flags OFF, regenerate the survival-forest trace and confirm
    it is IDENTICAL to the pre-change golden trace (the proven path is untouched).
13. **[transfer-validate]** Run the decision-core policy (trained in sim) against the
    live env on instance-1 and confirm sim↔real obs/action parity on a short rollout
    (the existing transfer-validation harness / `scripts/n*_e2e_test.py` analog).

---

## 5. Handoffs

- Perception mask (§2.2) and any wrapper-side `_normalize_raw` toggle →
  `minecraft-rl-environment-specialist`.
- Reward shaping for the clusters blind-explore (`_log_potential` PBRS, sim_env.py:94-99)
  → `reward-design-and-imitation-learning-specialist` (sim-side already; mirror only if
  the real env needs it).
- Policy training / transfer eval → `deep-rl-training-specialist`.

## 6. Quick reference — sim ↔ Java method map

| Concept | Sim (authoritative) | Java (to edit) |
|---|---|---|
| Nearest-resource ordering | `obs_adapter.py:97-142 gatherer_nearest_columns` | `GathererOverlayBuilder.java:55-102` (extract shared method) |
| Decision-core dispatch | `sim_env.py:206-223 _dispatch_decision_core` | `MotorBridge` (flag) + `HarvestSkill.start/tick` |
| Mine pointed column | `skills.py:216-239 mine_instance` | new `HarvestSkill` decisionCore branch |
| k clamp / empty-perception fail | `sim_env.py:214-222` | `HarvestSkill.start` |
| Bearing cue | `obs_adapter.py:191-200` | `GathererOverlayBuilder.populate` (flag) |
| Perception mask | `obs_adapter.py:205-220` | Python wrapper (likely NO Java change) |
| Clusters arena | `world.py:173-202 _cluster_bases` | `WorldOps.resetEpisode(...,"clusters")` + forceload widen |
| RNG | `world.py:55-93 _JavaRandom` | `WorldOps.epRand` (java.util.Random) |
