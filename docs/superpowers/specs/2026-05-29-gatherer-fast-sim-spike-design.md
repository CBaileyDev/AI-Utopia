# Gatherer Fast-Sim Spike — Design Spec

**Date:** 2026-05-29
**Status:** Draft for review
**Owner:** Carte
**Related:** `PROJECT_CONTEXT.md` §9/§16, `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`, Phase-0 fixes (`a191a2e`)

---

## 1. Why

Real Minecraft is a single-threaded simulation. Even at 12 Fabric instances we measured **~0.9 env-steps/sec aggregate**; the RTX 4080 sits idle. Rocket-League-class training (RocketSim) and Minecraft-domain JAX environments (Craftax: **1B env-steps/hour on a single RTX 4090**) run at **10⁴–10⁶ steps/sec** by never running the game — they simulate the *task*, vectorize across the GPU, and saturate the hardware.

AI-Utopia is unusually well-positioned to adopt this: the obs/action/reward layer is already a **backend-agnostic seam** —
`env/spaces.build_role_observation_space`, `build_role_action_space`, and `env/reward.compute_reward_stage_1` are pure contracts the env wrapper consumes, and `train/scenario_runner` + `M1EvalScenarioCallback` already validate a policy in *real* MC. A fast-sim backend is therefore **additive, not a rewrite**: train in sim, validate/deploy in real MC.

We start with the **gatherer** because its contract is frozen, M1B has a passing real-MC gate proof (`scripts/p0_gate_proof.py`: 64 logs in 64 steps), and its task is the simplest in the roster. It is the cheapest way to **measure the sim→real transfer cost** before committing to per-role sim modules.

## 2. Goal & success criterion

- **Primary:** a policy trained **entirely in the sim** clears the **real-MC M1B gate** — ≥80% success on "collect 64 oak_log within 1000 env-steps" over 3 consecutive `M1EvalScenarioCallback` evals — with **zero policy-code changes** between sim and real backends.
- **Secondary (the real output of the spike):** a measured **sim→real fidelity-debugging cost** — how many fidelity mismatches surface and how long they take to close. This number is what makes every future role's estimate honest; right now it's a guess.

Explicit **non-goals:** other roles; JAX end-to-end (cheap path first); hitting a specific FPS. Success is *transfer*, not throughput.

## 3. Scope

- **In:** the gatherer M1B task — flat arena, 64 reachable oak_log (matching `WorldOps.resetEpisode`), the 6 macro-skills (navigate / harvest / deposit_chest / search / wait), single agent.
- **Out:** builder/farmer/defender/explorer sims; multi-agent / CTDE; procedural world generation; crafting; the JAX end-to-end port (Section 8 Phase D, conditional).

## 4. Architecture

```
            ┌──────────────────────────────────────────────┐
            │  PyTorch RLModule policy (UNCHANGED)           │
            └───────────────┬──────────────────────────────┘
                            │  obs / action  (the contract)
        ┌───────────────────┴───────────────────┐
        │  spaces.py  +  reward.py  (source of truth) │
        └───────┬───────────────────────┬───────────┘
                │                        │
   ┌────────────▼─────────┐   ┌──────────▼───────────────┐
   │  SimBackend (NEW)     │   │  RealMCBackend (existing) │
   │  fast, vectorized     │   │  FabricBridge / Py4J      │
   │  TRAIN here           │   │  VALIDATE + DEPLOY here    │
   └───────────────────────┘   └───────────────────────────┘
```

The contract is the single source of truth; both backends must satisfy it **identically**. Train on `SimBackend`; the *same* RLModule drops into `RealMCBackend` via `run_scenario` for transfer validation and eventual deployment.

**Cheap path first (this spike):** a vectorized NumPy/Python sim wrapped as a Gym/PettingZoo vector env feeding the **current RLlib/PyTorch PPO** stack. Throughput target ~5k–50k steps/s (Ray-connector bound) — still ~10⁴× over real-MC, and it reuses the existing policy and training code.

**Scale path (later, conditional):** JAX end-to-end — `vmap` sim + Flax policy + PureJaxRL-style loop, with a weight-bridge back to PyTorch for real-MC deploy — for the 10⁵–10⁶ steps/s regime. Only pursued if the cheap path's throughput proves limiting *after* transfer is proven.

## 5. Components (isolated units)

1. **`SimWorld`** — vectorized world state: agent `(x,y,z)`, the 64-oak_log flat grid (mirrors `WorldOps.resetEpisode`: 8×8 grid, y=66, seeded jitter, in-bounds, no overlap/spawn-tile), inventory vector. Pure functions: `reset(seed)`, `apply_effect(state, effect)`. No rendering, no I/O, deterministic.
2. **`SkillModel`** — translates a macro-action `(skill_type, target_class, spatial_param, scalar_param)` into its multi-step world effect, matching the Java skills exactly: HARVEST walks to the nearest matching block at `WALK_PER_TICK=0.215`, breaks within `REACH_RADIUS=4.5` (cap = `round(scalar*64)`), with `findNearest` ground-preference `dy∈[-2,+1]`; NAVIGATE straight-lines toward `origin + spatial*[32,8,32]` (arrival radius 1.0); DEPOSIT/SEARCH/WAIT. **Move-until-AABB-collision** (no climb/jump).
3. **`ObsAdapter`** — builds the obs dict **byte-faithfully** from sim state: `g_resource_grid` (32×32×6, channel ids 0..5), `g_nearest_resources` (top-8, `dx/16`, `dy/8`), `position`, `inv_slot_*`, the in-range `action_mask` booleans, and the constant embeds (SHA-256 `agent_uuid_embed`, 512-d goal stub, zeroed comm). Parity target: `obs/GathererOverlayBuilder.java` + `env/wrapper.py`.
4. **`RewardAdapter`** — vectorized `compute_reward_stage_1`: the **oak_log-only single attractor** (P0 fix #2), PBRS potential, time/death/clip/exploit terms. Reuse the pure reward logic; do not re-derive constants.
5. **`SimEnv`** — wraps 1–4 as a vector env for RLlib (cheap path); same env_config contract keys as `AiUtopiaPettingZooEnv`.
6. **`TransferValidator`** — reuses `scenario_runner.run_scenario` / `M1EvalScenarioCallback` to run the sim-trained RLModule in **real MC** every N iters. The 80%-over-3-consecutive gate **is** the transfer test.

## 6. Fidelity risks (the cost tail — each is a transfer test, not an afterthought)

These are where the estimate's variance lives. A mismatch produces **no crash** — just a policy that scores well in sim and ~0% on the real gate.

- **Navigation / collision geometry** — must reproduce move-until-collision + `REACH_RADIUS=4.5`. The N16 float-precision attractor lived exactly here. #1 silent transfer-failure mode.
- **Obs normalization** — `g_resource_grid` channel ids, `dx/16` / `dy/8` scaling, the SHA-256 uuid embed, the 512-d goal stub must match `spaces.py`/Java exactly.
- **Skill semantics** — HARVEST auto-walk, `cap`, `STALL_TICK_BUDGET`, 1-tick break + `offerOrDrop`, item-pickup timing.
- **Reward parity** — including root-causing the **−633 reward outlier** (PBRS/exploit/clip) seen in v21; the same pure function fires in sim, so it must be understood, not masked.
- **Don't over-fit the sim to a real-env bug** — e.g. the cold-start spawn race exists only in real MC; the sim must not silently "fix" problems the real env still has, or transfer numbers lie.

## 7. Testing

- **State-parity tests** — drive identical action sequences through the sim and recorded real-MC traces; assert each obs field matches within tolerance.
- **Reward-parity unit tests** — `RewardAdapter` output == `compute_reward_stage_1` on identical inventories/transitions.
- **The transfer test** — a sim-trained policy ≥80% on the real-MC gate over 3 consecutive evals (the Section 2 success criterion).

## 8. Plan (phased)

- **A — Sim core:** `SimWorld` + `SkillModel` + `ObsAdapter` + `RewardAdapter` + parity tests.
- **B — Train in sim:** `SimEnv` + train the gatherer on the cheap path; confirm fast convergence and a clean upward slope.
- **C — Transfer:** validate the sim-trained policy in real MC; close fidelity gaps until the gate clears. **Record the cost** (Section 2 secondary goal).
- **D — Scale (conditional):** JAX end-to-end port, only if cheap-path throughput limits after transfer is proven.

Estimate: ~1.5–3 weeks for A+B (a fast, good-in-sim gatherer); the fidelity tail in C is the dominant variance (days to a few weeks). Total to a transfer-validated gatherer sim: **~2–5 weeks**.

---

## Appendix A — Role roster & naming convention (framework for M2+)

The spike validates the methodology; this is the framework it scales into.

**Three distinct things (cost ladder):**
- **Role** = a distinct RL policy: own obs overlay + reward + skills + CTDE slot → **+1 sim module to build & fidelity-validate**. (High cost.)
- **Persona** = the *same* policy run with a different goal (`goal_embedding`/`target_class` already condition the policy) → a planner-assigned goal + display name. (~Free.)
- **Skill** = a new parameterized action on an existing role. (Medium cost: +1 skill in sim & real.)

**Decisions:**
- **RL roles (policies = sim modules):** `gatherer`, `builder`, `farmer`, `defender`. **`explorer` = designated 5th** — its reward is genuinely distinct (information gain / map coverage / discovery, not inventory). Its inter-agent "reporting" goes through the **shared resource-map + LLM planner**, NOT learned agent-to-agent communication (emergent comm has brutal cross-agent credit assignment; the comm channel is masked in single-agent training per fix #4). Its sim module needs procedural resource placement + fog-of-war — more than the gatherer's fixed grid.
- **Personas (free):** Lumberjack / Miner / Quarryman = goal-conditioned `gatherer`; **Soldier = `defender`'s display name**.
- **Skills (the real functional gap):** `craft` / `smelt` / `repair` (currently nothing crafts, yet the reward table wants `furnace`/`iron_pickaxe`); `mine_vein` / `place_torch` for deep mining. Add to `gatherer`/`builder`, not as roles.
- **Naming is two-layer:** canonical `RoleId` (`gatherer`/`builder`/`farmer`/`defender`/`explorer`) is a **stable contract** (planner JSON, SQLite roles, `role_one_hot`) — never churn it; **display names** (Soldier, Lumberjack, Scout…) are free flavor.
- **Do NOT add:** self-assigning emergent specialists (out of scope — roles fixed at spawn), duplicate combat roles (CTDE lazy-agent pain), early redstone engineer (huge skill complexity, niche), mayor/diplomat (that's the planner/chat layer, not embodied).
