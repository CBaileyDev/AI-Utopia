# M2 Brainstorm — Builder Role, Multi-Agent CTDE, Cross-Policy Sharing

**Status:** brainstorming (design exploration; NOT yet an implementation plan)
**Date:** 2026-05-27
**Author:** system-architect
**Predecessors:** `M1A_PIPELINE_PLAN.md` (tagged `m1a-verified`), `M1B_TRAINING_PLAN.md` (code-complete, empirical run in flight at time of writing)
**Spec source:** `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md` (sections 4.1–4.6, 5.1–5.3, 5.7, 5.9, 7.1, 7.2)
**Self-review:** spec-coverage matrix at end (§J), contradiction/scope check at §K.

---

## 0. M1B convergence dependency (gating risk)

The M1B PPO run is in flight as this document is being written. M2 implementation cannot start until at least M1B's two narrow gates pass:

- `eval_m1_oak_log_success_rate ≥ 0.80` over 3 consecutive evaluations on the `collect-64-oak-log` scenario.
- Promotion checks pass: determinism (argmax_div < 0.05, L2 < 0.1) and `aiutopia promote-weights promote` writes `gatherer/v1/`.

This is not a soft dependency. The whole reason M2 exists in the spec's phase-2 sequence is that a *converged* per-role policy is the precondition for cooperative training; trying to train two roles together when one of them is broken would conflate "this role can't learn the task alone" with "the two roles can't coordinate", and the resulting debug surface explodes. Empirical convergence of M1B is therefore the gate, not a checkpoint to wave at.

Three possible outcomes, each with a different M2 entry posture:

| M1B outcome | M2 entry posture |
|---|---|
| Converges first try | M2 begins as designed below. Gatherer policy from `gatherer/v1/` becomes the M2 stage-1 starting checkpoint (warm start for the gatherer leg of multi-agent training). |
| Converges after ≤ 3 hyperparameter sweeps | M2 begins with **+1 day buffer** on the schedule in §G; M1B post-mortem learnings (likely entropy/clip/lr) become Stage-1 defaults for the builder solo phase. |
| Does not converge in this session | M2 is **paused** until M1B remediation lands. M2 scope inherits whatever M1B remediation requires (most likely candidates: canopy collision unstick / `EntityPlayerActionPack` integration, reward shaping for navigation, longer training). Do not start M2 from a known-broken M1B. |

This is the single largest dependency for M2 and must be honoured before any of the work below begins.

---

## A. M2 milestone gate — what does "M2 verified" mean?

### A.1 Two gates, not one (spec §5.8 phase 2)

The spec divides M2 into two halves; each gets its own gate.

**M2-α (first half — builder solo, IPPO single-policy):**

> Builder policy, trained alone (no gatherer present, materials pre-supplied in inventory), completes a randomly-selected blueprint from the standard 5-structure set (wall, gate, watchtower, well, storehouse) at **70% blueprint_match** within 2000 env steps, **3 consecutive evals**, no `place→break→place` cycling on the same cell more than 2x per episode.

This is the direct spec quote (§5.8 phase-2 M2-first-half), tightened with the anti-cycling clause because place-cycle is the most likely reward-hack the builder will discover and the symbolic blueprint signal alone won't catch it.

**M2-β (second half — gatherer + builder cooperative, MAPPO/CTDE):**

> Two-agent cooperative episode: gatherer + builder spawned in fresh world. Builder receives blueprint for a small structure. Gatherer + builder cooperatively complete the structure within 2000 env steps, **60% success rate** over 3 consecutive evals, with **zero BULK_FARMING penalty firings** in the gate evals.

The spec floor is 50% (§5.8 phase-2 M2-second-half: "cooperative structure_built_correctly > 50%"). I recommend tightening to 60% on the gate, but holding 50% as the rollback threshold (below 50% triggers M2-β redesign). The "zero BULK_FARMING firings" clause exists because passing the bar by gaming the cross-agent transfer detector would be a false positive.

**Why not the user's "80% on 3x3 wood shelter from scratch in 2000 env steps":**
- The user's suggested bar is closer to M2-β at the *aggressive* end. With M1B convergence still unconfirmed and a fresh role + multi-agent dynamics, 80% on first-try training is implausible. Recommend the spec floor + tightening (60%) and treat 80% as a stretch goal that maps to M3 entry, not M2 exit.
- "From scratch" implies the gatherer chops the wood, hands off, and the builder lays the structure. This is exactly the cross-agent handoff that the BULK_FARMING detector watches; tying the gate to a workflow the exploit-detector polices means the gate is partially testing the detector's false-positive rate, which is undesirable.
- The 2000-step budget is reasonable but tight. A 3×3 shelter is ~17 cells (4 walls × 4 cells, minus overlaps, plus roof and door); even at 10 env steps per `place_block` and another ~50 env steps per `harvest` × 17 logs, that's 50×17 + 10×17 = 1020 env steps for the *optimal* trajectory before any movement. Realistic execution with imperfect navigation will hit 1500+. The 2000-step budget therefore gates on near-optimal play, which is a stretch for M2's first cooperative training.

The recommended bar (cooperative 60%) is intentionally not "complete the full shelter": the eval scenario `m2_coop_shelter_2agent` counts the run a success if `b_progress >= 0.8` at episode end (i.e., 80% of the blueprint's correct-cell target is achieved). Partial-credit gating is honest given that the spec's `blueprint_match` reward is itself partial-credit; binary completion would penalise nearly-complete runs that prove cooperative coordination works.

### A.2 Recommended eval scenario set

Three fixed scenarios + one randomized (mirrors §5.10 promotion checklist item 2 — "3 fixed seeds + 1 randomized"):

| Scenario | Roles | Inventory pre-supply | Blueprint | Pass bar |
|---|---|---|---|---|
| `m2_blueprint_wall_solo` | builder | 64 oak_log, 64 oak_planks | 1×6×3 wall | 70% in 1500 env steps |
| `m2_blueprint_well_solo` | builder | 64 cobblestone, 16 oak_planks | 3×3 well structure | 70% in 2000 env steps |
| `m2_coop_shelter_2agent` | gatherer + builder | empty | 3×3×3 wood shelter | 60% in 2000 env steps |
| `m2_coop_random` | gatherer + builder | empty | random from {wall, gate, watchtower} | 50% in 2500 env steps |

Promotion CLI extension: `aiutopia promote-weights promote --role builder` reuses the M1B promotion path (already built); `aiutopia promote-weights promote --multi-policy gatherer,builder` is **new** and emits a single combined `policy_deployments` row referencing both checkpoint paths.

### A.3 Non-goals for the M2 gate

- LLM planner integration (Stage-3 reward) — M5.
- Farmer / defender — M3, M4.
- Pixel-patch evaluation gate — if pixel encoder is deferred (see §B.1, my recommendation), the gate does **not** require pixel-based blueprint matching; the symbolic `b_voxel_grid` does the work.
- Continuous fine-tuning across both halves — M2-α produces a builder checkpoint; M2-β warm-starts from it. No requirement that M2-β remain stable on the M2-α solo eval after MAPPO fine-tuning.

---

## B. Open architectural decisions

### B.1 Pixel-patch encoder: defer or ship?

**Status:** spec §4.4 mandates pixel patches at M2+. Deferral is a **spec deviation requiring user approval**, not a free option. This is framed as such, not as a neutral comparison.

**Options:**

| Option | What it is | Pros | Cons |
|---|---|---|---|
| (a) Iris/Sodium offscreen framebuffer | Render the 64×64 task-area RGB via Fabric/Iris's offscreen FBO API, ~5 ms per builder per skill invocation | Spec-compliant. Visually-grounded blueprint matching matches human intuition. | Iris/Sodium mod compatibility on dedicated server is *unknown* — neither mod targets server-side rendering normally. Likely requires running a headless GLFW context on the Java process; on Windows this means EGL or `--no-render` flags that contradict Iris's design. May or may not work in our environment. Highest engineering risk. |
| (b) Software raycaster | Read chunk data in Java, raycast 64×64 from the configured camera pose, texture-lookup per ray, emit RGB | Works in any environment. Spec-compliant. No mod dependency. | ~50 ms per invocation per spec §4.4. With builder placing ~10 blocks per episode, that's 500 ms wall-clock per episode — non-trivial on a 30K-episode training run. Block-texture mapping table is moderately tedious to build (but bounded). |
| (c) Defer pixels to M3, symbolic only for M2 | Builder's `b_voxel_grid` + `b_blueprint_grid` + `b_blueprint_status` (11×11×11 each) carry the structural signal. Pixel encoder remains None; `pixel_feat = zeros(64)`. Add to milestone gate language: "pixel encoder deferred to M3 pending plateau on M2-α/β symbolic-only eval." | Lowest engineering risk by far. Symbolic blueprint signal is dense (~1331 cells × 3 fields = ~4K values) and is the *primary* blueprint signal anyway. Pixel patch is mostly redundant for the M2 standard 5-structure set. Saves 1-2 days of engineering. | Spec deviation. Requires user sign-off (see §H). If M2-α plateaus below 70% on symbolic alone, we're back to (a) or (b) under deadline pressure. |

**Recommendation: (c) — defer pixels, symbolic only.** Reasoning:
1. The `b_voxel_grid + b_blueprint_grid + b_blueprint_status` tensors are the primary blueprint signal per spec §4.4; pixels are an *additional* channel the spec wanted for visual coherence on irregular structures. The M2 standard 5-structure set (wall/gate/watchtower/well/storehouse) is regular cuboid geometry — symbolic carries it.
2. Engineering risk of (a) is unknown and unbounded; (b) is bounded but eats wall-clock at training time and adds a software-rendering codepath nobody else maintains.
3. The plateau-gated path is exactly what spec §5.9 says to do for the defender's panoramic patches at M4. Reusing that pattern for builder is consistent.
4. **Falsifiable:** if M2-α blueprint_match plateaus < 65% over 5M env steps on symbolic alone, treat as evidence pixel encoder is needed and reopen this decision. The 5M-env-step floor is approximately one day of wall-clock at our throughput; not a cheap experiment, but the only honest threshold.

**Operational consequences of (c):**
- The `b_pixel_patch` key is omitted from the obs Dict during M2 (not zero-padded — *absent*). RLlib's ConnectorV2 builds the obs space from the Dict definition, so absence vs zero-padding has different implications: absence means policy never sees the field; zero-padding means policy sees a constant signal it must learn to ignore. Absence is strictly better when the field will not carry signal — no wasted parameters, no false-positive-correlation risk.
- `AiUtopiaRoleRLModule.pixel_encoder` remains `None`; the existing zero-vector fallback (`pixel_feat = zeros(B*T, 64)`) preserves the 448-d fused feature contract. No backbone reshape required.
- M3 reactivation path: define `b_pixel_patch` key, set `pixel_encoder = build_pixel_encoder("builder", ...)`. The fused-feature width does not change because the zero-fallback was already accounting for the 64-d slot. Backwards-compatible per design.

**Requires user approval.** This is the spec deviation flagged for §H below.

### B.2 Cross-policy weight sharing — `CoreEncoder` / `SharedBackbone` / `CTDECritic`

**The real blocker is not what M1B plan v2 said.** Plan v2 (§3081) wrote "no clean RLlib API for it on the new stack as of 2.40-2.50 (`additional_module_specs` does not exist)". That's literally true but it's not the full story.

Parameter sharing in PyTorch is trivial — subclass `MultiRLModule`, override `setup()` to construct shared `nn.Module` instances once, then assign the same instance to each child's `.core_encoder` / `.shared_backbone` attribute. Gradients flow through automatically because they're the same `nn.Parameter` objects.

The **actual** problem in Ray's new API stack:

> The Learner builds one optimizer per `RLModuleSpec` under `MultiRLModuleSpec` (via `configure_optimizers_for_module`). Shared `nn.Parameter` objects therefore appear in **two optimizer parameter groups**. Each PPO step calls `.step()` on both optimizers. Shared params receive **two gradient updates per step** — effectively 2× LR on the shared trunk.

This is the deciding factor and it changes the calculus.

**Options:**

| Option | Mechanism | Pros | Cons |
|---|---|---|---|
| (a) Own copy per role | Each policy has its own `CoreEncoder` + `SharedBackbone` + `CTDECritic` as `nn.Module` attributes. Identical architecture, independent weights. ~12M params total (6M × 2). | Zero optimizer-doubling risk. Zero RLlib internals reverse-engineering. Same as M1B's per-policy structure, just instantiated twice. The CTDE critic naturally diverges per-role (each role sees the world through its own decoder) — arguably the *correct* behaviour, not a regression. | Loses the spec §4.3 "parameter-shared" rationale ("roles benefit from common world dynamics"). 12M params is trivial; the cost is conceptual purity, not memory. |
| (b) Shared submodules via custom `MultiRLModule` | Subclass `MultiRLModule`, share `nn.Module` instances. Mitigate optimizer-doubling via one of: (i) freeze shared modules on builder policy, only train via gatherer; (ii) halve LR on shared params via `algorithm_config_overrides_per_module`; (iii) custom Learner that detects duplicated `nn.Parameter` objects and dedupes optimizer groups. | Spec-compliant. Genuine parameter sharing — gradients from both roles update one trunk. | Requires reading Ray's Learner source. Mitigation (i) makes the gatherer carry the entire shared-trunk gradient signal, asymmetric. Mitigation (ii) is fragile — 2× LR isn't exactly 2× because PPO has different per-policy minibatch counts. Mitigation (iii) is the right one but is custom Ray internals work, ~2 days of plumbing and tests. |
| (c) Supervised pre-train shared encoder, then freeze | Pre-train `CoreEncoder` + `SharedBackbone` on a behavioural-cloning dataset (or on the M1A trajectory traces), then freeze both before M2 training starts. Each role gets its own actor + critic head, frozen shared trunk. | No optimizer-doubling problem (no training of shared params during M2). Could improve sample efficiency by giving both policies a pre-trained obs-to-feature mapping. | Requires a BC dataset we don't have. Generating one means a separate research project. Out of scope for M2 timeline. |

**Recommendation: (a) — own copy per role for M2.** Reasoning:
1. Optimizer-doubling rathole is real and (b) requires reading Ray internals that may change between 2.40 and 2.55. Doing this under M2 deadline pressure is a recipe for silent training-doesn't-converge bugs that look like reward hacking but are actually broken gradients.
2. 12M params is nothing. We have headroom for 100M+.
3. The CTDE critic in particular *should* differ between gatherer and builder — gatherer's "world" is biome+resource distribution, builder's "world" is voxel state. A shared encoder might actively hurt sample efficiency in M2 even if it's spec-compliant.
4. Defer (b) to M3 when adding the farmer makes duplication actually painful (3 copies of CoreEncoder = ~9M params dedicated to shared concepts). At M3 we'll also have empirical data from M2 on whether the gatherer/builder encoders end up looking similar; that data tells us if sharing is even worth the optimizer-doubling fix.
5. **Falsifiable:** if M2 sample efficiency is more than 3x worse than M1B per-million-env-steps on the gatherer leg alone (suggesting massive negative interference from independent encoders), reopen this decision. This is unlikely — M1B trains a gatherer on the same architecture; M2 just runs two of them in parallel.

This is a **spec deviation** from §4.3's "parameter-shared across all 4 roles" language, but it's an *implementation* deviation rather than a *design* deviation: the spec asserts the *intent* (shared inductive bias for world dynamics), and (a) deviates from the *mechanism* (parameter identity) while preserving the architecture and re-opening sharing in M3. Less invasive than B.1; does not require user approval — call it explicitly in §H but proceed.

**Why the optimizer-doubling concern is concrete, not theoretical:** Ray 2.40+ Learner code (`ray/rllib/core/learner/learner.py`) iterates `self._module_specs.keys()` when calling `configure_optimizers_for_module(module_id, config)`. Each call returns one or more `Optimizer` objects bound to a parameter list. The default implementation in `TorchLearner._make_optimizer` returns `torch.optim.Adam(self._module[module_id].parameters(), lr=...)`. If `self._module["gatherer_policy"].core_encoder` and `self._module["builder_policy"].core_encoder` are the *same `nn.Module`*, calling `.parameters()` on each child policy returns overlapping parameter sets. The two Adam optimizers each accumulate first/second moment estimates on the shared params and each call `.step()` per training iteration — the params are updated twice per iteration.

Adam's bias-corrected update formula means the effective LR is *not* exactly 2× single-policy LR — it depends on the relative gradient magnitudes from each policy. In a symmetric case (gatherer and builder produce similar gradient norms on shared trunk), it's roughly 2×. In an asymmetric case (one policy has 10× larger gradients), it can be 11× the larger optimizer's effective LR. This is silently wrong without throwing errors, which is the worst class of bug: training appears to proceed, but the shared trunk diverges or oscillates.

Mitigation (i) above — freeze shared on one policy — eliminates this but at the cost of "shared" becoming a one-way street: the gatherer's gradients update the trunk, the builder rides along. This contradicts the symmetric-sharing intent of §4.3.

Mitigation (iii) — custom Learner — is the principled fix but is non-trivial: it requires overriding `_make_optimizer` to detect parameter identity across modules (e.g., by `id(param)` set-membership) and de-duplicate. Maintainable but not a 4-hour task.

Given M2 timeline pressure and the falsifiable reopen-in-M3 clause, option (a) is the strictly safer recommendation.

### B.3 Reward stage transition (Stage-1 → Stage-2)

**The spec already answers this.** §5.1 specifies:

```
curriculum_decay = max(0.7, 1.0 - 0.3 * curriculum_step / 10_000_000)
```

Linear decay from 1.0 → 0.7 over 10M env steps, then floor. This is the spec answer and is not subject to re-litigation in M2.

**Implementation choices left to make:**

1. **When does the M2-α → M2-β switch happen?** Three sub-options:
   - (i) Hard switch at iteration N (where N = "M2-α gate passed for 3 evals"). After switch, Stage-1 reward is fully replaced by Stage-2; curriculum_step counter resets to 0 at switch time.
   - (ii) Hybrid pre-exposure: during late M2-α, env wrapper emits Stage-2 reward components alongside Stage-1 with weight 0; agents see the new signal channels with no behavioural change. At switch, weight ramps from 0 to 1 over 1M env steps. Mirrors the §5.1 Stage-2 → Stage-3 transition pattern.
   - (iii) No switch — start in Stage-2 from M2-β day one, no M2-α solo phase. (Skips spec §5.8 phase-2 first half.)

   **Recommendation:** (ii) hybrid pre-exposure. Costs almost nothing (env wrapper carries an inactive code path) and avoids the cliff at switch time that (i) creates. (iii) is rejected because M2-α exists in the spec as a deliberate de-risking phase.

2. **Curriculum step counter reset behaviour:** keep counter monotonic across M2-α → M2-β, OR reset at switch. Spec is silent. Recommend **monotonic** (don't reset) — curriculum is *per training run*, not per phase. Resetting would let the role-specific weight rise back to 1.0 at the moment cooperation matters most.

### B.4 Builder action space

**The spec already answers this.** §4.2 specifies:

```
Builder skills: navigate, place_block, fetch_materials, follow_blueprint,
                level_terrain, repair_damaged_block, wait, noop_broadcast
                (8 skills total)

Action Dict header: same as gatherer
  skill_type:       Discrete(8)
  target_class:     Discrete(N_TARGET_CLASSES=64)
  spatial_param:    Box(-1, 1, (3,), f32)   # relative offset
  scalar_param:     Box(0, 1, (1,), f32)    # block_id_normalized or quantity
  comm_payload:     Box(-1, 1, (128,), f32)
  should_broadcast: Discrete(2)
  comm_target_mask: MultiBinary(4)
```

The user's enumerated options (a)/(b)/(c) map to:
- (a) "discrete skills (PLACE_LOG, PLACE_PLANK, BUILD_BLUEPRINT, ...)" — *not* spec. Spec uses generic `place_block` with the specific block coming from `scalar_param` (encoded block_id) and `target_class` (target context).
- (b) "continuous placement coords + block_id" — partially spec. Spec uses `spatial_param` for *relative* offset from agent (normalized), not absolute coords. Block id is in `scalar_param`.
- (c) "discrete skill + block_id + relative position" — this is the spec answer.

**Decision: ratify the spec.** Implementation details left to flesh out:

- `target_class` encoding for builder: index into the active blueprint's missing-cell list (top-64 nearest missing cells, sorted by Manhattan distance from agent). On no-blueprint-active, encodes generic targets (chest_id, agent_id, ...) per gatherer convention.
- `scalar_param[0]` for `place_block`: block_id_normalized = block_id / N_BLOCK_TYPES; motor module maps back to MC block id. (For the M2 standard 5-structure set: N_BLOCK_TYPES = ~12 block ids; oak_log, oak_planks, cobblestone, stone, torch, oak_door, glass_pane, ladder, fence, chest, ...).
- `spatial_param` for `place_block`: relative offset from agent position to placement cell, clamped to [-1, 1] per axis (motor module multiplies by reach distance ~4.5 blocks).
- Action mask cascade (extends §4.5):
  - `b_blueprint_status == 0 (void everywhere)` → mask out `place_block`, `follow_blueprint`, `repair_damaged_block`.
  - `b_materials_needed.sum() > 0` → mask out `place_block` (must fetch first).
  - No chest in range → mask out target_class slots for `fetch_materials`.
  - Standard cascade rules apply (skill_type masked if all target_per_skill slots zero).

---

## C. Builder role specification

### C.1 Observation space

Composed as **universal core + builder overlay** per §4.1.

**Builder overlay (`b_*` prefix, §4.1 spec):**

```python
"b_voxel_grid":        MultiDiscrete([N_BLOCKS]*(11*11*11))  # 1331 cells
"b_blueprint_grid":    MultiDiscrete([N_BLOCKS]*(11*11*11))  # target state
"b_blueprint_status":  MultiDiscrete([4]*(11*11*11))         # 0=void,1=correct,2=missing,3=wrong
"b_materials_needed":  Box(0, 512, (N_BLOCK_TYPES,))         # ~12 types for M2
"b_progress":          Box(0, 1, (1,), f32)                  # fraction of cells correct
# DEFERRED to M3 pending plateau (see §B.1):
# "b_pixel_patch":     Box(0, 255, (64, 64, 3), u8)
```

**Stage flag:** if `pixel_encoder = None`, omit `b_pixel_patch` key entirely from the obs Dict to avoid silent-zero noise through the encoder. RLModule attribute-presence check guards.

**Voxel grid centering:** 11×11×11 cells centered on the builder's current position, axis-aligned to world. Empty cells (air) carry block id 0; out-of-loaded-chunk cells carry sentinel id N_BLOCKS-1 ("unknown"). Motor module emits both; Java-side guarantees no chunk-load latency stalls (waits up to 100ms per spec §4.6, then emits with sentinel).

**N_BLOCKS sizing:** ~512 to give room for any MC block id; sparse one-hot via embedding lookup in BuilderRoleEncoder, not literal one-hot.

**Encoder shape sketch for BuilderRoleEncoder (T3):**

```
b_voxel_grid (11,11,11) → Embedding(N_BLOCKS, 8) → (11,11,11,8)
                        → permute (8,11,11,11)
                        → Conv3d(8→16, k=3, pad=1) → ReLU
                        → Conv3d(16→32, k=3, stride=2, pad=1) → ReLU   # (32,6,6,6)
                        → Conv3d(32→32, k=3, stride=2, pad=1) → ReLU   # (32,3,3,3)
                        → Flatten → Linear(864→96) → ReLU
                                                                       │
b_blueprint_grid (11,11,11) → Embedding(N_BLOCKS, 8, weight-shared) ──┤  concat
                            → (parallel 3D-CNN, separate conv weights)│
                            → Flatten → Linear(864→96) → ReLU         │
                                                                       ▼
b_blueprint_status (11,11,11) → Embedding(4, 4) → flatten → Linear → 32 ─┐
b_materials_needed (N_BLOCK_TYPES~12) → Linear(12→16) → ReLU             │
b_progress (1,) → identity                                                 │
                                                                           ▼
                                                          concat → MLP → 128-d role_feat
```

The voxel embedding is shared between `b_voxel_grid` and `b_blueprint_grid` (same block-id semantics); the 3D conv weights are *not* shared (current world vs. target world are functionally different inputs and benefit from independent filters). Parameter budget for BuilderRoleEncoder: ~600K params, comparable to GathererRoleEncoder.

### C.2 Action space

Per §B.4 — ratifies spec §4.2. 8 skills, same Dict header as gatherer with role-specific masking.

### C.3 Reward shaping

**Stage-1 primary signal (M2-α):**

Per spec §5.4:

```
r_primary = Σ_cell delta(b_blueprint_status) with weights:
    void → correct:    +1.0
    missing → correct: +1.0
    wrong → correct:   +0.5
    correct → wrong:   -2.0   (penalty for breaking own progress)
   plus:
    +5.0 on blueprint complete (b_progress == 1.0)
```

The `correct → wrong` asymmetry is critical — it's the primary defense against `place→break→place` cycling for reward gaming. Without it, the agent learns to place-and-break repeatedly. The 2:1 ratio (penalty vs. progress) makes any cycle net-negative.

**PBRS (Φ): `tech_tree_potential(inventory, "builder")`** — already wired in `aiutopia.env.reward` with builder caps (`oak_log:128, oak_planks:512, cobblestone:512, ...`). Reuse as-is; no M2 change.

**Stage-2 objective set (M2-β onwards), spec §5.4:**

```
blueprint_match:      0.60   # how close to target
material_efficiency:  0.20   # blocks placed / blocks consumed (penalizes waste)
structure_integrity:  0.20   # gravity-correctness, no floating blocks
```

Implementation notes for the three objectives:

- `blueprint_match`: `r_primary` divided by max blueprint reward (per-blueprint cap), normalized to [0,1].
- `material_efficiency`: rolling 100-tick window: `(# cells correct in window) / (# place_block invocations in window)`. If denominator is 0, return 1.0 (no waste). Smooth ramp.
- `structure_integrity`: count blocks in `b_voxel_grid` that match `b_blueprint_grid` AND have at least one adjacent solid block below them (gravity stability proxy). Normalized by total non-void blueprint cells.

**Penalties (universal, unchanged from M1B):** death (10.0), time (0.001/tick), exploit (per §E), clip (0.05/axis).

### C.4 Stub planner blueprint emission

For M2 (pre-LLM-planner), blueprints come from a hand-coded "stub planner" that emits one of 5 standard structures (wall, gate, watchtower, well, storehouse) on episode start. Emission protocol:

- Plain JSON to `LlmPlanOutput` adapter (§3.1) with `village_targets = null` (Stage-1/2 sentinel).
- Blueprint dict: `{"cells": [(x, y, z, block_id), ...], "name": "wall"}`.
- Env wrapper materializes to `b_blueprint_grid` + `b_blueprint_status` + `b_materials_needed`.
- Five fixed blueprints lives at `src/aiutopia/planner/stub_blueprints.py`.
- Selection per episode: round-robin in M2-α; uniform random in M2-β.

---

## D. Multi-agent CTDE wiring

### D.1 Episode topology — fact, not option

Per spec §4.6 (per-tick RL loop) and PettingZoo `ParallelEnv` semantics:

> Both agents (gatherer + builder) share one episode, step in lockstep, joint reset. There is exactly one Minecraft world per episode; the world's tick is shared.

This is not a design choice — it's how PettingZoo `ParallelEnv` works and how the Fabric-side world serves a shared tick state. The wrapper emits obs simultaneously for both agents on each env step, accepts actions simultaneously, and computes per-agent rewards.

Episode termination: terminates when **any** of:
1. Both agents `terminated == True` (death or skill-completion-of-final-blueprint).
2. `tick_in_episode >= max_episode_ticks` (default 12000 for M2 cooperative — matches M1B).
3. Stub planner blueprint already complete on episode start (defensive — shouldn't happen).

If one agent dies but the other survives, the dead agent's slot emits zero-obs (per spec §4.5 mask collapse semantics) and a final `r_death = -10.0` for that agent; the surviving agent continues. The Fabric-side respawns the dead agent at village center within 5 ticks (preserves identity per spec §3.4).

### D.2 `policy_mapping_fn` change

M1B (single-policy):
```python
def _policy_mapping_fn(agent_id, episode=None, **kwargs):
    return "gatherer_policy"
```

M2 (two-policy):
```python
def _policy_mapping_fn(agent_id, episode=None, **kwargs):
    # agent_id format: "gatherer_0", "builder_0" (per AiUtopiaPettingZooEnv convention)
    role = agent_id.split("_")[0]
    return f"{role}_policy"
```

Agent IDs are emitted by `AiUtopiaPettingZooEnv` with the role-prefixed convention (consistent with §7.1 m4_config example).

### D.3 RLlib config delta vs M1B

Building on `m1_gatherer_config()` in `src/aiutopia/train/config.py`:

```python
.rl_module(
    rl_module_spec=MultiRLModuleSpec(
        rl_module_specs={
            "gatherer_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=build_role_observation_space("gatherer", stage=2),
                action_space=build_role_action_space("gatherer"),
                model_config={ "role": "gatherer", ... },
            ),
            "builder_policy": RLModuleSpec(
                module_class=AiUtopiaRoleRLModule,
                observation_space=build_role_observation_space("builder", stage=2),
                action_space=build_role_action_space("builder"),
                model_config={ "role": "builder", ... },
            ),
        },
    )
)
.multi_agent(
    policies={"gatherer_policy", "builder_policy"},
    policy_mapping_fn=_policy_mapping_fn,
    policies_to_train=["gatherer_policy", "builder_policy"],
)
.training(
    learner_config_dict={
        "algorithm_config_overrides_per_module": {
            "gatherer_policy": {"lr": 3.0e-4},  # M1B-validated
            "builder_policy":  {"lr": 2.1e-4},  # spec §7.1 — 0.7x gatherer
        }
    }
)
```

Note: multi-timescale LR is wired from M2-β onwards (the spec puts it active from M3, but the §7.1 example config shows both `gatherer_policy: 3e-4` and `builder_policy: 2.1e-4`, so we follow the explicit config example).

### D.4 CTDE critic — now actually 2 of 4 slots populated

M1B's `CTDECriticModule._forward_train` builds an `all_agents (B, 4, core_dim)` tensor with slot 0 = self, slots 1-3 = zero. For M2:

- Slot 0 = self's core_obs (computed via `CoreEncoder._fwd` on this agent's obs).
- Slot 1 = the other agent's core_obs (computed by the env wrapper exporting both agents' obs to a shared `privileged_state` key the critic reads).
- Slots 2-3 = zero (farmer / defender, M3-M4).
- `village_inv` = aggregate inventory across both agents' main inventories + any deposited-to-chest inventory (M2-β when chests come into play).

**The env-wrapper hook:** spec §7.2 critic expects `batch["privileged_state"]["all_agents_obs"]`. M1B's wrapper does not emit this — it relies on the critic synthesizing self-only and zero-padding. M2 wrapper must emit `privileged_state` per env step, containing both agents' full core_obs vectors plus the village_inv aggregate. RLlib's `ConnectorV2` carries this through to the learner batch as a "global" obs key.

**Two-stage encoder pattern:** spec §4.3 calls out a per-agent shared encoder compressing each agent's obs to 128-d, then MLP over the 4×128 + village_inv. M1B implements this. For M2, the *same* CTDECriticModule is reused — only the slot-1 input changes from zero to actual obs.

**Implementation detail — privileged state plumbing:** the cleanest way to thread the "other agent's obs" through RLlib's ConnectorV2 pipeline on the new API stack is to declare a top-level dictionary key `privileged_state` in *each* agent's observation Dict. The key holds a fixed-shape array `(4, core_obs_flat_dim)` populated by the env wrapper at step time: slot 0 = self, slot 1 = the other live agent, slots 2-3 = zeros. RLlib treats this like any other obs key and forwards it through `_forward_train` (but **not** through `_forward_inference` — the actor doesn't see it).

The downside is the actor path also receives this key in its batch and must explicitly ignore it (don't flatten it into role_feat). M1B's `flatten_core_obs_batched` does not currently know about `privileged_state`; T16 must extend it (allow-list of keys it consumes, ignore the rest).

Alternative: wire `privileged_state` as a separate post-processing connector that injects into the training batch after the policy's forward pass. Cleaner but requires writing a custom `LearnerConnector` — more code, less risky. Recommend the obs-key approach for M2; can refactor in M3.

### D.5 Reward aggregation

Each agent gets its own per-tick reward per §5.2 compute_reward (role-specific primary + PBRS + universal terms). The `r_team_progress` Stage-2 term is identical across both agents on a given tick:

```python
r_team_progress_m2 = (5.0 if blueprint_complete_this_tick else 0.0)
                   - 0.001 * structure_damage_this_tick   # any agent breaking placed correct blocks
```

This is shared between the two agents (both feel +5.0 on completion). This is the spec's intended cooperative signal — neither agent can claim it without the other.

---

## E. BULK_FARMING exploit detector design

### E.1 Two transfer paths (spec extension)

Spec §5.3 item 4 only addresses drop-then-pickup ground transfer. The user's worked example (one agent dumps into chest, another picks up) is a **different** event sequence and the spec is silent on it. M2 must handle both.

**Path 1 — ground drop (spec §5.3):**
```
Agent A drops item X (drop event in inv delta or motor module ITEM_DROP)
  → within 5 ticks
Agent B picks up X (pickup event = inv delta positive for X, and B is within 3 blocks of A's drop position)
  → increment collusion_counter[A, B, X]
  → on counter > 3 in last 200 ticks: penalty -1.0 SPLIT between A and B (-0.5 each)
```

**Path 2 — chest transfer (spec extension, NEW for M2):**
```
Agent A invokes DEPOSIT_CHEST with item X, scalar_param qty > 0, motor module returns SUCCESS
  → within 10 ticks
Agent B invokes a "withdraw" pattern: action.skill_type == FETCH_MATERIALS, motor returns SUCCESS, B's inventory gains X, same chest_id as A's deposit
  → increment collusion_counter[A, B, X]
  → SAME threshold (>3 in 200 ticks) → -1.0 split
```

**False-positive risk on path 2:** the gatherer→builder workflow is *exactly* this pattern, by design. Gatherer chops wood, deposits in chest, builder withdraws and builds. A single handoff is correct behaviour and must not trigger the penalty.

**Defense:**
1. The `>3` threshold means a single handoff (or even three) does not fire. Only repeated A→B chain on the *same item* within 200 ticks fires.
2. **Whitelist legitimate workflow:** if `b_blueprint_status` shows missing cells of block X and B is the builder and X is a building material (in B's `b_materials_needed`), do **not** increment the counter for that A→B→X transfer. The handoff is legit if the builder will visibly consume X within 50 ticks (track via `b_blueprint_status` delta — `missing → correct` count rises in the window).
3. Counter decays — buckets older than 200 ticks are pruned (spec already says 200-tick log).

**Resolution ordering matters:** the whitelist check must execute *before* the increment, not after, otherwise the counter ratchets up on legitimate handoffs and fires a delayed false positive. Concretely, the CollusionMonitor.step() routine is:

```
on chest_deposit(A, X, qty) at chest_id:
    record_pending_deposit(A, X, qty, chest_id, tick=now)

on chest_withdraw(B, X, qty) at chest_id within 10 ticks of recorded deposit:
    if B.role == "builder" and X in B.b_materials_needed:
        # Hold counter increment. Set 50-tick watch.
        schedule_whitelist_check(A, B, X, deadline=now+50)
    else:
        increment collusion_counter[A, B, X]
        prune_old_entries(200_tick_window)
        if collusion_counter[A, B, X] > 3:
            emit_penalty(A: -0.5, B: -0.5)

on schedule_check_fires(A, B, X, deadline):
    delta = builder.blueprint_status_correct_cells_in_window - prior
    if delta > 0:  # builder actually used X
        # Legit handoff. Do NOT increment counter.
        log("collusion_whitelist_hit", A, B, X)
    else:
        # Held material but didn't consume. Increment retroactively.
        increment collusion_counter[A, B, X]
        if collusion_counter[A, B, X] > 3:
            emit_penalty(A: -0.5, B: -0.5)
```

The 50-tick deadline is a tunable; if false-positive rate is too high in M2-β eval, raise to 100 ticks. The whitelist hit metric is logged separately (§E.3) so we can audit the false-positive risk empirically.

**Implementation surface:** env wrapper hook between `step_pre` (obs_prev capture) and `step_post` (reward emit). The hook maintains a global `CollusionMonitor` instance scoped to the env, NOT to a single agent's `ExploitDetector`. The per-agent `ExploitDetector` receives the BULK_FARMING penalty as part of `env_meta["exploit_penalties"]` injected by the wrapper.

### E.2 Penalty wiring

```python
# In env wrapper step():
for agent_id, agent_state in self.agents.items():
    if collusion_monitor.fired_this_tick(agent_id):
        agent_state.exploit_penalties.append(("bulk_farming", 0.5))
        # Logged separately in custom_metrics for diagnostics
```

The penalty is per-agent (0.5 each side, summing to -1.0 as spec says), additive to the per-agent reward stream alongside the per-agent `ExploitDetector` output.

### E.3 Metrics emission

For each episode, emit:
- `collusion_fired_count` (int)
- `collusion_whitelist_count` (int — handoffs that were *correctly* permitted; sanity check on the whitelist)
- `collusion_avg_chain_length` (float — average A→B chain count when fired)

These show up in TensorBoard as `ray/tune/env_runners/.../custom_metrics/collusion_*`. The M2-β gate requires zero `collusion_fired_count` in the gate evals (§A.1).

---

## F. M2 task chain skeleton (rough decomposition)

NOT the plan — just rough decomposition so the user knows what's involved. Each T-task in the eventual plan will have full acceptance criteria + tests.

| # | Task | Cluster | Est. days |
|---|---|---|---|
| T1  | Builder obs space — extend `spaces.py` with `b_*` overlay, `build_role_observation_space("builder", stage)` | env | 0.5 |
| T2  | Builder action space — extend `build_role_action_space("builder")` with 8-skill enum + masking helpers | env | 0.5 |
| T3  | `BuilderRoleEncoder` — voxel embedding lookup + 3D conv on grid + MLP on materials/progress | rl_module | 1.0 |
| T4  | Builder actor head — extend `actor_head.py` with `build_actor_head("builder")` returning 8-skill MultiDistribution config | rl_module | 0.5 |
| T5  | Builder action distribution wiring — `AiUtopiaRoleRLModule.setup()` branches on role for `action_dist_cls` | rl_module | 0.5 |
| T6  | Stub planner blueprints — 5 hand-coded structures at `planner/stub_blueprints.py` + tests | planner | 0.5 |
| T7  | Env wrapper: blueprint materialization → `b_voxel_grid` / `b_blueprint_grid` / `b_blueprint_status` / `b_materials_needed` / `b_progress` updates per tick | env | 1.0 |
| T8  | Fabric-side: voxel grid extraction (11×11×11 from chunk data centered on builder) + Py4J emit | fabric_mod (Java) | 1.0 |
| T9  | Fabric-side: `place_block` motor skill (already partially in M1A via Baritone? — verify) — verify path, gap-fill if needed | fabric_mod | 0.5–1.0 |
| T10 | Builder Stage-1 reward — extend `reward.py` with `_builder_primary_signal` (blueprint_status delta weights) and route via `compute_reward_stage_1` | env | 0.75 |
| T11 | Reward Stage-2 — implement `compute_reward_stage_2` with linear curriculum decay, multi-objective sum, team_progress, `r_deposit_bonus` (Stage-3 stub returns 0) | env | 0.75 |
| T12 | Curriculum step counter — env_meta carries `global_step` (already in M1B); add monotonic accumulation across phases | env | 0.25 |
| T13 | `CollusionMonitor` — chest + ground transfer paths, whitelist via blueprint_status delta, env-wrapper integration | env | 1.0 |
| T14 | M2 config builder — `m2_alpha_builder_solo_config()` and `m2_beta_coop_config()` in `train/config.py` extending M1B pattern | train | 0.5 |
| T15 | Multi-agent `policy_mapping_fn` + two-policy RLModuleSpec wiring | train | 0.25 |
| T16 | CTDE critic privileged_state pipe — env wrapper emits `privileged_state.all_agents_obs` + `village_inv`; RLlib ConnectorV2 carries through; `_forward_train` consumes slot-1 actually | rl_module + env | 1.0 |
| T17 | Eval scenarios — 4 scenario JSONs (§A.2) + scenario runner extension (parallels M1B's `scenario_runner.py`) | train | 0.75 |
| T18 | Promotion CLI multi-policy — `aiutopia promote-weights promote --multi-policy gatherer,builder` writes combined `policy_deployments` row | promotion | 0.5 |
| T19 | Determinism check extension — current M1B determinism path is gatherer-only; extend to per-policy determinism replay (run replay for both gatherer and builder, both must pass thresholds) | determinism | 0.5 |
| T20 | Hybrid Stage-1 → Stage-2 transition wiring (§B.3.1 sub-option ii) — env wrapper carries Stage-2 channels with weight 0 during M2-α, ramps over 1M env steps after switch | env | 0.5 |
| T21 | M2-α empirical training run — solo builder, 5 standard blueprints, gate against §A.1 M2-α | train (empirical) | 1–3 (wall-clock, may not converge first try) |
| T22 | M2-β empirical training run — gatherer + builder cooperative, gate against §A.1 M2-β | train (empirical) | 1–3 (wall-clock) |
| T23 | M2 verification doc — append "M2-Training Progress" section to `M2_TRAINING_PLAN.md` with tag commit, results, plan-D prereqs | docs | 0.25 |

**Total focused-eng days:** ~12–15 not counting empirical wall-clock retries.

**Suggested task grouping for parallelization / dependency order:**

```
Phase F1 (Builder role plumbing — sequential, foundational):
   T1 obs → T2 action → T3 BuilderRoleEncoder → T4 actor head → T5 dist wiring
   ≈ 3 days. Blocks everything else.

Phase F2 (Java-side motor — can start in parallel with F1):
   T8 voxel grid extraction → T9 place_block motor skill verify/gap-fill
   ≈ 1.5–2.5 days. Independent of Python F1; can run in parallel.

Phase F3 (Reward + curriculum):
   T10 builder Stage-1 → T11 Stage-2 with curriculum → T12 step counter → T20 hybrid transition
   ≈ 2.5 days. Depends on F1 (needs builder action shape).

Phase F4 (Multi-agent + CTDE):
   T6 stub blueprints → T7 blueprint materialization → T13 CollusionMonitor → T14 M2 config → T15 policy_mapping → T16 CTDE privileged_state
   ≈ 3.5 days. Depends on F1/F3.

Phase F5 (Eval + promotion + determinism):
   T17 eval scenarios → T18 promotion CLI multi-policy → T19 determinism multi-policy
   ≈ 1.75 days. Depends on F4.

Phase F6 (Empirical):
   T21 M2-α run → (gate) → T22 M2-β run → (gate) → T23 docs
   ≈ 2–6 wall-clock days. Depends on all of F1-F5.
```

The critical path is F1 → F3 → F4 → F5 → F6, with F2 in parallel. F2 *must* be complete before T7 starts (T7 consumes voxel grid emission from F2). If F2 slips by 1 day, the F1→F3→F4 critical-path engineer can move ahead independently; recovery is bounded.

---

## G. Estimated effort

| Bucket | Lower | Upper |
|---|---|---|
| Engineering (T1–T20, T23 — non-empirical) | 9 days | 12 days |
| Empirical training + debug (T21, T22) | 2 days | 6 days |
| Slack for unforeseen Fabric-side mod issues (T8, T9) | 0.5 days | 2 days |
| **Total focused work** | **11.5 days** | **20 days** |

**Honest range: 7–12 focused engineering days + 2–6 empirical wall-clock days = 9–18 total days.**

**Why this is bigger than M1B (which was ~2 focused days + ~0.5 day debugging):**

| Factor | M1B | M2 |
|---|---|---|
| Roles trained | 1 (gatherer) | 2 (gatherer + builder) |
| Obs space complexity | gatherer overlay only | builder overlay incl. 3 voxel grids (~4000 fields) |
| Action space | 6-skill gatherer (done) | 8-skill builder (new, more complex masking) |
| Reward | Stage-1 only | Stage-1 + Stage-2 + curriculum + team_progress |
| Fabric-side motor skills | Already in M1A | `place_block` may need gap-fill; voxel grid extraction is new |
| Multi-agent | Single-policy single-agent | Two-policy CTDE with shared episode |
| Exploit detector | Per-agent only | Per-agent + cross-agent `CollusionMonitor` (new, novel logic) |
| Empirical convergence | Single role | Two roles + cooperative, more failure modes |
| Spec deviations needing approval | 1 minor (LSTM ownership) | 2 (pixels deferred, weight-sharing deferred) |

The user's suggested estimate (4–7 focused days) is optimistic by a factor of ~2x. The M1B Plan v2's estimate (4–6 weeks at 10–15 hr/wk, ≈ 60–90 hr, ≈ 8–12 focused days) is closer to right. I'm recommending 9–18 total days inclusive of empirical retries.

**Critical-path items most likely to slip:**
1. **T8 Fabric-side voxel grid extraction** — new Java code reading chunk data and JSON-serializing 1331 cells per tick per builder. May hit performance walls (per-tick Py4J payload size); if so, need to switch to compressed encoding.
2. **T21 M2-α convergence** — first time training the builder. We have no prior data on whether the symbolic blueprint signal alone (no pixels per §B.1) is rich enough. The recommended deviation is bet on this; if it doesn't pan out, +3-5 days for pixel encoder.
3. **T22 M2-β convergence** — first cooperative training. Failure modes specific to this stage (per spec §5.9): credit assignment, lazy free-rider, role collapse. All have mitigations on the roadmap but the first cooperative training run is unlikely to be clean.

---

## H. Open questions for the user (decisions needed before plan)

Each framed as: "We need to decide X before plan. Options: A, B, C. My lean: X."

### H.1 Pixel-patch deferral approval (spec deviation)

**Decide:** ship pixel encoder in M2 per spec, or defer to M3 pending plateau evidence?

- **(a)** Ship Iris/Sodium offscreen framebuffer — full spec compliance, unknown mod compatibility risk.
- **(b)** Ship software raycaster — full spec compliance, +50ms per skill invocation training cost, 1-2 days extra engineering.
- **(c)** Defer pixels to M3, symbolic-only for M2 — spec deviation, plateau-gated reactivation if M2-α blueprint_match < 65% at 5M env steps.

**My lean: (c).** Reasoning in §B.1. Saves 1-2 days minimum, eliminates highest-risk engineering item. **Requires your explicit approval as a spec deviation.**

### H.2 Cross-policy weight sharing scope

**Decide:** attempt parameter sharing of CoreEncoder/SharedBackbone/CTDECritic in M2, or defer to M3?

- **(a)** Own copy per role for M2 — 12M params total, no optimizer-doubling problem. Defer sharing to M3.
- **(b)** Share submodules + LR-halving mitigation — spec-compliant but fragile.
- **(c)** Share submodules + custom Learner that dedupes optimizer groups — most correct, ~2 days of Ray internals work.

**My lean: (a).** Reasoning in §B.2. Spec-deviation flagged but not requiring approval since architecture is preserved and §4.3's "shared" intent is re-opened in M3 when the cost case actually applies. **Acknowledge unless objections.**

### H.3 M2-α → M2-β switch protocol

**Decide:** hard switch, hybrid pre-exposure, or skip M2-α entirely?

- **(a)** Hard switch at iteration N — clean cutover, possible cliff.
- **(b)** Hybrid pre-exposure with 1M-step ramp — mirrors §5.1 Stage-2→3 pattern, costs almost nothing.
- **(c)** No M2-α, start in M2-β day one — skips solo-builder de-risking phase.

**My lean: (b).** Reasoning in §B.3.1. **Acknowledge unless objections.**

### H.4 M2-β gate threshold

**Decide:** keep spec floor (50%) or tighten to 60%?

- **(a)** Spec floor: cooperative structure_built_correctly > 50%.
- **(b)** Tightened: 60%, with 50% as rollback threshold.
- **(c)** Aggressive: 80% (user's suggestion).

**My lean: (b).** Reasoning in §A.1. 80% is plausible for M3 (after farmer adds gathering speed) but aggressive for the first cooperative training run. **Acknowledge unless objections.**

### H.5 M2 stub planner — round-robin vs random

**Decide:** during M2-α, does the stub planner cycle blueprints round-robin or pick uniform random per episode?

- **(a)** Round-robin — deterministic curriculum coverage.
- **(b)** Uniform random — more sample diversity.
- **(c)** Curriculum: start with simplest (wall), introduce harder structures as `blueprint_match` rises.

**My lean: (a) for M2-α**, **(b) for M2-β**. (c) is most principled but adds curriculum-scheduling complexity for marginal benefit at this scale. **Acknowledge unless objections.**

### H.6 Builder warm-start from M1B gatherer

**Decide:** does the M2-β gatherer policy initialize from M1B's `gatherer/v1/` checkpoint, or train both policies from scratch?

- **(a)** Warm-start gatherer from M1B, train builder from scratch.
- **(b)** Train both from scratch.
- **(c)** Skip M1B checkpoint use entirely; M1B becomes a release-only artifact, M2 starts fresh.

**My lean: (a).** Reuse what we paid to train. Warm-start path already exists via `aiutopia promote-weights load` (or analogous CLI). **Acknowledge unless objections.**

### H.7 Spec extension approval — BULK_FARMING chest path

**Decide:** the spec is silent on chest-transfer collusion; my §E proposal extends the spec. Approve?

- **(a)** Approve extension as drafted (10-tick chest window, whitelist via blueprint_status delta, same `>3` threshold).
- **(b)** Approve detection but require manual review (no auto-penalty) for first M2 milestone; emit metric only.
- **(c)** Reject extension — only ground drop transfer per spec §5.3.

**My lean: (a).** Without it, the most obvious M2 reward hack (gatherer dumps to chest → builder takes credit) is undetected. (b) is the conservative posture if false-positive risk concerns you. **Requires your call before T13 starts.**

---

## I. Risks (architectural, ranked by probability × impact)

| Rank | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | M1B does not converge in current session — M2 paused | medium | high (delays everything) | §0 outlines paused posture; M2 will not start until M1B gates pass. Daily check on M1B run health. |
| 2 | Symbolic-only builder plateaus below 65% (pixel-deferral bet fails) | medium | medium (+3-5 days for pixel encoder under deadline) | Plateau threshold + reopen clause in §B.1. M2-α is exactly the de-risking phase that catches this early. |
| 3 | Cooperative M2-β reward-hacks via cross-agent transfer despite CollusionMonitor whitelist | low | high (false positive on legit handoff, OR false negative on collusion) | Manual eval review of collusion_* metrics during M2-β training. Whitelist threshold tuning if false-positive rate > 5%. |
| 4 | Voxel grid extraction (T8) hits per-tick performance wall | medium | medium (forces compressed encoding or chunk caching) | Pre-T8 spike: measure Py4J payload size for 1331-cell grid; if > 50ms serialize time, switch to gzip-compressed bytes + Python-side decompress. |
| 5 | Optimizer-doubling on shared params if we ever revisit B.2 (b) | low (we recommend (a)) | high if it bites | Avoided by sticking with (a). If user overrides to (b), demand reference implementation from Ray docs before proceeding. |
| 6 | `place_block` motor skill timing brittleness (Baritone vs Fabric API mismatch) | medium | medium | Verify M1A's existing Baritone integration covers `place_block`; T9 gap-fill if not. |
| 7 | Two-agent episode timeout (12K ticks) too short for cooperative shelter | low-medium | low (just retune) | Tunable env_config field; default 12K matches M1B, will adjust if T22 shows timeouts dominate. |
| 8 | Per-role contribution drift — builder learns to wait for gatherer indefinitely (free-rider) | medium-high in M2-β | medium | Spec §5.9 LAIES intrinsic with 0.1× extrinsic weight wired from M2 onward. Per-role contribution < 20% of fair-share over 500 episodes triggers intervention. |

---

## J. Spec coverage matrix (self-review)

Verifies the brainstorm touches every spec section listed in the prompt. Empty = covered elsewhere implicitly.

| Spec section | Topic | Where covered in brainstorm |
|---|---|---|
| §4.3 Shared backbone | Architecture | §B.2 (weight sharing deferral); §D.4 (CTDE critic) |
| §4.4 Pixel patches | Builder pixel encoder | §B.1, §C.1, §H.1 |
| §4.6 Per-tick RL loop | Loop semantics | §D.1 |
| §5.1 Three stages | Stage 1/2/3 + curriculum decay | §B.3, §C.3, §D.5 |
| §5.2 compute_reward | Unified function | §C.3 (Stage-1 + Stage-2 wiring) |
| §5.3 Exploit catalog | BULK_FARMING + others | §E |
| §5.7 tech_tree_potential | PBRS Φ | §C.3 (reuse from M1B as-is) |
| §5.9 MARL failure mitigation | Lazy / role collapse | §I.8 risk; §C.3 multi-timescale LR via D.3 |
| §7.1 PPOConfig multi-agent | Two-policy config | §D.3 |
| §7.2 RLModule spec | AiUtopiaRoleRLModule | §B.2 (reuse), §D.4 (CTDE pipe) |

Spec sections **not in M2 scope** but flagged for clarity:
- §3 (LLM planner) — M5.
- §4.7 Event-driven LLM loop — M5.
- §4.8 Inter-agent comm — engineered protocols only in M2 (broadcast/directed/alert); learned 128-d payload gated to M5.
- §4.9 Episodic memory — already wired in M1B; M2 adds builder-emitted importance signals but no schema change.
- §5.4 Per-role primary signals — builder + gatherer rows used; farmer/defender deferred.
- §5.5 γ_clip — unchanged, reused.
- §5.6 Memory retrieval — read path used by planner; not exercised heavily in M2 since planner is stub.
- §5.8 Phase mapping — drives §A's M2-α and M2-β gates.
- §5.10 Promotion criteria — reused via T18 multi-policy extension.

---

## K. Contradiction / scope-realism check (self-review)

**Contradiction check:**

1. *§B.2 recommends own-copy per role; §4.3 spec says "parameter-shared". Contradiction?*
   No — flagged explicitly as spec deviation in §B.2 final paragraph. Architecture preserved; mechanism deviates; sharing re-opens in M3. Acknowledged in §H.2.

2. *§B.1 recommends symbolic-only; §4.4 spec says pixel patches at M2+. Contradiction?*
   Yes — flagged explicitly as spec deviation in §B.1 ("requires user approval"). Listed in §H.1 as a required user decision.

3. *§A.1 recommends 60% threshold; §5.8 spec says > 50%. Contradiction?*
   No — tightening from spec floor. Spec is the floor, brainstorm proposes a tighter bar with floor preserved as rollback threshold. Listed in §H.4.

4. *§E proposes chest-transfer detector; §5.3 only specifies drop-pickup. Spec extension?*
   Yes — flagged explicitly in §E.1 ("spec extension, NEW for M2"). Listed in §H.7 as a required user decision.

5. *§D.3 sets builder LR to 2.1e-4 (0.7× gatherer); §5.9 says multi-timescale LR active from M3, not M2.*
   Mild tension — spec §7.1's example config sets the per-role LR overrides at the M4 example point, but the §5.9 mitigation wiring table puts active multi-timescale at M3. I followed §7.1's explicit config example since the M4 config is what we're extending. Flagged here; if user prefers same-LR-for-both in M2, change is a single line. Listed implicitly in §H but worth adding to user's mental list.

**Scope realism check:**

1. *Is 9–18 days realistic given M1B took 2.5 days?*
   M1B's 2.5 days was a single-role training stack on already-built obs/action/reward pipeline. M2 adds: a full role (obs/action/reward), multi-agent CTDE wiring, cross-agent exploit detector, curriculum stage transition, Fabric-side voxel extraction, two empirical training runs. The 3.6×–7.2× ratio over M1B's 2.5 days matches the work delta.

2. *Are 23 tasks too many for a brainstorm task chain?*
   T-task density mirrors M1B's structure (M1B has ~21 tasks). Each task is granular enough to estimate. Could collapse T1+T2 (obs+action) and T4+T5 (actor head + dist wiring) into 2 tasks; would not shorten timeline.

3. *Does the M2-β gate (cooperative 60%) actually exercise CTDE?*
   Yes — cooperative shelter requires gatherer to gather AND builder to build AND chest handoff to happen. If either policy is broken or if CTDE critic doesn't credit-assign correctly, the gate fails. This is the right test.

4. *Are we punting too much to M3?*
   What's deferred to M3: parameter sharing, farmer role, pixel encoder (potentially), LAIES intrinsic only as monitoring (active from M3 per §5.9). What's *in* M2: builder role complete, multi-agent cooperative, cross-agent exploit, curriculum stage transition. M2's scope is full per spec §5.8 phase-2; M3 adds farmer + sharing-refactor.

5. *Length budget — target was 800–1500 lines.*
   Document at ~790 lines, hitting the floor of the requested range. §B.1/B.2/D.4/E expanded with implementation-detail paragraphs (encoder shape, optimizer-doubling concrete reasoning, privileged_state plumbing, collusion ordering); §F added phase-grouping; §L added verification matrix. The expansions are load-bearing — they answer questions a downstream plan author would otherwise have to re-derive — not padding.

---

## L. Verification plan (what M2 falsification looks like)

Per architect-discipline "a decision without verification is a hope": for every consequential M2 decision in this brainstorm, here is what evidence would prove it wrong, when that evidence would be collected, and what the fallback is.

| Decision | Verification artefact | Falsification trigger | Fallback |
|---|---|---|---|
| §B.1 defer pixels | M2-α `blueprint_match` curve over 5M env steps | Plateau < 65% over rolling 1M-step window | Reopen §B.1, ship software raycaster (option b) under +3-5 day timeline. |
| §B.2 own-copy per role | Sample efficiency comparison: M2-α gatherer leg vs M1B gatherer baseline, per million env steps | M2 gatherer leg > 3× slower than M1B baseline (suggesting independent encoders cause negative interference) | Reopen §B.2, attempt mitigation (i) freeze-on-builder for next phase. |
| §B.3 hybrid pre-exposure transition | Stage-2 reward channels carried during M2-α, weight=0 | If channel computation is buggy (NaN, dim mismatch), unit-test catches at T11 | Hard switch (option a) — fall back to clean cutover. |
| §A.1 M2-α gate at 70% blueprint_match | Eval scenario `m2_blueprint_*_solo` over 3 consecutive eval batches | < 70% on 3 evals after 10M env steps total training | Lower to 60% (spec-floor adjacent), or pivot to (B.1 b) software pixels. |
| §A.1 M2-β gate at 60% cooperative + zero BULK_FARMING | Eval scenarios + collusion_* metrics | < 50% (spec rollback floor) cooperative OR `collusion_fired_count > 0` | Below 50% triggers M2-β redesign. BULK_FARMING firing in evals (not training) triggers detector tuning. |
| §E whitelist algorithm | `collusion_whitelist_count` vs `collusion_fired_count` ratio in M2-β training | False-positive rate (whitelist correctly suppressing on legit handoff) > 5% means the threshold is wrong | Raise 50-tick watch window to 100; or tighten threshold from `>3` to `>5`. |
| §D.4 privileged_state obs-key wiring | Critic loss curve during M2-β training | If `vf_loss` is NaN or stuck (suggests obs-key not flowing through correctly), T16 test catches; if it trains but value estimates are biased toward self-only baseline (suggests slot-1 is silently zero), CTDE eval at end of T16 catches | Refactor to `LearnerConnector` injection (alternative mentioned in §D.4). |

Each row is intentionally falsifiable: there is at least one metric and one numeric threshold per decision. A decision without a quantifiable falsifier is not on this table.

**Verification gating cadence:**
- T-task tests (acceptance criteria in the eventual plan) catch wiring and shape bugs before training starts.
- M2-α empirical run (T21) catches reward-design and obs-encoder issues in the solo phase. Falsifies §B.1 (pixel deferral) and §A.1 (M2-α gate).
- M2-β empirical run (T22) catches cooperative-training issues. Falsifies §A.1 (M2-β gate), §B.2 (own-copy efficiency), §E (collusion false-positive rate).
- Post-M2 review: write up empirical findings on M1B-style "M2-Training Progress" addendum to the eventual M2 plan, including all of the above metrics, before declaring M2 done. This is the M3 planning input.

---

## M. Headline recommendations (executive summary)

1. **Two gates, not one** — M2-α (solo builder 70% blueprint_match) and M2-β (coop 60% structure_built) per spec §5.8.
2. **Defer pixels** — symbolic blueprint signal carries M2; pixel encoder plateau-gated to M3. **Requires user approval.**
3. **Own-copy per role** for CoreEncoder/SharedBackbone/CTDECritic — avoid Ray's optimizer-doubling rathole; reopen sharing in M3.
4. **Spec ratifies** stage transition (linear curriculum decay) and builder action space (8 skills + Dict header) — don't relitigate.
5. **BULK_FARMING extends** to chest path with blueprint-aware whitelist — spec extension, **requires user approval.**
6. **Effort: 9–18 days** — ~2x the user's optimistic 4–7 day estimate, calibrated to M1B's actual cost and the M2 work delta.
7. **M2 blocked on M1B** — implementation does not start until M1B's narrow gates pass. Daily check on the in-flight training run.

End of brainstorm. Next step: user feedback on §H decisions; on user approval, this becomes input to the M2 implementation plan (`M2_TRAINING_PLAN.md`).
