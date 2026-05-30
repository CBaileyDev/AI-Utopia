# M2 Roadmap & Decisions — for your return (2026-05-30, autonomous run)

## 🟢 SCOPE DECISION (2026-05-30, user) — PEACEFUL survival world

The project targets a **peaceful** (or near-peaceful) natural survival world. This
**descopes the entire combat/threat layer**: no hostile mobs, no hunger drain → the
tick-warp-combat-observability blocker, fight/flee/eat skills, and the Defender role
are all **dropped or trivial**. The D3 survival-recon "dies step 1" result is moot.
What REMAINS (the focus): natural terrain + survival mechanics (timed breaking, tools,
scattered forests), gather/build/farm roles, multi-agent cooperation — and the
Explorer "find the forest" problem, which is now REAL + testable (the flat toy had no
exploitable signal; a natural world does → this unblocks D1). **Key first risk:** the
obs scans topmost-non-air per column → on natural trees that's LEAVES, so logs may be
invisible (Phase-D scoping §3.2). Recon in progress: can the proven Lumberjack perceive
+ harvest real forest trees on peaceful natural terrain?



Honest map of where the project stands and the **decisions only you should make**,
so picking the next milestone is fast. Companion to `Research/MORNING_BRIEF.md`
(the ablation + scout detail) and `NEXT_SESSION.md` (volatile state).

---

## What is PROVEN (bank these)

- **M1B Lumberjack works and transfers sim→real 3/3** (HARVEST-spam policy; robust,
  survival-forest fidelity). The fast-sim → real-transfer methodology is proven **for
  one collect-style role**.
- **The architecture fork is decided: Fork A** (thin reactive controller + smart
  producers), corroborated by external research (Kimi swarm, `Research/okcomputer/`)
  and our ablation. Flat-RL-fails-long-horizon-exploration is field consensus, not a
  novel failure; the decision-core matches Plan4MC's "Finding-skill".
- **A reactive controller + perception-triggered harvest clears foraging** when given
  a bearing (oracle or even a fixed heading on the small arena).

## What is OPEN / corrected (do NOT build on these as if solved)

- **Partial-information scouting (the real Explorer role) is UNTESTED — and untestable
  in the current sim.** The toy arena gives no exploitable signal about a distant
  cluster's direction (you only know "found / not found"), so *systematic coverage is
  optimal there* and "a producer that beats coverage by using observations" is
  ill-posed. Two of tonight's scout results were withdrawn over-claims (the `clusters`
  arena was degenerate — B always south; the 50/50 `SweepScout` is open-loop coverage
  tuned to a known radius, not partial-info reasoning). See MORNING_BRIEF "UPDATE 2".
- **The decision-core policy learns ~nothing beyond a reactive script** (oracle
  ablation: a zero-learning follower ≥ trained PPO in every cell); PPO is also
  non-convergent here. Fine *if* you accept Fork A's thin controller; a learned
  controller is unvalidated.
- **The M1 stack is gatherer-HARDCODED** (`spaces`, `role_encoder`, `actor_head`,
  action-dist all `raise NotImplementedError` for other roles) — a second role is real
  M2 plumbing + design, not a config flip.

---

## The DECISIONS awaiting you (each unblocks a concrete build)

### D1 — Where does the Explorer's "where to go" come from? (the Fork-A producer)
The producer is the crux and it needs an environment with *exploitable signal* to be
meaningful. Options:
- **(D1a) Real-MC signal.** In real Minecraft, distant forests ARE partially
  observable (tall trees over terrain, biome color, chunk heightmaps). The Explorer
  question becomes real only against real MC or a sim enriched with such signal.
  *Implication:* the Explorer is a real-MC / rich-sim problem, not a flat-toy problem.
- **(D1b) Memory/map producer.** A persistent occupancy/semantic map + frontier
  selection (research's Level-1/2). Needs the env to reward *efficient* search
  (steps-to-clear at *variable/unknown* radius) or it degenerates to coverage.
- **(D1c) Accept coverage.** If "find wood" just needs completeness (not speed), a
  systematic spiral suffices and the Explorer is ~solved (cheaply). Decide whether the
  village cares about search *efficiency*.
**My rec:** treat the Explorer as a real-MC problem (D1a) — don't sink more time into
the signal-free toy. Defer until a role actually needs distant-resource search in real MC.

### D2 — Which second role next, and does the sim express it?
The methodology is proven for *collect* roles. In the FLAT sim, Miner ≈ Gatherer (mine
stone instead of wood) — proves only resource-agnosticism (low value). Roles that
*meaningfully* test generalization need new mechanics:
- **Farmer** — plant → grow (delayed) → harvest: genuinely different (temporal/delayed
  reward, a skill *sequence*). Sim-expressible (plant action + growth timer). New skills
  Java-side for transfer.
- **Defender** — combat: new obs (hostiles), new action (attack), survival pressure.
  Biggest jump; closest to "survival village".
- **Miner** — cheapest, lowest-signal in the flat sim.
**My rec:** **Farmer** is the best generalization test that's still tractable (delayed
reward is the new axis; no combat complexity). But this is a design call (crop model,
reward shape, transfer skills) — greenlight + I'll spec→build it.

### D3 — Push the proven Lumberjack toward the real survival village?
The end goal is a *persistent survival* village; everything so far is a peaceful toy
arena. A high-value, different-axis step: run the proven Lumberjack as a persistent
agent in a real survival world (hunger, day/night, real trees) and see what breaks.
This attacks the *real* end-goal gap (survival realism) rather than adding roles to a
toy. **My rec:** strong candidate — it's where the deployable-village risk actually lives.

### D4 — Finish M1B formally (promotion, §5.10)?
Mostly ceremonial (the 3/3 transfer IS the validation). Blocked on: the M1B
checkpoint's `aiutopia_metrics.json` is empty (gate inputs never persisted) + Gate 5
determinism needs a live server + GPU. Low value/effort ratio; do it only if you want
the formal tag. I can do it once you confirm it's worth the plumbing.

---

## Recommended priority (my call, you override)

1. **D3 — Lumberjack in a real survival world.** Attacks the actual end-goal gap
   (survival realism), reuses the proven policy, unambiguous (does it survive + gather?).
2. **D2 — Farmer** as the second-role generalization test (delayed-reward mechanic).
3. **D1 — Explorer** deferred until a real-MC role needs distant search (D1a).
4. **D4 — promotion** whenever you want the formal M1B tag.

The common thread: **move off the flat peaceful toy toward survival + real MC**, where
the end-goal risks (survival, real-resource search, multi-role coordination) actually
live — rather than refining a sim that keeps rewarding whatever exploits its structure.

## ⭐ The key strategic takeaway (why I did NOT just build more sim roles tonight)

Across this whole session one pattern recurs: **the fast-sim's "do-everything" skills
make every collect-role a skill-spam task where the policy barely learns.** The M1B
gatherer "works" because `HARVEST` internally finds + chains through the field — the
policy mostly emits HARVEST. The decision-core tried to move the deciding into the
policy and we found it learns ~nothing (the env/oracle decides). A sim Miner/Farmer
would very likely repeat this: "it collects (via skill-spam), policy learned little."

**Consequence:** "generalize the methodology by adding sim roles" mostly proves
*resource-agnosticism*, not genuine multi-role intelligence — and each such claim is a
fresh over-claim risk (this session caught 5). Genuine learning/decision tests require
either:
- an environment where skills **can't** do everything, so the policy *must* sequence/
  decide (real-MC survival, or a sim with partial obs + non-chaining skills + real
  consequences), **or**
- accepting Fork-A **thin controllers** and investing in the **producer/planner layer**
  (the M5 LLM-planner + memory + scout) — i.e. the intelligence lives above the policy.

This is why the recommendations push toward real-MC/survival (D3) and the planner layer,
not more flat-sim roles. I deliberately did not build a toy role tonight because it would
manufacture a low-value "win" of exactly the kind that's been getting walked back.

## ✅ D3 RECON DONE (2026-05-30) — survival is a MAJOR MILESTONE, not a quick increment

Ran the proven Lumberjack under real survival pressure (runtime `runCommand`, no Java
rebuild). Report: `Research/SURVIVAL_RECON.md`; code: `scripts/survival_recon.py`
(commits `87972f2`, `019bab2`). Honest findings (subagent self-corrected 3 over-claims):
- **Dies on step 1 to a single zombie** (mob-caused, isolated by a no-threat control
  that survived 60 steps). No armor, no fight/eat skill, no learned flee.
- **INFRA BLOCKER — atomic combat under tick-warp:** combat resolves *between* obs
  samples, so the policy never gets a mid-combat observation. **Survival behavior is
  unobservable/untrainable while tick-warp is on** — this must be solved first.
- `g_hostiles_nearby` **populates correctly with real mobs** (positive — only zero in
  training). The policy **ignores it** (OOD: trained at constant health/hunger, empty
  hostiles).
- Starvation caps at 1 HP on `/difficulty normal` (never kills) → **mobs, not hunger,
  are the killer** → an EAT skill is NOT the priority; surviving mobs is.
- Carpet fake players spawn **invulnerable** until `/gamemode survival` (re-applied per
  episode) — a real-MC gotcha for any survival work.
- Flagged (not root-caused): the recon harness collected 0 oak_log even with no
  pressure — likely the documented real-HARVEST non-determinism (memory: "#1 Phase-C
  item"), since `transfer_eval` still gets 3/3.

**Survival milestone scope (the real shape):** (1) solve tick-warp combat
observability (per-tick obs during threat, or tick-warp off in combat) — the
prerequisite; (2) a survival reward (penalize death/damage); (3) survival-aware
training (mobs/hunger varied in the env so the OOD signals become in-distribution);
(4) the cheapest survival CAPABILITY is **flee** — it needs NO new skill (NAVIGATE
away from `g_hostiles_nearby` already in the action+obs space), just training; fight
(ATTACK) / eat are later. This is M-level work (multi-session), worth a deliberate
go/no-go before sinking build hours.

## Infra built this run (kept, reusable)
`sim/scout.py` (FrontierScout, SweepScout), `sim/world.py clusters_omni`,
`scripts/dc_scout_follower.py`, `scripts/dc_ablation.py`, the sim mask/cue knobs +
their tests (suite 227 green). All sound as infrastructure even though the scout
*claims* were withdrawn.
