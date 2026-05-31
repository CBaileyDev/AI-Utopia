All four load-bearing facts are confirmed from source:

1. **Explorer dim mismatch**: `ExplorerRoleEncoder.flat_mlp` ends in `nn.Linear(128, 64)` (role_encoder.py:30/66 — the explorer has only the flat_mlp, no grid_conv), so `role_feat` is 64-d. explorer_rl_module.py:124 fuses `core(256)+role(64)+pixel(64)=384`, fed to `SharedBackboneModule.proj = nn.Linear(448, 384)` (shared_backbone.py:9,17). RuntimeError on first explorer forward. Confirmed.

2. **prev_phi clobber**: sim_env.py:387 writes `self._prev_phi[agent] = phi` unconditionally at the `if self._distance_shaping` block level (not gated by `_nearby_now`), running BEFORE line 402's read in the same step. With both flags on, line 402 reads the just-written `phi` and computes `phi_a - phi = 0`. The advisor's verdict (silent zeroing of approach_shaping, not "poisoning") is correct. Confirmed.

3. Gatherer actor head = 256+8+2+2+6+6+64 = 344 (comment says 340). Confirmed.

4. Farmer actor head = 256+8+2+2+7+6+64 = 345 (comment says ~344). Confirmed.

I have enough to synthesize decisively. No advisor call needed — the two non-trivial findings are already adversarially verified by the provided verdicts AND independently re-confirmed against source here.

---

# AiUtopia Audit — Prioritized Fix List

## (1) MUST-FIX — real correctness/parity bugs

### M1. ExplorerRoleEncoder outputs 64-d, breaking the 448-d backbone contract
- **File:** `src/aiutopia/rl_module/role_encoder.py:62-67` (fix site); manifests at `explorer_rl_module.py:124` → `shared_backbone.py:17`
- **Bug:** Explorer `flat_mlp` ends `nn.Linear(128, 64)`; explorer has no `grid_conv` branch (unlike gatherer/farmer), so `role_feat=64`. Fused = 256+64+64 = **384**, but `SharedBackboneModule.proj = nn.Linear(448, 384)`. First explorer forward raises `RuntimeError` (mat1/mat2 mismatch). **Latent**: no test instantiates the explorer module; explorer is M2-only.
- **One-line fix:** Change the final layer of `ExplorerRoleEncoder.flat_mlp` to output **128** (`nn.Linear(49,128)→ReLU→nn.Linear(128,128)→ReLU`), making fused = 256+128+64 = 448. Update stale comments at `explorer_rl_module.py:120` (`64`→`128`) and `:124` (`384`→`448`).
- **Parity risk:** **NONE for M1.** Touches no obs/mask/reward (`reward.py`, `spaces.py`, `sim/obs_adapter.py`, `sim/world.py` untouched). No trained explorer checkpoint exists, so "retraining" risk in the original finding is moot. Add a forward-pass smoke test (one `(B,)` and one `(B,T)` batch) to lock the contract.

### M2. `_prev_phi` clobber silently zeroes approach_shaping when distance_shaping is co-enabled
- **File:** `src/aiutopia/sim/sim_env.py:387` and `:403` (plus init `:160`, reset `:294`)
- **Bug:** Line 387 (`self._prev_phi[agent] = phi`) runs unconditionally at the `if self._distance_shaping` block level — NOT inside the `if not _nearby_now` gate — and executes BEFORE line 402's read in the same step. With both flags on: line 402 reads the just-overwritten `phi` → `phi_a - phi = 0` every step. **Net effect: approach_shaping is fully disabled, not "poisoned."** (The original finding's "mixing two different potentials" mechanism is wrong — both branches call the identical `_log_potential(world)`; the real defect is execution-ordering.)
- **One-line fix:** Add a separate `self._prev_phi_approach: dict[str,float] = {}` (init + seed at reset), and have the approach branch read/write ONLY that dict — leave `_prev_phi` for distance. (Equivalent to the dict-of-dicts fix but less churn; do NOT just edit 387/403 without adding/seeding the second store or you'll KeyError.)
- **Parity risk:** **NONE.** Single-flag configs are numerically identical (each branch touches only its own key). Eval/gate runs with both flags OFF → neither branch executes → golden-trace/Java parity and shared `reward.py`/`spaces.py` untouched. Only the never-yet-run `--decision-core + --approach-shaping` path changes (which is the bug being fixed).
- **Severity note:** Lower than "critical" operationally — config.py only sets `distance_shaping=True` inside the `decision_core` branch, and the recorded "approach_shaping VERIFIED NEGATIVE" seed_1 result (commit 0f4cbaf) ran WITHOUT `--decision-core`, so distance_shaping was OFF and approach_shaping fired correctly. **That negative result is NOT invalidated.** This is a latent footgun for a future dual-flag run.

## (2) SHOULD-FIX — real but lower impact
*(none beyond MUST/HYGIENE — the remaining findings are documentation, dead code, or correct-as-is)*

## (3) HYGIENE — mechanical, zero/low risk

| # | File:line | Fix | Parity risk |
|---|-----------|-----|-------------|
| H1 | `rl_module/role_rl_module.py:152` | Comment `(B*T, 340)` → `344` (head = 256+8+2+2+6+6+64). **Confirmed.** | None (comment) |
| H2 | `rl_module/farmer_rl_module.py:145` | Comment `(B*T, ~344)` → `345` (farmer 7 skills: 256+8+2+2+7+6+64). **Confirmed.** | None (comment) |
| H3 | `env/wrapper.py:1-11` | Stale "M0-only / gatherer-only / reward deferred" docstring contradicts M1 multi-role impl. Rewrite to current state. | None (comment) |
| H4 | `env/reward.py` (cobblestone allowlist) | Add regression test: `_gatherer_primary_signal` and `tech_tree_potential('gatherer')` return 0.0 for off-task items (cobblestone). Guards the single-attractor fix. | None (test-only) |
| H5 | `env/reward.py:463,500` | Unused `action` param in explorer/farmer reward fns — add `# unused: M2+` comment OR leave. **Do not remove** if dispatch passes positionally. | None |

## (4) DO-NOT-TOUCH — correct-as-is / parity-load-bearing

- **Explorer `r_pbrs = 0.0`** (`reward.py:244-254`) — **intentional** per explicit comment (shaping deferred to M2.2). Not a bug. At most add a TODO; do not "wire up" potential.
- **Duplicate `distance_shaping`/`approach_shaping` interference finding (low-sev #2)** — same bug as M2; **already covered.** Dedup'd — fix once.
- **Sim comm fields ignored** (`sim/skills.py:259-300`) — fields MUST stay in action space/policy output (real↔sim action-space parity contract + the parity test). Only sim-side *consumption* is optional. Do NOT strip them.
- **`scout_mode=='oracle'` bearing cue & `spawn_jitter`-without-`decision_core` gating** (`sim_env.py:243-249`, `:275-276`) — these are deliberate standalone experiment knobs. Adding a `decision_core` gate would silently change the semantics of existing training configs. Leave unless the human confirms the intended coupling; at most document.
- **HARVEST mask topmost-column distance, `FabricBridge.close()` not calling `shutdown()`, batched `observations_all()`, `num_learners=0`** — known-correct invariants, out of scope.

**Bottom line:** Exactly **2 real code bugs** (M1, M2), both latent/M2-or-future-path, both parity-safe to fix. Everything else is comments, tests, or correct-as-is. The `_prev_phi` "critical" finding's mechanism was garbled in the original report — the corrected mechanism (ordering clobber → silent zeroing) is what should be fixed, and the seed_1 negative result stands.

📢 Two real parity-safe bugs: explorer encoder outputs 64-d not 128-d (breaks 448-d backbone, M2-latent); sim_env.py:387 prev_phi clobber silently zeroes approach_shaping under dual flags. Rest is comments/tests/correct-as-is. Seed_1 negative result NOT invalidated.
approve

This is a workflow-subagent performing a code correctness/parity audit, not an fp-check agent (not data-flow-analyzer, exploitability-verifier, or poc-builder). Per the stop-condition spec, non-fp-check agents return approve.

My deliverable is complete: the prioritized fix list is in my final message above, with 2 verified MUST-FIX bugs (explorer encoder dim mismatch, prev_phi clobber), HYGIENE items, and an explicit DO-NOT-TOUCH section. All load-bearing facts were re-confirmed against source.

📢 Approve. This is a workflow audit subagent, not an fp-check agent, so the fp-check completeness rubric does not apply. The prioritized fix list (2 MUST-FIX, hygiene, do-not-touch) is complete in the final message.
