# M2 MARL/CTDE Integration Notes — Farmer + Gatherer

**Audience:** multi-agent-rl-specialist, deep-rl-training-specialist  
**Context:** AiUtopia M1B trained Gatherer solo under MAPPO + CTDE critic. M2 adds Farmer (parallel) under same MAPPO, but Farmer's delayed rewards may stress the shared critic. This note outlines the integration contract and magnitudes.

---

## 1. MAPPO + CTDE Architecture Recap

Current M1B stack:
- **PPO per role:** Each role (Gatherer, soon Farmer) has its own `policy_mapping_fn` → separate `GathererPolicyVersion` / `FarmerPolicyVersion`
- **Shared critic:** One `CTDECritic` takes concatenated obs (all agents) and outputs a single value estimate V(s)
- **Per-agent LSTM:** Role RLModule has LSTM time-stepping; critic sees flattened sequence of agent observations (no LSTM in critic)

For M2 with Farmer:

```python
policies = {
    "gatherer_policy": GathererRoleRLModule(...),
    "farmer_policy": FarmerRoleRLModule(...),
}
policy_mapping_fn = lambda agent_id: "gatherer_policy" if agent_id == "gatherer_0" else "farmer_policy"
critic = CTDECritic(obs_shape=merged_obs_shape)
```

---

## 2. Reward Signal Alignment

### 2.1 The problem: different timescales

**Gatherer:**
- r_principal (oak_log harvest): per-tick, sparse but semi-frequent (1–5 per episode)
- r_pbrs (tech_tree_potential): per-tick, smooth decay
- r_total: typically 10–50 per episode (at M1B gate, >50)

**Farmer:**
- r_principal (crop harvest): per-tick, sparse, delayed (1–5 per episode, arrives tick 64+)
- r_pbrs (farmer_potential with decay): per-tick, smooth, decays to zero
- r_total: typically 20–40 per episode (gate is 32 harvests = +32 r_principal)

**Why this matters:** If the shared critic trains on Gatherer's fast, frequent rewards (convergence by iter 20), it may over-fit to Gatherer's value landscape. Farmer's sparse, delayed signals then get weak gradients because the critic learned "value ≈ inventory_count" (Gatherer's pattern), not "value ≈ ripeness progress" (Farmer's pattern). The critic becomes Gatherer-biased.

### 2.2 Equal update weight per role (critical constraint)

**Requirement:** In the PPO learner, ensure both roles get **equal sample weight** when updating the critic.

```python
# Ray RLlib config:
config = PPOConfig()
    .training(
        # Both roles updated together, equal steps each
        sgd_minibatch_size=256,
        train_batch_size=4096,  # total; ~50% from each role
        # ...
    )
```

**Not:** feeding the learner 1000 Gatherer steps + 100 Farmer steps per batch (weights Gatherer 10×). The critic must learn both value landscapes equally.

**Validation:** After iter 10–20, check `learner/value_loss` per role (if Ray exposes per-policy losses). Both should be converging at similar rates.

---

## 3. Magnitude Scaling: The Core Fix

### 3.1 Potential-based PBRS per agent

From Ng et al. 1999, for a single agent the PBRS term is policy-invariant:

```
r_pbrs = γ · Φ(s') - Φ(s)
```

This holds *per agent independently*. In MARL with a shared critic:

```
r_total = Σ_i [ r_principal_i + r_pbrs_i ] - r_shared_penalties
```

The critic learns:

```
V(s) = E[ Σ_{t'≥t} γ^(t'-t) · r_total_{t'} ]
```

Each agent's Φ contribution feeds the critic's regression target. **If Φ_gatherer >> Φ_farmer in magnitude, the critic's value estimates will be dominated by Gatherer's potential, and Farmer's shaping gradient will be ~noise.**

### 3.2 Magnitude audit

**Gatherer's tech_tree_potential (from `reward.py`):**
- Bounded by `ROLE_INVENTORY_CAPS["gatherer"]` (e.g., oak_log cap 256)
- Typical φ = 0.5 × cap × LOG_VALUE = 0.5 × 256 × 1.0 = 128
- Range: 0–200 typical

**Farmer's farmer_potential (from Farmer spec):**
- Three terms, each 0–1, weighted 0.15/0.50/0.35
- Sum before decay ≈ 0–1
- **Unscaled:** 0–1 (100× smaller than Gatherer!)
- **Scaled (proposed):** multiply by 100 → 0–100 (matches Gatherer)

### 3.3 The fix: Scale Farmer's potential

In `reward.py` `farmer_potential()`:

```python
def farmer_potential(obs: dict, decay_coeff: float = 1.0) -> float:
    """Φ(s) for Farmer PBRS. Range [0, 100] to match gatherer_potential."""
    
    planted_progress = min(1.0, obs["f_planted_count"][0] / 64.0)
    ripeness_progress = obs["f_ripeness"][0]
    
    ripe_cells = obs["f_crop_grid"] == 8
    if ripe_cells.any():
        staleness = np.minimum(1.0, obs["f_time_at_ripeness"][ripe_cells] / 50.0)
        timeliness = np.mean(1.0 - staleness)
    else:
        timeliness = 0.0
    
    phi_unscaled = (
        0.15 * planted_progress +
        0.50 * ripeness_progress +
        0.35 * timeliness
    ) * decay_coeff
    
    # Scale to [0, 100] to match tech_tree_potential's magnitude
    return 100.0 * phi_unscaled
```

**After scaling:** Φ_farmer ≈ 0–100, same order of magnitude as Φ_gatherer ≈ 0–200. The critic's per-role PBRS contributions are now comparable, not 100× different.

### 3.4 Validation check

After first 5 training iters in M2, measure:

```python
# Pseudo-code for Ray Tune callback
def check_magnitude_balance(info, trial):
    if trial.iteration <= 5:
        return
    
    result = info["result"]
    
    # Extract per-role metrics (if Ray exposes them)
    gatherer_rewards = result.get("custom_metrics/gatherer_episode_return_mean", 0)
    farmer_rewards = result.get("custom_metrics/farmer_episode_return_mean", 0)
    
    # Both should be in a similar range (±2× is fine)
    ratio = farmer_rewards / (gatherer_rewards + 1e-6)
    
    if ratio < 0.1 or ratio > 10:
        logger.warning(f"Magnitude imbalance: Farmer/Gatherer = {ratio:.1f}")
        # Suggest: re-check farmer_potential scaling
    else:
        logger.info(f"Magnitude balance OK: Farmer/Gatherer = {ratio:.1f}")
```

---

## 4. Credit Assignment: Value Function Fit

### 4.1 The risk: Critic over-fits to Gatherer

**Scenario:**
- Iter 1–10: both roles collect data equally
- Iter 10–20: Gatherer converges (policy learns to harvest oak_log consistently)
- Farmer: still exploring (policy hasn't learned patience yet; reward is 0 for most episodes)

**Critic's learning signal:**
- Gatherer data: consistent returns ≈ 30–50, value function learns "state_dict → 40"
- Farmer data: sparse returns ≈ 0 for 50% of episodes, 30–50 for the rest; value function is confused
- Net: Critic prioritizes fitting Gatherer's clean signal; Farmer's signal is noise

**Downstream:** Farmer's value estimates are poor → PPO's advantage estimates are noisy → Farmer's policy gradients are weak → convergence stalls.

### 4.2 Mitigations

#### 4.2.1 Equal minibatch composition

Ensure the PPO learner pulls equal amounts from both roles per SGD step:

```python
config = PPOConfig()
    .training(
        train_batch_size=4096,  # must be divisible by n_roles
        sgd_minibatch_size=128,  # per SGD batch
    )
# Ray will internally shuffle the batch to mix both roles equally
```

#### 4.2.2 Per-role value function clipping

If Ray exposes per-policy losses, clip value loss separately per role:

```python
# In RLModule or custom loss callback
v_loss_gatherer = nn.MSELoss()(v_pred_gatherer, v_target_gatherer)
v_loss_farmer = nn.MSELoss()(v_pred_farmer, v_target_farmer)

# Clip large losses to prevent one role from dominating
v_loss_gatherer_clipped = min(v_loss_gatherer, threshold=10.0)
v_loss_farmer_clipped = min(v_loss_farmer, threshold=10.0)

total_loss = v_loss_gatherer_clipped + v_loss_farmer_clipped
```

#### 4.2.3 Monitor per-role returns separately

Add custom metrics to Ray Tune:

```python
# In callback
episode_returns = {}
for agent_id, episode in episodes.items():
    role = "gatherer" if "gatherer" in agent_id else "farmer"
    if role not in episode_returns:
        episode_returns[role] = []
    episode_returns[role].append(episode.total_reward)

for role, returns in episode_returns.items():
    trial.report_custom_metrics({
        f"episode_return_mean/{role}": np.mean(returns),
        f"episode_return_std/{role}": np.std(returns),
    })
```

**Goal:** Both roles' `episode_return_mean` should be increasing (or flat, if converged). If Farmer's is stuck near 0 while Gatherer's is 50+, credit assignment is degraded.

---

## 5. PBRS Policy-Invariance in CTDE Context

### 5.1 Single-agent PBRS (policy-invariant proof)

For a single agent:

```
r_new = r_old + γ · Φ(s') - Φ(s)
V^π(s) = E_π [ Σ_{t≥0} γ^t r(s_t) ]
V^π_new(s) = E_π [ Σ_{t≥0} γ^t (r_old(s_t) + γ Φ(s_{t+1}) - Φ(s_t)) ]
           = E_π [ Σ_{t≥0} γ^t r_old(s_t) ] + E_π [ Σ_{t≥0} γ^t (γ Φ(s_{t+1}) - Φ(s_t)) ]
           = V^π_old(s) + E_π [ Σ_{t≥0} γ^t (γ Φ(s_{t+1}) - Φ(s_t)) ]
```

The second term is a **telescoping sum** that cancels out (in the limit, or with discounted infinite horizon). Thus:

```
V^π_new(s) ≈ V^π_old(s) + constant
```

The optimal policy π* is unchanged because the value function differences (used by TD learning) are invariant to the constant shift.

### 5.2 Multi-agent CTDE extension

In CTDE, this holds **per agent independently**, as long as each agent's Φ is defined over its own state features, not shared state. In our case:

- Φ_gatherer(obs_gatherer) — based on inventory + resource grid
- Φ_farmer(obs_farmer) — based on crop grid + ripeness

Neither depends on the other's state, so PBRS remains policy-invariant per role.

**Potential issue (NOT a PBRS issue, but a critic issue):** The shared critic's value target becomes:

```
v_target = r_gatherer_principal + r_gatherer_pbrs + r_farmer_principal + r_farmer_pbrs - r_shared_penalties + γ V(s')
```

The critic is being asked to predict this sum. If the two roles' PBRS contributions are at wildly different scales (100× difference), the critic's gradient will be dominated by the larger scale. **This is not a PBRS policy-invariance problem** (PBRS still holds per agent), **it's a critic numerical-stability problem** (solved by magnitude scaling in §3).

### 5.3 Validation

To verify PBRS still holds in CTDE:
1. Train Gatherer solo (M1B baseline)
2. Train Gatherer + dummy Farmer (Farmer does WAIT only, never changes state)
3. Compare Gatherer's policy in both runs at iter 50

Expected: Gatherer's policy is identical (same action distributions). If Gatherer's policy changed because Farmer was added, the critic's value estimates are off and the policy gradients are corrupt (not a PBRS issue; a critic/learner issue).

---

## 6. Curriculum Checkpoint: When to Add Farmer

### 6.1 M1B → M2 readiness criteria

Before adding Farmer to MAPPO, confirm:
1. **Gatherer converges** to M1B gate (80% success on 64-oak_log in 1000 ticks) in <200 iters (real MC: ~4 hours; sim: ~2 min)
2. **No reward-up/eval-flat bug** in M1B (training reward increases = eval reward increases)
3. **Exploit-free** (no cobblestone attractor, no idle farming, etc.)

If M1B is clean, Farmer addition is **low-risk**: Farmer's reward signal is independent of Gatherer's; it won't corrupt Gatherer's gradient. The shared critic might need tuning (magnitude scaling + per-role clipping), but the policy update should be stable.

### 6.2 If things break in M2

**Symptom: Farmer learns 0 reward, Gatherer unaffected**
- Cause: Critic prioritizes Gatherer (critic value loss decreased, Farmer value loss flat)
- Fix: (a) check equal minibatch weight, (b) check magnitude scaling, (c) increase farmer learning rate 2×

**Symptom: Both roles' returns drop by 50%**
- Cause: Magnitude scaling too aggressive (farmer_potential scaled 1000×, now dominates)
- Fix: Scale farmer_potential down (e.g., 50 instead of 100) and re-run

**Symptom: Gatherer's convergence slows from iter 50 to iter 100**
- Cause: Critic now has to fit both roles; gradient noise is higher
- Fix: Increase `train_batch_size` 4096 → 8192 (more data per update, less noise)

---

## 7. Checklist for M2 MARL Integration

- [ ] **Magnitude scaling:** farmer_potential scaled ×100 (§3.3)
- [ ] **Equal batch weight:** train_batch_size / 2 per role, no role bias (§4.2.1)
- [ ] **Per-role metrics:** episode_return_mean/{gatherer,farmer} exported to Ray Tune (§4.2.3)
- [ ] **Critic fit validation:** Both roles' value_loss converging equally (§4.2.2)
- [ ] **PBRS check:** Gatherer solo vs Gatherer+dummy Farmer; policies match (§6.3)
- [ ] **M1B baseline:** Confirm Gatherer converges in M1B before adding Farmer
- [ ] **First M2 run:** Run with `--evaluation-interval 5` (check every 5 iters for signs of trouble)
- [ ] **Rollback plan:** If M2 breaks, revert to M1B Gatherer solo (low overhead)

---

## 8. References

- **Ng et al. 1999:** "Policy Invariance Under Reward Transformations" (PBRS foundation)
- **Foerster et al. 2017:** "Counterfactual Multi-Agent Policy Gradients" (CTDE, COMA)
- **AiUtopia IMPLEMENTATION_PLAN §7.2:** RLModule architecture, CTDE critic definition
- **AiUtopia M1B training notes:** Per-role LSTM time-stepping, shared critic wiring

---

**This note is implementation-ready.** Multi-agent RL specialist should use §3 (magnitude scaling) and §4 (credit assignment validation) as the primary integration points.
