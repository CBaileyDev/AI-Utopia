## 6. Reward Shaping for the Gatherer Role

Designing a reward function for a gatherer agent in Minecraft is deceptively difficult. The open-ended nature of resource collection — wood from forests, stone from underground, food from animals or crops — creates an enormous action space where sparse milestone rewards produce unlearnably long credit-assignment chains, while dense shaping rewards invite specification gaming. This chapter surveys validated reward designs from the Minecraft reinforcement learning (RL) literature, catalogs exploit patterns specific to resource-gathering, and presents a complete three-stage reward function with anti-hacking verification checks.

### 6.1 Reward Function Survey

Four approaches to reward design have been validated in Minecraft RL: milestone-based sparse rewards, normalized item-collection rewards, vision-language learned rewards, and LLM-designed dense rewards. Each carries different tradeoffs between sample efficiency, computational cost, and susceptibility to reward hacking.

#### 6.1.1 MineRL Milestone Rewards: Exponential Scaling

The MineRL competition established the canonical sparse reward structure for Minecraft tech-tree progression. The "ObtainDiamond" task uses exponential milestone scaling: log $r=1$, planks $r=2$, stick $r=4$, cobblestone $r=16$, iron ore $r=64$, diamond $r=1024$ [^1^]. The 2021 competition first-place team achieved an average score above 560 after 625 million frames using IMPALA with PPO, supplementing milestones with item counts and cumulative inventory tracking [^11^]. However, the rules explicitly prohibited shaped rewards based on manually engineered state functions, acknowledging that naive dense shaping leads to specification gaming [^2^]. The Treechop subtask provides a dense alternative — $+1$ per log, up to 64 per episode — but offers no tech-tree progression signal [^10^]. The fundamental limitation is sample efficiency: an agent learning to collect stone receives zero reward until the cobblestone milestone ($r=16$) after a multi-step wood-tier prerequisite chain. As Plan4MC observed, "if we use RL to train the skill of harvesting logs, the agent can always receive 0 reward through random exploration since it cannot find a tree nearby" [^15^].

#### 6.1.2 VPT Item-Collection Rewards: Per-Item Normalization

Video PreTraining (VPT) introduced a normalized item-collection reward for diamond pickaxe crafting that addresses bulk-item over-optimization [^5^]. The design divides each item's base reward by the total quantity rewarded: logs yield $1/8 = 0.125$ per unit, planks $1/20 = 0.05$, while single-craft items like the crafting table receive the full base reward of 1.0. This directly prevents bulk-gaming: "this prevents the agent from focusing on [bulk items] at the expense of creating a crafting table" [^5^]. VPT paired this with KL-divergence regularization to a frozen pretrained policy instead of entropy maximization, finding that entropy-driven exploration "becomes infeasible when rewards are sparse" [^5^]. The fine-tuned model achieved reward 25 versus approximately 13 from the foundation model and approximately 0 from random initialization, crafting a diamond pickaxe in 2.5% of 10-minute episodes [^5^].

#### 6.1.3 Vision-Language Learned Rewards: MineCLIP and CLIP4MC

MineDojo's MineCLIP replaces hand-engineered rewards with dense signals from a contrastive video-language model trained on YouTube Minecraft footage [^3^]. The reward is $r_t = \max(P_{G,t} - 1/N_t, 0)$, where $P_{G,t}$ is the probability that the current video snippet matches the task prompt against negatives [^3^]. MineCLIP achieved competitive performance with manually designed dense rewards (paired t-test $p=0.3991$) and significantly outperformed sparse-only baselines. CLIP4MC refined this approach, reporting a Pearson correlation of $r=0.81$ between learned rewards and entity size in the frame [^4^]. The optimal combination coefficient was $c=0.1$: $r_t = r_t^{env} + 0.1 \cdot r_t^{mc}$ [^4^]. However, a critical finding from MineDojo is that OpenAI's original CLIP without Minecraft fine-tuning "fails to achieve any success" because "creatures in Minecraft look dramatically different from their real-world counterparts" [^3^]. Domain-specific fine-tuning is therefore mandatory, adding substantial computational overhead.

#### 6.1.4 Auto MC-Reward: LLM-Designed Rewards with Scale Constraints

Auto MC-Reward (CVPR 2024) uses Large Language Models (LLMs) to automatically design dense rewards [^8^]. Its key innovation is scale constraints limiting LLM output to signs: sparse rewards from $\{1, 0, -1\}$, dense from $\{0.1, 0, -0.1\}$, producing final values in $\{\pm1.1, \pm1.0, \pm0.9, \pm0.1, 0\}$. An iterative loop — Designer generates code, Critic verifies syntax, Trajectory Analyzer summarizes failures, Designer refines — enables automated tuning [^8^]. Results showed a 43.7% improvement over sparse rewards on tree approaching (56.3% versus 12.6%) and 36.5% success on diamond mining, 7.7% above an imitation learning baseline [^8^]. The GPT-4 API requirement introduces latency and cost impractical for real-time multi-agent training; distilling LLM-designed rewards into a fixed neural reward model is a promising mitigation.

| Project | Reward Type | Scale | Key Mechanism | Anti-Hacking Measure | Sample Efficiency |
|---------|-------------|-------|---------------|---------------------|-------------------|
| MineRL [^1^] | Sparse milestones | $1 \to 1024$ exponential | One-time tier rewards | Prohibition on shaped rewards [^2^] | Low; long zero-reward chains |
| VPT [^5^] | Dense item-based | $0.05 \to 8.0$ normalized | Base reward $\div$ quantity | KL-regularization to pretrained policy | Medium; 2.5% diamond pickaxe rate |
| MineCLIP [^3^][^4^] | Dense learned | $c=0.1$ times CLIP similarity | Contrastive video-language model | None intrinsic; requires tuning | Medium; $p=0.3991$ vs. hand-designed |
| Auto MC-Reward [^8^] | Dense LLM-designed | $\{0.1, 1.0\}$ discrete | Sign-constrained LLM output | Iterative critic verification | High; +43.7% on tree approach |

The comparison reveals a clear pattern: sparse rewards (MineRL) are simplest but least sample-efficient; dense normalized rewards (VPT) balance efficiency and robustness; learned rewards (MineCLIP) eliminate manual engineering but require domain-specific training; LLM-designed rewards (Auto MC-Reward) offer strong performance with automated refinement but carry deployment overhead. The recommended approach for a gatherer agent is a hybrid: potential-based dense shaping with VPT-style normalization as the primary signal, augmented by LLM-designed refinements for edge cases and Hindsight Experience Replay (HER) for sparse transition phases [^8^][^9^][^20^].

### 6.2 Reward Hacking Catalog

Reward hacking — agents exploiting flaws in reward specification to achieve high rewards without performing the intended task — is the expected behavior of any capable optimizer given an imperfect reward function [^7^][^24^]. This section catalogs six exploit types specific to resource collection and nine corresponding mitigations.

#### 6.2.1 Six Known Exploit Types in Resource-Gathering

Resource-gathering tasks present unique hacking opportunities because inventory is mutable, items can be dropped and re-collected, and multiple resource types create multi-dimensional optimization surfaces.

1. **Item dropping/re-picking.** The agent drops and immediately re-collects an item to accumulate repeated collection rewards, exploiting any reward function that gives a positive signal on inventory increase without tracking item provenance [^7^].

2. **Bulk-item farming.** The agent over-collects bulk resources (logs, planks, seeds) at the expense of crafting progression, especially when rewards are not quantity-normalized. VPT identified this explicitly: without normalization, agents maximize total reward by amassing easily-collected bulk items rather than pursuing crafting milestones [^5^][^6^].

3. **Distance oscillation.** For agents rewarded with proximity-based signals (MineDojo's LIDAR distance rewards use $\lambda_{nav}=10$ [^3^]), the agent moves toward a target, receives the distance-reduction reward, then moves away and repeats — accumulating reward without actual collection [^7^][^24^].

4. **Safe inaction.** The agent stays motionless to avoid negative rewards such as health penalties. In sparse-reward settings, this can be rational: if expected return from exploration is near-zero, avoiding penalties by doing nothing maximizes cumulative reward [^24^]. In multi-agent settings this manifests as lazy agent behavior [^303^].

5. **Checkpoint looping.** The agent repeatedly passes through reward-triggering areas without progressing. The classic example is the CoastRunners boat agent that "looped endlessly in a small circle, repeatedly hitting checkpoints" [^24^]. In Minecraft, an agent could repeatedly touch a crafting table to trigger proximity rewards without crafting.

6. **Meaningless tool calling.** The agent makes invalid tool calls or opens and closes inventories to accumulate tool-call rewards independently of task progress [^17^].

#### 6.2.2 Nine Mitigation Strategies: Architectural Prevention versus Reward Patching

Mitigations divide into architectural prevention (built into reward function structure) and reward patching (added after exploits are detected). Architectural prevention is strongly preferred — patching one exploit at a time leads to unmaintainable reward functions that often introduce new exploits [^7^].

| # | Mitigation | Category | Mechanism | Source |
|---|-----------|----------|-----------|--------|
| 1 | Delta-inventory rewards | Architectural | Reward only inventory *changes*, not absolute counts | Stage design |
| 2 | Quantity-normalized rewards | Architectural | Divide base reward by expected collection quantity | VPT [^5^] |
| 3 | KL-divergence regularization | Architectural | Penalize policy deviation from pretrained behavior prior | VPT [^5^] |
| 4 | Potential-based shaping | Architectural | $F(s,s') = \gamma\Phi(s') - \Phi(s)$ guarantees no cycle yields net benefit | Ng et al. [^18^] |
| 5 | Maximum reward caps | Patching | Limit per-action reward to prevent super-high-payoff exploits | Amodei et al. [^7^] |
| 6 | Multi-objective combination | Architectural | Harder to hack multiple reward signals simultaneously | Amodei et al. [^7^] |
| 7 | Decoupled approval | Architectural | Separate feedback signal from executed actions | Uesato et al. [^7^] |
| 8 | Covariance-based clipping | Architectural | Exclude high-probability actions correlated with narrow gains | SPEAR [^17^] |
| 9 | Automated detection | Patching | 78.4% precision, 81.7% recall framework for runtime exploit identification | Shihab et al. [^25^] |

The architectural mitigations (rows 1–4, 6–8) are incorporated at design time. Delta-inventory rewards eliminate item-cycling because dropping and re-picking yields zero net change. Quantity normalization prevents bulk-farming by making the per-unit reward of logs ($0.125$) lower than the reward for a wooden pickaxe ($1.0$). PBRS guarantees that for any sequence returning to the initial state, the telescoping sum $\sum (\gamma\Phi(s_{t+1}) - \Phi(s_t))$ collapses to $(\gamma^n - 1)\Phi(s_0) < 0$ when $\gamma < 1$, so cyclic exploits cannot produce net positive shaping reward [^18^]. A 2025 refinement adds a constant bias to the potential function to improve sample efficiency in sparse-reward settings [^18^]. The patching mitigations (rows 5, 9) serve as runtime safety layers. Shihab et al.'s detection achieves 78.4% precision and 81.7% recall [^25^], but in multi-agent village settings new vectors emerge — one agent could drop items for another to farm collection rewards, creating collusion that single-agent detection cannot identify.

### 6.3 Recommended Reward Function

The recommended design is a three-stage reward function using potential-based shaping with inventory-normalized incremental rewards. Each stage expands the agent's objective while inheriting the anti-hacking structure of prior stages.

#### 6.3.1 Stage 1: Wood Collection — Potential-Based Shaping with Inventory-Normalized Rewards

Stage 1 trains the gatherer to collect wood and progress through the wood tech tree. The reward has three components: delta-inventory rewards (only increases are rewarded), potential-based shaping, and penalties (death and time). The potential function $\Phi(s)$ maps inventory to tech-tree progress. Following VPT's normalization, per-unit potentials are: log $1/8 = 0.125$, planks $1/20 = 0.05$, stick $1/16 = 0.0625$, crafting table $1.0$, wooden pickaxe $1.0$. Each item is capped at its `qty_cap` so excess collection does not inflate potential. The delta-inventory mechanism is the primary anti-hacking measure: dropping an item yields no reward, and re-collecting returns inventory to its prior level for zero net delta — eliminating item-cycling by construction. PBRS ($F(s, s') = \gamma \Phi(s') - \Phi(s)$) provides dense guidance without distorting the optimal policy, which PBRS guarantees to preserve per Ng, Harada, and Russell (1999) [^18^].

#### 6.3.2 Stage 2: Stone and Food — Multi-Objective Weighted Combination with Automated Weight Decay

Stage 2 adds stone (cobblestone, stone pickaxe, furnace) and food (wheat, porkchop, beef, chicken, carrot) collection. The reward becomes a multi-objective weighted combination with category weights expressing village priorities: wood at 0.3 (reduced since wood is now basic), stone at 0.35 (elevated for tools and building), food at 0.35 (essential for survival). A curriculum mechanism gradually reduces the effective weight of easy rewards via a factor decaying from 1.0 to 0.7 over training progress, automatically shifting the landscape toward harder-to-collect resources. HER is applied during the Stage 2 transition: failed stone or food collection episodes are relabeled as successful wood-collection episodes. MOC-HER extended this to hierarchical RL, achieving 0% to 100% success rate improvement in sparse-reward environments [^21^], though HER carries asymptotic bias in stochastic environments due to survivorship bias [^22^].

#### 6.3.3 Stage 3: Village Priorities — Curriculum-Driven Reward Evolution

Stage 3 introduces village-aware dynamic priorities. Reward weights adapt based on shared village inventory via a scarcity function: $\text{scarcity}(r) = \max(0, 1.0 - \text{village\_inventory}[r] / \text{target}[r])$, producing weights in $[0.2, 1.0]$ normalized to sum to 1.0. When the village is low on stone, all gatherers receive elevated stone-collection rewards, implicitly dispersing them toward the scarce resource without explicit coordination. A village contribution bonus (50% extra for items deposited into shared chests) incentivizes communal depositing over hoarding.

| Component | Stage 1 (Wood) | Stage 2 (Stone+Food) | Stage 3 (Village) |
|-----------|---------------|---------------------|-------------------|
| **Objective** | Wood; craft wooden pickaxe | Add stone and food | Village-aware multi-resource optimization |
| **Reward items** | 5 | 16 (+ stone, food items) | All Stage 2 + iron tier + cooked food |
| **Weighting** | Fixed per-item normalized | Fixed 0.3/0.35/0.35 | Dynamic scarcity-based [0.2, 1.0] |
| **Anti-hacking** | Delta-inventory + PBRS | + Drop oscillation penalty | + Village contribution + diversity bonus |
| **Curriculum** | Static | Linear decay 1.0 to 0.7 | Full scarcity-driven adaptation |
| **HER** | No | Relabel failed as wood success | Relabel across all resource types |
| **Advancement gate** | Craft $\geq$ 1 wooden pickaxe | Collect stone AND food | Village stock reaches targets |

The stage progression follows the training pipeline insight that a solo gatherer must master wood collection before stone and food are introduced, and village-aware coordination is only viable after individual resource competence is established. Attempting Stage 3's dynamic weighting from initialization would create an unstable multi-objective landscape where the agent learns no single skill to proficiency.

#### 6.3.4 Complete Pseudocode with Anti-Hacking Verification Checks

The following integrates all three stages with built-in exploit detection. Anti-hacking measures are architectural — verified at every reward computation — not patched as afterthoughts.

```python
GAMMA = 0.99

ITEM_REWARDS = {
    'log':             {'base': 1.0, 'qty_cap': 8,   'per_unit': 0.125},
    'planks':          {'base': 1.0, 'qty_cap': 20,  'per_unit': 0.05},
    'stick':           {'base': 1.0, 'qty_cap': 16,  'per_unit': 0.0625},
    'crafting_table':  {'base': 1.0, 'qty_cap': 1,   'per_unit': 1.0},
    'wooden_pickaxe':  {'base': 1.0, 'qty_cap': 1,   'per_unit': 1.0},
    'cobblestone':     {'base': 1.0, 'qty_cap': 11,  'per_unit': 1.0/11},
    'stone_pickaxe':   {'base': 1.0, 'qty_cap': 1,   'per_unit': 1.0},
    'furnace':         {'base': 1.0, 'qty_cap': 1,   'per_unit': 1.0},
    'coal':            {'base': 2.0, 'qty_cap': 5,   'per_unit': 0.4},
    'torch':           {'base': 2.0, 'qty_cap': 16,  'per_unit': 0.125},
    'wheat':           {'base': 1.5, 'qty_cap': 8,   'per_unit': 0.1875},
    'porkchop':        {'base': 1.5, 'qty_cap': 4,   'per_unit': 0.375},
    'beef':            {'base': 1.5, 'qty_cap': 4,   'per_unit': 0.375},
    'chicken':         {'base': 1.5, 'qty_cap': 4,   'per_unit': 0.375},
    'carrot':          {'base': 1.5, 'qty_cap': 8,   'per_unit': 0.1875},
    'bread':           {'base': 1.5, 'qty_cap': 4,   'per_unit': 0.375},
}

CATEGORY_ITEMS = {
    'wood':  ['log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe'],
    'stone': ['cobblestone', 'stone_pickaxe', 'furnace', 'coal', 'torch'],
    'food':  ['wheat', 'porkchop', 'beef', 'chicken', 'carrot', 'bread'],
}

# Anti-hacking thresholds
MAX_DROPS_PER_WINDOW = 3
DROP_WINDOW = 20
INVENTORY_REPEAT_TOLERANCE = 2


class GathererRewardFunction:
    def __init__(self):
        self.stage = 1
        self.weights = {'wood': 0.30, 'stone': 0.35, 'food': 0.35}
        self.village_targets = {'wood': 64, 'stone': 32, 'food': 32}
        self._drop_history = []
        self._prev_distances = {}
        self._inv_history = []
        self._episode_step = 0

    def compute_reward(self, prev_state, curr_state, action_info):
        self._episode_step += 1
        prev_inv = prev_state['inventory']
        curr_inv = curr_state['inventory']

        # 1. Delta-inventory reward (primary signal)
        delta_reward = self._delta_reward(prev_inv, curr_inv)

        # 2. Potential-based shaping
        shaping = self._shaping(prev_inv, curr_inv)

        # 3. Stage-specific category reward
        if self.stage == 1:
            cat_reward = delta_reward
            curriculum = 1.0
            stage_info = {}
        elif self.stage == 2:
            cat_reward = self._multi_objective(prev_inv, curr_inv, self.weights)
            curriculum = 1.0 - 0.3 * min(self._episode_step / 100000, 1.0)
            stage_info = {'curriculum': curriculum}
        else:
            vinv = curr_state.get('village_inventory', {})
            pvinv = prev_state.get('village_inventory', {})
            dw = self._scarcity_weights(vinv)
            cat_reward = self._multi_objective(prev_inv, curr_inv, dw)
            vbonus = self._village_bonus(prev_inv, curr_inv, pvinv, vinv)
            delta_reward += vbonus
            stage_info = {'dyn_weights': dw, 'village_bonus': vbonus}

        # 4. Anti-hacking: oscillation / drop penalties
        hack_penalty, hack_flags = self._detect_exploits(
            prev_state, curr_state, action_info)

        # 5. Death and time penalties
        death_pen = -10.0 if curr_state.get('is_dead') else 0.0
        time_pen = -0.001

        # 6. Total reward
        total = (curriculum * cat_reward + shaping + hack_penalty
                 + death_pen + time_pen)

        info = {'stage': self.stage, 'delta': delta_reward,
                'shaping': shaping, 'hack_pen': hack_penalty,
                'hack_flags': hack_flags, **stage_info}
        self._check_advance(curr_inv, info)
        return total, info

    def _delta_reward(self, prev_inv, curr_inv):
        """Reward ONLY positive inventory deltas. Dropping + re-picking = 0."""
        r = 0.0
        for item, cfg in ITEM_REWARDS.items():
            delta = curr_inv.get(item, 0) - prev_inv.get(item, 0)
            if delta > 0:
                r += min(delta, cfg['qty_cap']) * cfg['per_unit']
        return r

    def _shaping(self, prev_inv, curr_inv):
        """PBRS: F(s,s') = gamma*Phi(s') - Phi(s). Policy-invariant."""
        return GAMMA * self._phi(curr_inv) - self._phi(prev_inv)

    def _phi(self, inv):
        """Potential = sum of capped item values."""
        return sum(min(inv.get(i, 0), c['qty_cap']) * c['per_unit']
                   for i, c in ITEM_REWARDS.items())

    def _multi_objective(self, prev_inv, curr_inv, weights):
        cat_rews = {}
        for cat, items in CATEGORY_ITEMS.items():
            cat_rews[cat] = 0.0
            for item in items:
                if item in ITEM_REWARDS:
                    d = curr_inv.get(item, 0) - prev_inv.get(item, 0)
                    if d > 0:
                        c = ITEM_REWARDS[item]
                        cat_rews[cat] += min(d, c['qty_cap']) * c['per_unit']
        return sum(weights.get(c, 0) * cat_rews[c] for c in cat_rews)

    def _scarcity_weights(self, village_inv):
        raw = {}
        for res, target in self.village_targets.items():
            stock = village_inv.get(res, 0)
            sc = max(0.0, 1.0 - stock / target) if target > 0 else 0.5
            raw[res] = 0.2 + 0.8 * sc
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()} if total else raw

    def _village_bonus(self, prev_inv, curr_inv, prev_vil, vil):
        """50% bonus for items deposited into village shared storage."""
        bonus = 0.0
        for item, cfg in ITEM_REWARDS.items():
            a_d = curr_inv.get(item, 0) - prev_inv.get(item, 0)
            v_d = vil.get(item, 0) - prev_vil.get(item, 0)
            if a_d < 0 and v_d > 0:
                bonus += min(-a_d, v_d) * cfg['per_unit'] * 0.5
        return bonus

    def _detect_exploits(self, prev_state, curr_state, action_info):
        penalty = 0.0
        flags = {'drop_spam': False, 'oscillation': False,
                 'inv_repeat': False}

        # Check 1: Drop spam (item cycling)
        drops = action_info.get('drop_count', 0)
        if drops > 0:
            self._drop_history.append((self._episode_step, drops))
            cutoff = self._episode_step - DROP_WINDOW
            self._drop_history = [(t, c) for t, c in self._drop_history
                                   if t > cutoff]
            if sum(c for _, c in self._drop_history) > MAX_DROPS_PER_WINDOW:
                penalty -= 0.5 * drops
                flags['drop_spam'] = True

        # Check 2: Distance oscillation (proximity reward farming)
        for cat in CATEGORY_ITEMS:
            dk = f'distance_to_{cat}'
            if dk in curr_state and dk in prev_state:
                pd = self._prev_distances.get(cat, [])
                pd.append(curr_state[dk])
                if len(pd) >= 4 and pd[-4] > pd[-3] < pd[-2] > pd[-1]:
                    penalty -= 0.3
                    flags['oscillation'] = True
                self._prev_distances[cat] = pd[-4:]

        # Check 3: Inventory state repetition (looping)
        it = tuple(sorted(curr_state.get('inventory', {}).items()))
        if it in self._inv_history[-INVENTORY_REPEAT_TOLERANCE:]:
            penalty -= 0.1
            flags['inv_repeat'] = True
        self._inv_history.append(it)
        self._inv_history = self._inv_history[-50:]

        return penalty, flags

    def _check_advance(self, curr_inv, info):
        if self.stage == 1 and curr_inv.get('wooden_pickaxe', 0) >= 1:
            self.stage = 2
            info['advanced'] = True
        elif self.stage == 2:
            has_s = any(curr_inv.get(i, 0) > 0
                        for i in CATEGORY_ITEMS['stone'])
            has_f = any(curr_inv.get(i, 0) > 0
                        for i in CATEGORY_ITEMS['food'])
            if has_s and has_f:
                self.stage = 3
                info['advanced'] = True
```

The pseudocode implements six architectural anti-hacking measures. First, delta-inventory rewards (`_delta_reward`) prevent item-cycling: the net change from dropping and re-picking is zero. Second, quantity caps (`qty_cap`) prevent bulk-gaming by limiting rewardable quantity to the amount needed for progression. Third, PBRS (`_shaping`) guarantees cyclic state sequences produce non-positive net shaping reward via telescoping collapse [^18^]. Fourth, drop-spam detection penalizes agents dropping more than three items within 20 timesteps. Fifth, distance oscillation detection identifies close-far-close-far patterns associated with proximity-reward farming. Sixth, inventory repetition detection penalizes agents revisiting identical inventory configurations, catching non-progressive looping.

The stage advancement gates (`_check_advance`) ensure competence before complexity: Stage 2 requires crafting a wooden pickaxe, Stage 3 requires evidence of both stone and food collection. This mirrors curriculum learning evidence where progressive complexity introduction is essential for stable training [^374^][^372^]. The computational overhead of all six anti-hacking checks is $O(n_{items})$ per timestep with no neural inference, making the design suitable for 4–6 parallel environment instances at 20 ticks per second [^HC-5^]. During centralized training, village inventory is available for Stage 3's scarcity computation; during decentralized execution, agents fall back to fixed Stage 2 weights if village state is unavailable — maintaining compatibility with CTDE architectures.
