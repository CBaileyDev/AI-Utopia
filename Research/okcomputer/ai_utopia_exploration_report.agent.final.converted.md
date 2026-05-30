# Sparse-Reward Exploration Learning & Multi-Agent Architecture
# Decision Report for AI-Utopia


## 1. Sparse-Reward Exploration Learning

When extrinsic rewards are sparse, the agent must generate its own learning signal. This section evaluates intrinsic motivation methods, ranks them for fast-sim PPO, provides a stabilization recipe for masked multi-head action spaces, and assesses BC warm-start as an exploration bootstrap.

### 1.1 Intrinsic Motivation Methods

**RND** ^1^measures novelty as prediction error of a trainable network matching a fixed random target on the current observation. The deterministic target makes RND robust to the Noisy-TV problem by construction — it captures only epistemic uncertainty, not environment noise ^1^. Overhead is minimal (~1.1x, one forward pass at ~11 learner steps/sec) ^2^. RND requires dual value heads ($V_E$, $V_I$) with $\gamma_E = 0.999$, $\gamma_I = 0.99$ ^3^. Its global novelty bonus fails in procedurally-generated environments where states are never revisited ^4^, and without RMS normalization achieves zero reward ^5^. On MiniGrid-DoorKey-8x8, tuned RND reaches 82% success ^5^.

**ICM** ^6^uses forward dynamics prediction error as the exploration bonus. It cannot distinguish unpredictable transitions due to novelty from those due to stochasticity, making it susceptible to the Noisy-TV problem ^6^. Under partial observability, the forward model predicts from incomplete information, producing noisy intrinsic rewards prone to "detachment" — when bonuses decay, the agent abandons exploration frontiers ^7^. ICM scored 83% on MiniGrid-DoorKey-8x8 ^5^but failed on 5 of 8 DM-HARD-8 long-horizon tasks ^8^.

**NGU / Agent57** ^9^combines episodic novelty (k-NN within episodes) with lifelong novelty (RND) via a UVFA learning 32 distinct policies. Agent57 surpassed human baseline on all 57 Atari games but required 78 billion frames with 256 distributed actors ^10^— architecturally incompatible with minutes-to-train fast-sim.

**BYOL-Explore** ^8^predicts its own future latent representation and uses prediction error as intrinsic reward. Robust to pixel-level noise — it solves 5.5/8 DM-HARD-8 tasks ^8^— but fails when stochasticity operates at the latent level ^11^. Compute matches RND ^2^. With 16-block visibility, latent predictions must extrapolate far beyond the observation horizon, making this method better suited to pixel inputs than vector states.

**Count-based and RE3.** Position-based counting tracks visited $(x, y)$ and rewards with $\beta / \sqrt{N(s)}$ ^12^. For low-dimensional vector observations, state count outperforms ICM and maximum entropy methods ^4^. RE3 ^13^uses a fixed random encoder (no gradients) with k-NN entropy estimation at ~1.05x overhead, achieving 95% on MiniGrid-DoorKey-8x8 — above RND (82%) and ICM (83%) ^5^.

### 1.2 Episodic Memory and Frontier-Based Exploration

**Episodic Curiosity (EC)** ^14^uses a learned reachability network measuring novelty via environment-step distance from episodic memory, overcoming the "couch potato" problem ^15^. PPO+EC is 1.84x slower and adds 13M parameters ^16^; agents also preferentially explore room corners and blind alleys ^17^.

**E3B** ^18^extends count-based episodic bonuses to continuous spaces: $b(s_t) = \phi(s_t)^T [\sum \phi(s_i)\phi(s_i)^T + \lambda I]^{-1} \phi(s_t)$ with embedding $\phi$ learned via an inverse dynamics model. E3B achieves SOTA on 16 MiniHack tasks ^19^at minimal overhead. Key hyperparameters: $\lambda \in \{0.01, 0.1, 1.0\}$ (final 1.0), intrinsic coefficient $\beta \in \{1.0, 0.1, 0.01, 0.001, 0.0001\}$ ^1^.

**Go-Explore** ^14^separates exploration (archive of cells and trajectories) from robustification (imitation learning). It scored 43k+ on Montezuma's Revenge but requires ~30 billion frames and environment resettability ^20^. The latent variant LGE ^21^removes hand-designed cells but remains incompatible with fast-sim budgets.

**Frontier-based** methods ^17^define frontiers as boundaries between explored and unexplored space on an occupancy grid. They require maintaining an explicit spatial map — architecturally complex when observations are egocentric and zero beyond 16 blocks. Learned frontier selection via RL ^22^improves over nearest-frontier heuristics but compounds system complexity.

### 1.3 Ranking by Suitability for Fast-Sim PPO

| Method | Compute Cost | PO Handling | Needs Map | Verdict |
|--------|-------------|-------------|-----------|---------|
| Position count / SimHash | Zero | Good if position known | Implicit | **Tier 1** ^12^|
| RE3 (random encoder + k-NN) | ~1.05x | Good | No | **Tier 1** ^13^|
| E3B-lite (small ID model) | ~1.2x | Excellent | No | **Tier 2** ^18^|
| RND (predictor network) | ~1.1x | Poor | No | Tier 2, combine w/ episodic ^1^|
| E3B + RND (multiplicative) | Low-mod. | Excellent | No | **Tier 3** ^23^|
| BYOL-Explore | ~1.1x | Moderate | No | Tier 3, if pixel obs ^8^|
| EC (reachability network) | 1.84x slower | Good | No | Avoid: too slow ^16^|
| Full NGU / Agent57 | 78B frames | Good | Implicit | **Avoid** ^10^|
| Go-Explore / LGE | ~30B frames | Moderate | Optional | **Avoid** ^20^|
| Frontier + RL | Moderate | Good | Yes (occupancy) | Avoid: map complexity ^22^|

Tier 1 methods — position counting or RE3 — add near-zero overhead and are proven on MiniGrid navigation ^5^. Tier 2 upgrades to E3B-lite, trading a small inverse dynamics model for better noise robustness ^18^. Tier 3 combines E3B and RND multiplicatively, producing large statistically significant gains on contextual MDPs ^23^. Methods marked "avoid" are excluded on training budget: NGU demands 256 actors ^10^, Go-Explore requires 30B frames ^20^, and frontier methods need an explicit occupancy map.

### 1.4 PPO Stabilization for Multi-Head Action Spaces with Masking

Masking after distribution sampling — renormalizing probabilities but computing gradients from the unmasked distribution — causes KL divergence explosion because invalid-action gradients are not zeroed ^14^. RLlib's adaptive KL multiplier compounds the problem: if sampled KL exceeds $2 \times \text{kl\_target}$, `kl_coeff *= 1.5` each update, producing non-finite losses ^21^.

**Fix 1:** Set `kl_coeff=0.0` and rely on `clip_param=0.2` alone. Consensus across RLlib maintainers ^24^, Spinning Up ^25^, and SB3 ^26^: the KL penalty is "much more brittle" than clipping for trust-region control.

**Fix 2:** Mask logits before softmax: `masked_logits = logits + clamp(log(action_mask), min=FLOAT_MIN)` where `FLOAT_MIN` $\approx -3.4 \times 10^{38}$ ^17^ ^27^. This zeros both probability and gradient for invalid actions ^14^.

**Fix 3:** Entropy schedule `[[0, 0.01], [1000000, 0.001]]` ^28^. For multi-head spaces, sum per-head entropy over masked distributions only ^19^ ^29^.

**Fix 4:** Clamp Gaussian log-std to $[-5, 2]$ for all heads including unused ones ^30^.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `kl_coeff` | 0.0 | Disable KL penalty ^24^|
| `clip_param` | 0.2 | Standard PPO trust region ^25^|
| `entropy_coeff_schedule` | [[0, 0.01], [1M, 0.001]] | Linear decay ^28^|
| `grad_clip` | 0.5 | Global L2 norm cap |
| `vf_loss_coeff` | 0.5 | Value loss weight |
| `num_sgd_iter` | 10 | PPO epochs per batch |
| `lr` | 3e-4 | Standard PPO rate |
| Log-std clamp | $[-5, 2]$ | Prevent unused head drift ^30^|
| Mask application | Pre-softmax (FLOAT_MIN) | Zero prob and gradient ^17^|

### 1.5 Behavior Cloning Warm-Start for Exploration Bootstrapping

PIRLNav demonstrated BC pretraining $\rightarrow$ RL finetuning achieves 65.0% ObjectNav success (+5.0% SOTA), but naive BC$\rightarrow$RL fails without a two-phase regime: first freeze the actor and train the critic on frozen-BC rollouts (~8M steps), then warm the actor LR from zero while decaying the critic LR ^31^. Without critic warmup, poor value estimates destroy the pretrained actor ^31^.

Demonstrator quality is critical. PIRLNav matched BC pretraining accuracy across three demonstrator types: human demos reached 66.1% VAL success, frontier exploration 51.3%, shortest paths 43.6% ^31^. Task-specific strategies transfer; task-agnostic patterns (generic spiral/lawnmower) do not. For offline-to-online alternatives, AWAC solves sparse-reward dexterous manipulation in 20 min online ^32^; IQL achieves SOTA on D4RL AntMaze stitching ^33^; Online Decision Transformer shows ~9x better finetuning than IQL but needs entropy regularization ^34^.

Verdict: viable for a task-specific scripted searcher with proper two-phase critic warmup; not recommended for generic spiral search.

## 2. Where Does the Intelligence Live?

### 2.1 Hierarchical and Hybrid Architectures

Four paradigms compete for the policy↔planner↔skill stack. The **Options framework** defines temporally extended actions as $\langle I, \pi, \beta \rangle$—initiation set, internal policy, and termination condition—enabling macro-action selection ^14^. Option-Critic extends this to end-to-end deep RL ^20^, but gains are tied to compositional environment structure, and option discovery under multi-agent non-stationarity remains open ^21^. **Feudal Networks** (FuN) decouple learning via a Manager that sets abstract goals in latent space and a Worker that executes primitive actions conditioned on them ^27^. Long-timescale credit assignment is excellent, yet Manager scalability concerns emerge as the goal space grows ^35^.

**HTN** (Hierarchical Task Network) planning and **GOAP** (Goal-Oriented Action Planning) offer the opposite tradeoff: superior debuggability at the cost of hand-engineering. HTN decomposes goals through structured task hierarchies with method preconditions ^19^; Guerrilla Games uses HTN in DECIMA to drive NPC decisions in *Horizon Zero Dawn* with Prolog-style backtracking and in-game visualization ^36^. GOAP searches backward from goals through precondition-effect graphs via A* ^30^. Both require exhaustive domain modeling and generalize poorly beyond predefined decompositions ^37^.

The **LLM-as-planner** paradigm offers the strongest generalization. Systems such as Voyager ^14^, GITM ^21^, and JARVIS-1 ^17^use an LLM as the high-level reasoning layer while delegating low-level control to learned policies, code generators, or scripted actions. Interpretability is high—plans are expressed in natural language or code—but API latency can exceed one to two seconds per decision ^27^.

### 2.2 Survey of Comparable Minecraft Agent Architectures

| System | High-Level Planner | Exploration Mechanism | Where-To-Go Source | Learned / Scripted |
|---|---|---|---|---|
| **Voyager** | GPT-4 automatic curriculum ^14^| LLM-driven novelty search; curriculum proposes next item/biome based on agent state | Planner generates task goal; code agent writes Mineflayer API calls | Curriculum is LLM-generated; no RL exploration policy |
| **GITM** | LLM Decomposer → Planner → Interface ^21^| Scripted BFS (surface) / DFS (underground) on 20×20 chessboard grid | **Split**: LLM decides WHAT; hardcoded BFS/DFS handles HOW | BFS/DFS is scripted; LLM decides WHEN to call `explore(object)` |
| **Plan4MC** | LLM offline skill graph + skill search ^27^| RL Finding-skill: LSTM high-level policy (PPO, count-based) + DQN low-level goal-reaching | Learned LSTM selects exploration direction from historical (x,y); skill graph plans skill sequence | **Learned**: LSTM policy trained with count-based intrinsic rewards; LLM used ONLY offline |
| **DreamerV3** | World model (RSSM) + imagined rollouts ^24^| Entropy-regularized actor in imagined trajectories; percentile-based return normalization | World model predicts future states; actor learns from imagined rollouts ^26^| **Learned**: World model, policy, and exploration all learned end-to-end |
| **BASALT (GoUp)** | Finite state machine ^28^| Fine-tuned VPT provides behavioral priors for walking/searching | FSM scripts flow (walk→search→detect→execute); VPT handles movement | **Hybrid**: Scripts for task flow; ML for perception and control |

The empirical spread is wide. Voyager discovered 63 unique items—3.3× prior SOTA—and was the only method to reach diamond-tier ^20^. GITM achieves 67.5% on ObtainDiamond, improving prior SOTA (VPT at 20%) by 47.5%, and unlocks all 262 items in the Overworld tech tree ^16^. Plan4MC requires only 7M environmental steps, making it the most sample-efficient demonstration-free RL method ^27^. DreamerV3 is the first algorithm to collect diamonds from scratch without human data or curricula ^24^. GoUp won BASALT 2022 by dividing tasks into ML-solvable and script-solvable parts: fine-tuned VPT for movement, YOLOv5 for detection, and a finite state machine for execution flow ^28^.

### 2.3 Consensus Pattern: "Find a Distant Resource"

Across every successful system, a universal three-layer pattern emerges for finding distant resources. At the highest layer, a planner determines **what** to find and **where** it might be located—a GPT-4 curriculum in Voyager ^14^, an LLM decomposer in GITM ^21^, a skill graph search in Plan4MC ^27^, or a world model in DreamerV3 ^26^. At the middle layer, a navigator produces intermediate spatial targets: Mineflayer API calls in Voyager, BFS/DFS grid waypoints in GITM, LSTM-selected goal locations in Plan4MC, or RSSM-predicted subgoals in DreamerV3. At the lowest layer, a reactive controller handles visuomotor execution—movement, obstacle avoidance, and terrain traversal.

The critical finding is that exploration intelligence is **split**. Strategic direction—which resource, which biome, when to switch—originates from the planner or a dedicated producer. Tactical execution—navigation patterns, obstacle avoidance—resides in the learned policy or scripted controller. Plan4MC exemplifies this split: its LLM constructs a static skill graph entirely offline, while an RL-trained Finding-skill handles all online exploration via a count-based LSTM policy maximizing unique grid cells visited ^27^. GITM likewise embodies the split: the LLM decides *what* object to find, but traversal is performed by a hardcoded BFS/DFS graph search independent of the LLM ^21^.

No validated pure end-to-end approach exists for long-horizon sparse-reward open-world tasks. Every successful system reviewed—Voyager, GITM, Plan4MC, DreamerV3, and the BASALT winners—employs hierarchical decomposition separating high-level reasoning about *what* from low-level control of *how*. Flat RL fails to learn directed exploration without this structural scaffolding ^38^. The implication for multi-agent architectures is direct: the intelligence for finding distant resources should live in a dedicated producer module that reasons about spatial knowledge and resource distributions, not solely in the control policy.

## 3. The Explorer/Scout Producer

### 3.1 Producing Exploration Targets from Partial Observation

The simplest reliable mechanism for emitting directional exploration targets from egocentric partial observations is a frontier-based pipeline that requires no learning at the base level. An online occupancy grid maps each observed cell into one of four states — {unknown, free, occupied, visited_count} — using a sparse hash-map representation that scales to the effectively infinite Minecraft surface ^14^. The Wavefront Frontier Detector (WFD) performs two nested breadth-first searches: the outer BFS traverses only known-free cells from the agent's current position, and when a cell adjacent to unknown territory is encountered, an inner BFS extracts the complete frontier region ^26^. Because WFD restricts scanning to known regions rather than the full map, it achieves $O(F)$ complexity where $F$ is the frontier cell count, making it suitable for real-time operation on each observation update ^20^.

Frontier scoring uses a geometric utility function $u_{\text{geo}}(f) = \text{length}(f) / \text{dist}(\text{robot}, f, \text{map})$ that rewards large frontiers near the agent ^25^. Information-theoretic alternatives that maximize expected map-uncertainty reduction can paradoxically increase total exploration time; prioritizing information gain yields fast short-term coverage but ultimately prolongs full-map completion because the method conflates budget-constrained and quality-constrained objectives ^39^. For the scout role, the highest-scoring frontier's centroid is emitted as a relative $(dx, dz)$ bearing or a cardinal-direction estimate with distance, sufficient for the downstream reactive controller to orient locomotion without requiring a full path plan.

### 3.2 Progressive Enhancement Path

The frontier pipeline admits four enhancement levels that progressively trade implementation simplicity for capability. Level 1 combines the occupancy grid, WFD, and geometric scoring without any learned component; this alone achieves complete coverage of bounded environments and is sufficient for the initial scout role ^40^. Level 2 adds a count-based visitation bonus that penalizes revisiting already-covered grid cells, biasing frontier selection toward less-visited regions. Count-based exploration outperforms Plan4MC's learned LSTM high-level policy on both coverage and revisit-count metrics ^9^, making it the highest-ROI first enhancement. Level 3 incorporates semantic priors from LLM world knowledge — for example, propagating biome temperature gradients to score frontiers by their likelihood of containing the target resource type ^37^. Minecraft biome placement follows temperature-smoothed climate zones where warm land has a 4/6 assignment probability, enabling temperature trajectories to serve as directional signals ^41^. Level 4 replaces the hand-designed scorer with a learned high-level policy similar to Plan4MC's Finding-skill, where a recurrent network observing historical positions emits goal coordinates and is trained with PPO to maximize unique grid-cell visitation ^28^. This progression is staged deliberately: evidence shows that tasks with Finding-skills achieve 40% conditional success versus 25% without ^42^, but the learned policy should only be introduced if geometric and count-based methods plateau.

### 3.3 Multi-Agent Information Sharing

Cooperating scouts share discovered locations through a shared map updated via delta messages. Each scout transmits only newly observed cells — changed terrain types, resource presences, and timestamps — rather than the full map state, avoiding synchronization overhead while ensuring all agents merge current environmental data into their local copies ^27^. This pattern is naturally compatible with CTDE: during centralized training, the critic conditions on the full shared-map state to provide low-variance advantage estimates; during decentralized execution, each actor conditions only on its local observation plus its local map copy ^14^. Parameter sharing is organized by role — all scouts share one network, all foragers another — maintaining within-role experience diversity while enabling cross-role behavioral specialization ^1^. Communication itself should be targeted rather than broadcast. IC3Net's learned gating mechanism controls when each agent communicates as a discrete decision ^25^, and MAGIC's ablation studies confirm that even with unlimited observability, full broadcast underperforms targeted communication ^37^. A scout gates its resource-location transmissions based on forager proximity and estimated relevance, reducing bandwidth while improving coordination precision.

## 4. Synthesis and Recommendation

**Executive Summary.** Five findings drive this recommendation. (1) Flat RL policies consistently underperform scripted or hybrid controllers on long-horizon sparse-reward navigation tasks—this is the established field consensus, validated by MineRL BASALT winners and DreamerV3 results ^28^ ^38^. (2) Five independent research dimensions converge on the same three-layer architecture: a high-level producer sets WHAT/WHERE, a mid-level navigator emits bearings, and a low-level reactive controller handles HOW ^14^ ^21^ ^27^ ^24^. (3) A goal-switching decision module that demotes skills and forces navigation when blind already implements the Finding-skill pattern seen in Plan4MC; the gap is a stabilized training pipeline and a producer that emits bearings from partial observations ^27^. (4) The forks are not mutually exclusive: Fork A's runtime trajectories become Fork B's BC warm-start data ^31^. (5) PPO instability has a known three-line fix (disable KL penalty, mask before softmax, entropy decay schedule) that must be applied before any exploration bonus ^24^.

### 4.1 Fork Analysis

**Fork A: thin reactive controller + Explorer/Scout producer.** This aligns with the consensus across every successful Minecraft agent reviewed ^14^ ^21^ ^27^ ^24^ ^28^. The decision-core already functions as a Finding-skill—demoting HARVEST and forcing NAVIGATE when blind—effectively acting as a high-level goal selector ^27^. What is missing is a producer converting partial observations into frontier-based bearings. Level 1 frontier detection (occupancy grid + Wavefront Frontier Detector + geometric scoring) requires no learning and provides clear sim→real transfer ^26^ ^40^. The perception mask is architecturally sound but should be driven by the scout's bearings rather than hardcoded cues.

**Fork B: end-to-end learned search.** This requires stabilized PPO plus an exploration bonus on a search-requiring arena. BC warm-start from a scripted follower is unlikely to exceed the follower because the demonstrator lacks behavioral diversity for transferable priors ^31^. The follower is already task-specific yet beats PPO—suggesting the bottleneck is architectural, not prior quality ^31^. Fork B also carries sim→real transfer risk: learned exploration policies are sensitive to distribution shift.

**Staged hybrid.** Plan4MC demonstrates this pattern: LLM constructs the skill graph offline while an RL-trained Finding-skill operates online ^27^. Fork A's explorer trajectories can become BC warm-start data for a future unified search policy (Fork B) ^31^.

### 4.2 Recommendation: Fork A with Staged Path to B

**Immediate (week 1–2):** Stabilize PPO: `kl_coeff=0.0`, mask logits with FLOAT_MIN ($\approx -3.4 \times 10^{38}$) before softmax, entropy schedule `[[0, 0.01], [1000000, 0.001]]` ^17^ ^24^ ^28^. Build the Level 1 scout: 2D occupancy grid (sparse hash-map) with WFD frontier detection, scoring frontiers by $\text{size} / (\text{distance} + 1)$ and emitting the highest-scoring centroid as a bearing ^26^ ^25^.

**Short-term (week 3–6):** Add an episodic bonus—RE3 (fixed random encoder, dim=64, k=4) or $(x, z)$ position count with $\beta = 0.01 / \sqrt{N(s)}$ ^13^ ^12^. Both add near-zero overhead and are proven on MiniGrid ^5^. Collect trajectories as demos.

**Medium-term (week 6+):** If trajectories show rich coverage, use BC→PPO warm-start with two-phase critic warmup (freeze actor, train critic on frozen BC rollouts for ~8M steps, then warm actor LR from zero) ^31^. Enter Fork B only when Fork A has produced sufficient data.

### 4.3 Highest-Leverage Next Experiments

**Fork A experiment.** Deploy stabilized PPO (`kl_coeff=0.0`, `clip_param=0.2`, `grad_clip=0.5`) ^24^ ^25^with the Level 1 scout on a 2-cluster blind arena. Measure held-out clearance using real (non-oracle) frontier bearings. Success: stable training (finite KL, no collapsing seeds) exceeding the scripted follower baseline.

**Fork B experiment.** Run RE3 (k=4, encoder dim=64) with stabilized PPO on a search-requiring arena where fixed-heading must fail. Compare clearance to the scripted follower on 3+ seeds.

**Evidence that would change the recommendation.** If Fork B achieves clearance strictly greater than the scripted follower with stable training across 3+ seeds, shift priority to Fork B. If Fork A fails to exceed the follower with stabilized PPO and frontier bearings, reconsider whether the arena contains learnable structure.

### 4.4 Concrete Techniques to Try First

| Technique | Parameter | Value | Source |
|---|---|---|---|
| PPO stabilization | `kl_coeff` | 0.0 (disabled) | ^24^|
|  | `clip_param` | 0.2 | ^25^|
|  | `entropy_coeff_schedule` | [[0, 0.01], [1000000, 0.001]] | ^28^|
|  | `grad_clip` | 0.5 | Standard |
|  | Mask application | Pre-softmax (FLOAT_MIN) | ^17^|
|  | Log-std clamp | $[-5, 2]$ | ^30^|
| Exploration bonus (RE3) | Random encoder dim | 64 | ^13^|
|  | k-NN neighbors ($k$) | 4 | ^13^|
|  | Overhead | ~1.05× | ^13^|
| Exploration bonus (count) | Position count $\beta$ | $0.01 / \sqrt{N(s)}$ | ^12^|
| Scout producer (Level 1) | Grid representation | Sparse hash-map | ^14^|
|  | Frontier detection | WFD (two-pass BFS) | ^26^|
|  | Scoring function | $\text{size} / (\text{distance} + 1)$ | ^25^|
|  | Output | Frontier centroid bearing | ^40^|

The PPO configuration eliminates the KL penalty because adaptive multipliers ($kl_{coeff} \times 1.5$ when sampled KL exceeds $2 \times$ target) produce non-finite losses when masking gradients are not zeroed ^21^. RE3 uses a fixed random encoder with no trainable parameters, making it cheaper than RND (~1.1× overhead) ^1^ ^13^. The count-based alternative is cheaper still if $(x, z)$ position is available ^12^. The Level 1 scout requires no learning: WFD runs in $O(F)$ frontier-cell complexity, and geometric scoring outperforms information-theoretic alternatives that increase total exploration time ^39^. Stabilized PPO, a lightweight bonus, and a geometric scout form the minimal viable system for Fork A.