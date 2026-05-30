## 1. Sparse-Reward Exploration Learning

When extrinsic rewards are sparse, the agent must generate its own learning signal. This section evaluates intrinsic motivation methods, ranks them for fast-sim PPO, provides a stabilization recipe for masked multi-head action spaces, and assesses BC warm-start as an exploration bootstrap.

### 1.1 Intrinsic Motivation Methods

**RND** [^22^] measures novelty as prediction error of a trainable network matching a fixed random target on the current observation. The deterministic target makes RND robust to the Noisy-TV problem by construction — it captures only epistemic uncertainty, not environment noise [^22^]. Overhead is minimal (~1.1x, one forward pass at ~11 learner steps/sec) [^66^]. RND requires dual value heads ($V_E$, $V_I$) with $\gamma_E = 0.999$, $\gamma_I = 0.99$ [^95^]. Its global novelty bonus fails in procedurally-generated environments where states are never revisited [^55^], and without RMS normalization achieves zero reward [^47^]. On MiniGrid-DoorKey-8x8, tuned RND reaches 82% success [^47^].

**ICM** [^32^] uses forward dynamics prediction error as the exploration bonus. It cannot distinguish unpredictable transitions due to novelty from those due to stochasticity, making it susceptible to the Noisy-TV problem [^32^]. Under partial observability, the forward model predicts from incomplete information, producing noisy intrinsic rewards prone to "detachment" — when bonuses decay, the agent abandons exploration frontiers [^96^]. ICM scored 83% on MiniGrid-DoorKey-8x8 [^47^] but failed on 5 of 8 DM-HARD-8 long-horizon tasks [^34^].

**NGU / Agent57** [^29^] combines episodic novelty (k-NN within episodes) with lifelong novelty (RND) via a UVFA learning 32 distinct policies. Agent57 surpassed human baseline on all 57 Atari games but required 78 billion frames with 256 distributed actors [^101^] — architecturally incompatible with minutes-to-train fast-sim.

**BYOL-Explore** [^34^] predicts its own future latent representation and uses prediction error as intrinsic reward. Robust to pixel-level noise — it solves 5.5/8 DM-HARD-8 tasks [^34^] — but fails when stochasticity operates at the latent level [^158^]. Compute matches RND [^66^]. With 16-block visibility, latent predictions must extrapolate far beyond the observation horizon, making this method better suited to pixel inputs than vector states.

**Count-based and RE3.** Position-based counting tracks visited $(x, y)$ and rewards with $\beta / \sqrt{N(s)}$ [^131^]. For low-dimensional vector observations, state count outperforms ICM and maximum entropy methods [^55^]. RE3 [^77^] uses a fixed random encoder (no gradients) with k-NN entropy estimation at ~1.05x overhead, achieving 95% on MiniGrid-DoorKey-8x8 — above RND (82%) and ICM (83%) [^47^].

### 1.2 Episodic Memory and Frontier-Based Exploration

**Episodic Curiosity (EC)** [^1^] uses a learned reachability network measuring novelty via environment-step distance from episodic memory, overcoming the "couch potato" problem [^28^]. PPO+EC is 1.84x slower and adds 13M parameters [^4^]; agents also preferentially explore room corners and blind alleys [^5^].

**E3B** [^21^] extends count-based episodic bonuses to continuous spaces: $b(s_t) = \phi(s_t)^T [\sum \phi(s_i)\phi(s_i)^T + \lambda I]^{-1} \phi(s_t)$ with embedding $\phi$ learned via an inverse dynamics model. E3B achieves SOTA on 16 MiniHack tasks [^16^] at minimal overhead. Key hyperparameters: $\lambda \in \{0.01, 0.1, 1.0\}$ (final 1.0), intrinsic coefficient $\beta \in \{1.0, 0.1, 0.01, 0.001, 0.0001\}$ [^22^].

**Go-Explore** [^1^] separates exploration (archive of cells and trajectories) from robustification (imitation learning). It scored 43k+ on Montezuma's Revenge but requires ~30 billion frames and environment resettability [^2^]. The latent variant LGE [^3^] removes hand-designed cells but remains incompatible with fast-sim budgets.

**Frontier-based** methods [^5^] define frontiers as boundaries between explored and unexplored space on an occupancy grid. They require maintaining an explicit spatial map — architecturally complex when observations are egocentric and zero beyond 16 blocks. Learned frontier selection via RL [^7^] improves over nearest-frontier heuristics but compounds system complexity.

### 1.3 Ranking by Suitability for Fast-Sim PPO

| Method | Compute Cost | PO Handling | Needs Map | Verdict |
|--------|-------------|-------------|-----------|---------|
| Position count / SimHash | Zero | Good if position known | Implicit | **Tier 1** [^131^] |
| RE3 (random encoder + k-NN) | ~1.05x | Good | No | **Tier 1** [^77^] |
| E3B-lite (small ID model) | ~1.2x | Excellent | No | **Tier 2** [^21^] |
| RND (predictor network) | ~1.1x | Poor | No | Tier 2, combine w/ episodic [^22^] |
| E3B + RND (multiplicative) | Low-mod. | Excellent | No | **Tier 3** [^24^] |
| BYOL-Explore | ~1.1x | Moderate | No | Tier 3, if pixel obs [^34^] |
| EC (reachability network) | 1.84x slower | Good | No | Avoid: too slow [^4^] |
| Full NGU / Agent57 | 78B frames | Good | Implicit | **Avoid** [^101^] |
| Go-Explore / LGE | ~30B frames | Moderate | Optional | **Avoid** [^2^] |
| Frontier + RL | Moderate | Good | Yes (occupancy) | Avoid: map complexity [^7^] |

Tier 1 methods — position counting or RE3 — add near-zero overhead and are proven on MiniGrid navigation [^47^]. Tier 2 upgrades to E3B-lite, trading a small inverse dynamics model for better noise robustness [^21^]. Tier 3 combines E3B and RND multiplicatively, producing large statistically significant gains on contextual MDPs [^24^]. Methods marked "avoid" are excluded on training budget: NGU demands 256 actors [^101^], Go-Explore requires 30B frames [^2^], and frontier methods need an explicit occupancy map.

### 1.4 PPO Stabilization for Multi-Head Action Spaces with Masking

Masking after distribution sampling — renormalizing probabilities but computing gradients from the unmasked distribution — causes KL divergence explosion because invalid-action gradients are not zeroed [^1^]. RLlib's adaptive KL multiplier compounds the problem: if sampled KL exceeds $2 \times \text{kl\_target}$, `kl_coeff *= 1.5` each update, producing non-finite losses [^3^].

**Fix 1:** Set `kl_coeff=0.0` and rely on `clip_param=0.2` alone. Consensus across RLlib maintainers [^8^], Spinning Up [^12^], and SB3 [^10^]: the KL penalty is "much more brittle" than clipping for trust-region control.

**Fix 2:** Mask logits before softmax: `masked_logits = logits + clamp(log(action_mask), min=FLOAT_MIN)` where `FLOAT_MIN` $\approx -3.4 \times 10^{38}$ [^5^][^6^]. This zeros both probability and gradient for invalid actions [^1^].

**Fix 3:** Entropy schedule `[[0, 0.01], [1000000, 0.001]]` [^14^]. For multi-head spaces, sum per-head entropy over masked distributions only [^16^][^18^].

**Fix 4:** Clamp Gaussian log-std to $[-5, 2]$ for all heads including unused ones [^20^].

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `kl_coeff` | 0.0 | Disable KL penalty [^8^] |
| `clip_param` | 0.2 | Standard PPO trust region [^12^] |
| `entropy_coeff_schedule` | [[0, 0.01], [1M, 0.001]] | Linear decay [^14^] |
| `grad_clip` | 0.5 | Global L2 norm cap |
| `vf_loss_coeff` | 0.5 | Value loss weight |
| `num_sgd_iter` | 10 | PPO epochs per batch |
| `lr` | 3e-4 | Standard PPO rate |
| Log-std clamp | $[-5, 2]$ | Prevent unused head drift [^20^] |
| Mask application | Pre-softmax (FLOAT_MIN) | Zero prob and gradient [^5^] |

### 1.5 Behavior Cloning Warm-Start for Exploration Bootstrapping

PIRLNav demonstrated BC pretraining $\rightarrow$ RL finetuning achieves 65.0% ObjectNav success (+5.0% SOTA), but naive BC$\rightarrow$RL fails without a two-phase regime: first freeze the actor and train the critic on frozen-BC rollouts (~8M steps), then warm the actor LR from zero while decaying the critic LR [^321^]. Without critic warmup, poor value estimates destroy the pretrained actor [^321^].

Demonstrator quality is critical. PIRLNav matched BC pretraining accuracy across three demonstrator types: human demos reached 66.1% VAL success, frontier exploration 51.3%, shortest paths 43.6% [^321^]. Task-specific strategies transfer; task-agnostic patterns (generic spiral/lawnmower) do not. For offline-to-online alternatives, AWAC solves sparse-reward dexterous manipulation in 20 min online [^189^]; IQL achieves SOTA on D4RL AntMaze stitching [^173^]; Online Decision Transformer shows ~9x better finetuning than IQL but needs entropy regularization [^192^].

Verdict: viable for a task-specific scripted searcher with proper two-phase critic warmup; not recommended for generic spiral search.
