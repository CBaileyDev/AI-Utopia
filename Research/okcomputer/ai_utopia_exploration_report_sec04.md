## 4. Synthesis and Recommendation

**Executive Summary.** Five findings drive this recommendation. (1) Flat RL policies consistently underperform scripted or hybrid controllers on long-horizon sparse-reward navigation tasks—this is the established field consensus, validated by MineRL BASALT winners and DreamerV3 results [^14^][^19^]. (2) Five independent research dimensions converge on the same three-layer architecture: a high-level producer sets WHAT/WHERE, a mid-level navigator emits bearings, and a low-level reactive controller handles HOW [^1^][^3^][^6^][^8^]. (3) A goal-switching decision module that demotes skills and forces navigation when blind already implements the Finding-skill pattern seen in Plan4MC; the gap is a stabilized training pipeline and a producer that emits bearings from partial observations [^6^]. (4) The forks are not mutually exclusive: Fork A's runtime trajectories become Fork B's BC warm-start data [^321^]. (5) PPO instability has a known three-line fix (disable KL penalty, mask before softmax, entropy decay schedule) that must be applied before any exploration bonus [^8^].

### 4.1 Fork Analysis

**Fork A: thin reactive controller + Explorer/Scout producer.** This aligns with the consensus across every successful Minecraft agent reviewed [^1^][^3^][^6^][^8^][^14^]. The decision-core already functions as a Finding-skill—demoting HARVEST and forcing NAVIGATE when blind—effectively acting as a high-level goal selector [^6^]. What is missing is a producer converting partial observations into frontier-based bearings. Level 1 frontier detection (occupancy grid + Wavefront Frontier Detector + geometric scoring) requires no learning and provides clear sim→real transfer [^10^][^30^]. The perception mask is architecturally sound but should be driven by the scout's bearings rather than hardcoded cues.

**Fork B: end-to-end learned search.** This requires stabilized PPO plus an exploration bonus on a search-requiring arena. BC warm-start from a scripted follower is unlikely to exceed the follower because the demonstrator lacks behavioral diversity for transferable priors [^321^]. The follower is already task-specific yet beats PPO—suggesting the bottleneck is architectural, not prior quality [^321^]. Fork B also carries sim→real transfer risk: learned exploration policies are sensitive to distribution shift.

**Staged hybrid.** Plan4MC demonstrates this pattern: LLM constructs the skill graph offline while an RL-trained Finding-skill operates online [^6^]. Fork A's explorer trajectories can become BC warm-start data for a future unified search policy (Fork B) [^321^].

### 4.2 Recommendation: Fork A with Staged Path to B

**Immediate (week 1–2):** Stabilize PPO: `kl_coeff=0.0`, mask logits with FLOAT_MIN ($\approx -3.4 \times 10^{38}$) before softmax, entropy schedule `[[0, 0.01], [1000000, 0.001]]` [^5^][^8^][^14^]. Build the Level 1 scout: 2D occupancy grid (sparse hash-map) with WFD frontier detection, scoring frontiers by $\text{size} / (\text{distance} + 1)$ and emitting the highest-scoring centroid as a bearing [^10^][^12^].

**Short-term (week 3–6):** Add an episodic bonus—RE3 (fixed random encoder, dim=64, k=4) or $(x, z)$ position count with $\beta = 0.01 / \sqrt{N(s)}$ [^77^][^131^]. Both add near-zero overhead and are proven on MiniGrid [^47^]. Collect trajectories as demos.

**Medium-term (week 6+):** If trajectories show rich coverage, use BC→PPO warm-start with two-phase critic warmup (freeze actor, train critic on frozen BC rollouts for ~8M steps, then warm actor LR from zero) [^321^]. Enter Fork B only when Fork A has produced sufficient data.

### 4.3 Highest-Leverage Next Experiments

**Fork A experiment.** Deploy stabilized PPO (`kl_coeff=0.0`, `clip_param=0.2`, `grad_clip=0.5`) [^8^][^12^] with the Level 1 scout on a 2-cluster blind arena. Measure held-out clearance using real (non-oracle) frontier bearings. Success: stable training (finite KL, no collapsing seeds) exceeding the scripted follower baseline.

**Fork B experiment.** Run RE3 (k=4, encoder dim=64) with stabilized PPO on a search-requiring arena where fixed-heading must fail. Compare clearance to the scripted follower on 3+ seeds.

**Evidence that would change the recommendation.** If Fork B achieves clearance strictly greater than the scripted follower with stable training across 3+ seeds, shift priority to Fork B. If Fork A fails to exceed the follower with stabilized PPO and frontier bearings, reconsider whether the arena contains learnable structure.

### 4.4 Concrete Techniques to Try First

| Technique | Parameter | Value | Source |
|---|---|---|---|
| PPO stabilization | `kl_coeff` | 0.0 (disabled) | [^8^] |
|  | `clip_param` | 0.2 | [^12^] |
|  | `entropy_coeff_schedule` | [[0, 0.01], [1000000, 0.001]] | [^14^] |
|  | `grad_clip` | 0.5 | Standard |
|  | Mask application | Pre-softmax (FLOAT_MIN) | [^5^] |
|  | Log-std clamp | $[-5, 2]$ | [^20^] |
| Exploration bonus (RE3) | Random encoder dim | 64 | [^77^] |
|  | k-NN neighbors ($k$) | 4 | [^77^] |
|  | Overhead | ~1.05× | [^77^] |
| Exploration bonus (count) | Position count $\beta$ | $0.01 / \sqrt{N(s)}$ | [^131^] |
| Scout producer (Level 1) | Grid representation | Sparse hash-map | [^1^] |
|  | Frontier detection | WFD (two-pass BFS) | [^10^] |
|  | Scoring function | $\text{size} / (\text{distance} + 1)$ | [^12^] |
|  | Output | Frontier centroid bearing | [^30^] |

The PPO configuration eliminates the KL penalty because adaptive multipliers ($kl_{coeff} \times 1.5$ when sampled KL exceeds $2 \times$ target) produce non-finite losses when masking gradients are not zeroed [^3^]. RE3 uses a fixed random encoder with no trainable parameters, making it cheaper than RND (~1.1× overhead) [^22^][^77^]. The count-based alternative is cheaper still if $(x, z)$ position is available [^131^]. The Level 1 scout requires no learning: WFD runs in $O(F)$ frontier-cell complexity, and geometric scoring outperforms information-theoretic alternatives that increase total exploration time [^13^]. Stabilized PPO, a lightweight bonus, and a geometric scout form the minimal viable system for Fork A.
