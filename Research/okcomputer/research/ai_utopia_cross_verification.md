# Cross-Verification: AI-Utopia Exploration Learning Report

## High Confidence Findings (Confirmed by ≥2 dimensions, independent sources)

### HC-1: PPO KL Penalty Must Be Disabled with Action Masking
- **Sources**: Dim04 (RLlib forums, Huang & Ontanon 2020, Spinning Up, SB3)
- **Finding**: `kl_coeff=0.0` is the consensus fix. The KL penalty is "much more brittle than entropy." Rely only on PPO clipping (`clip_param=0.2`).
- **Agreement**: Unanimous across RLlib maintainers, academic implementations, and practitioner forums.

### HC-2: Mask Must Be Applied BEFORE Softmax
- **Sources**: Dim04 (Huang & Ontanon 2020, RLlib, CleanRL)
- **Finding**: Replace invalid logits with FLOAT_MIN (-3.4e38) before softmax. Never sample then reject. This zeros both probability AND gradient for invalid actions.

### HC-3: Hierarchical Architecture is Consensus for Long-Horizon Open-World Tasks
- **Sources**: Dim06, Dim07, Dim08, Dim11
- **Finding**: Every successful Minecraft/open-world agent uses hierarchical decomposition: high-level planner decides WHAT/WHERE, low-level policy decides HOW. Flat RL fails on long-horizon sparse-reward tasks.
- **Consistent pattern**: Voyager, GITM, JARVIS-1, Plan4MC, DreamerV3, MineRL BASALT winners all follow this pattern.

### HC-4: E3B (Elliptical Episodic Bonuses) is Top-Ranked Exploration Method
- **Sources**: Dim02, Dim03
- **Finding**: E3B achieves SOTA on MiniHack and Habitat, is simple to implement, low overhead, works with vector observations, and is ideal for procedural environments. Both dimensions independently recommend it.

### HC-5: Exploration Intelligence is SPLIT Between Planner and Controller
- **Sources**: Dim06, Dim07, Dim08
- **Finding**: The high-level module provides strategic direction (resource type, biome), while the low-level policy handles tactical execution (navigation, terrain traversal). There is no "pure" end-to-end approach among successful systems.

### HC-6: NGU/Agent57 is Overkill for Fast-Sim Single-Role Training
- **Sources**: Dim01, Dim02
- **Finding**: Full NGU requires 256 distributed actors, 78B frames, 32 policy heads. Completely unsuitable for minutes-to-train fast-sim. Simplified episodic component may be viable but full version is not.

### HC-7: Frontier-Based Methods Require a Map
- **Sources**: Dim03, Dim09
- **Finding**: Frontier exploration requires an occupancy/grid map. With egocentric vector obs zero beyond 16 blocks, the agent must maintain a belief map from egocentric observations. This is achievable but adds architectural complexity.

### HC-8: BC Warm-Start Can Help but Implementation Matters Critically
- **Sources**: Dim05 (multiple papers)
- **Finding**: BC→PPO works for exploration bootstrapping IF: (1) two-phase critic warmup is used, (2) demonstrator is task-specific, (3) KL regularization to BC policy during early RL. Naive BC→RL fails catastrophically without these safeguards.

### HC-9: Shared Map with Delta Updates is Simplest Multi-Agent Sharing Pattern
- **Sources**: Dim10, Dim11
- **Finding**: Scouts write only changed resource locations. CTDE training with centralized critic conditioned on full map, decentralized actors using local obs + local map copy. Role-based parameter sharing (shared within roles, different across).

### HC-10: RE3 or Simple Position Count is Cheapest Exploration Baseline
- **Sources**: Dim01, Dim02
- **Finding**: RE3 (random encoder + k-NN entropy) has near-zero compute cost. Position-based count (if x,y available) is even cheaper. Both proven effective on MiniGrid navigation tasks.

---

## Medium Confidence Findings (1 authoritative source)

### MC-1: Per-Head Entropy Should Be Summed Over Masked Distributions
- **Source**: Dim04 (RLlib discussion)
- **Finding**: For multi-head action spaces, sum entropy across heads, computing each over the masked distribution only. Not widely validated in literature.

### MC-2: Entropy Coefficient Should Decay Over Training
- **Source**: Dim04 (RLlib forums)
- **Finding**: `entropy_coeff_schedule=[[0, 0.01], [1000000, 0.001]]`. Standard practice but specific numbers are heuristic.

### MC-3: IC3Net-Style Gating Beats Full Broadcast for Communication
- **Source**: Dim10 (MAGIC ablations)
- **Finding**: Even with perfect observability, targeted communication outperforms broadcast. "Deciding when and whom is critical."

### MC-4: Plan4MC's Finding-Skill Pattern Matches AI-Utopia's Needs
- **Source**: Dim07, Dim08
- **Finding**: Plan4MC's hierarchical Finding-skill (high-level goal policy + goal-conditioned low-level) is the closest architectural match to the decision-core fork. LLM is used ONLY offline to build skill graph.

### MC-5: Voyager's Curriculum is Load-Bearing
- **Source**: Dim07 (Voyager ablation)
- **Finding**: Removing curriculum → -93% discovered items. Removing self-verification → -73% performance. The exploration strategy (automatic curriculum) is more important than the execution mechanism.

---

## Low Confidence Findings (Weak sourcing or unverified)

### LC-1: LSTM Can Implicitly Learn Spatial Map
- **Source**: Dim01 (speculative claim)
- **Finding**: No intrinsic motivation method directly solves spatial memory — the LSTM must implicitly learn a spatial map. This is a theoretical gap, not an empirical finding.

### LC-2: SMMAE Adaptive Exploration for Multi-Agent
- **Source**: Dim11 (single paper, 2024)
- **Finding**: SMMAE adjusts exploration based on multi-agent uncertainty. Early work, limited validation.

---

## Conflict Zones

### CZ-1: Is Exploration Learned or Scripted? (CENTRAL CONFLICT)
- **Dim07/Dim08**: Exploration is predominantly learned in SOTA systems (DreamerV3 world model, Plan4MC Finding-skill, VPT behavioral priors).
- **Dim06**: Successful systems use hybrid approaches — LLM/HTN planner + learned skills.
- **PROJECT ABLATION**: Zero-learning scripted follower ties or beats trained PPO in every condition. The perception mask (env scaffolding) is load-bearing, not the policy.
- **Resolution**: This is NOT a true contradiction. The project ablation tests a SINGLE flat PPO policy on a VERY simple explore task (one blind hop). SOTA systems use hierarchical architectures with explicit exploration handling (count-based, episodic memory, world models). The ablation result says "flat PPO can't learn directed exploration" — this is consistent with the consensus that flat RL fails on long-horizon exploration. The question is whether a PROPERLY ARCHITECTED system (hierarchical + exploration bonus + stabilized PPO) could learn it.

### CZ-2: Best Intrinsic Reward Method for This Specific Setup
- **Dim01**: Recommends RE3 or simple position count (cheapest, proven on MiniGrid).
- **Dim02**: Recommends E3B-lite (best balanced for vector obs).
- **Dim03**: Recommends E3B + RND multiplicatively (best empirical performance).
- **Resolution**: These are staged recommendations, not contradictions. Start with RE3/count (cheapest baseline), upgrade to E3B if needed, combine with RND for best performance. All three agree: start simple.

### CZ-3: Whether BC Warm-Start Teaches Learnable Search
- **Dim05**: "Cautiously yes" — PIRLNav showed BC→RL works for navigation, but task-specific demonstrator matters. Generic spiral/lawnmower is task-agnostic and transfers poorly.
- **Implication**: A scripted searcher that biases toward oracle cues (task-specific) may transfer better than a generic spiral. But the evidence is from ObjectNav (different domain).

### CZ-4: Should LLM Be Online or Offline Planner?
- **Dim07**: Voyager uses GPT-4 online (every decision). Plan4MC uses LLM ONLY offline (skill graph construction). GITM uses LLM online.
- **Resolution**: Offline LLM (Plan4MC pattern) has lower latency and is more suitable for real-time Minecraft. Online LLM (Voyager pattern) is more flexible but requires API calls. Tradeoff is latency vs adaptability.
