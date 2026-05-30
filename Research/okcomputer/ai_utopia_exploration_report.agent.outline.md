# Sparse-Reward Exploration Learning & Multi-Agent Architecture — Decision Report for AI-Utopia

## Executive Summary (5 bullets, ~200 words)
### Key Findings
#### Fork A (thin reactive controller + smart producers) is the consensus architecture across the field and matches the project's ablation evidence
#### PPO instability has a known 3-line fix (kl_coeff=0.0, entropy schedule, mask-before-softmax) — not an algorithmic blocker
#### E3B episodic bonuses or simple position counting are the right exploration complexity for a fast-sim; NGU/Go-Explore are overkill
#### BC warm-start from a task-specific scripted searcher can bootstrap exploration but requires two-phase critic warmup
#### A staged hybrid is viable: Fork A now produces data that enables Fork B later

## 1. Sparse-Reward Exploration Learning (~1200 words, 2 tables)
### 1.1 Intrinsic Motivation Methods: RND, ICM, NGU, BYOL-Explore, and Count-Based
#### 1.1.1 RND: random network distillation mechanism, low compute, Noisy-TV robust, poor partial-observability handling
#### 1.1.2 ICM: forward dynamics prediction error, fails on Noisy-TV and stochastic transitions, unsuitable for Minecraft's partial observability
#### 1.1.3 NGU/Agent57: episodic + lifelong novelty, requires 256 distributed actors and 78B frames — overkill for fast-sim
#### 1.1.4 BYOL-Explore: temporal consistency in latent space, moderate compute, better for visual tasks than vector obs
#### 1.1.5 Count/pseudo-count and RE3: position-based counting is cheapest; RE3 (random encoder + k-NN) is near-zero overhead; both proven on MiniGrid navigation
### 1.2 Episodic Memory & Frontier-Based Exploration
#### 1.2.1 Episodic Curiosity (Savinov et al.): reachability network overcomes "couch potato" problem, 1.84x slower than PPO
#### 1.2.2 E3B: elliptical episodic bonuses achieve SOTA on MiniHack with low overhead; best balanced option for vector obs
#### 1.2.3 Go-Explore: brilliant on Montezuma but requires 30B frames and environment resettability; latent variant (LGE) more practical but still complex
#### 1.2.4 Frontier-based methods: require occupancy map; learned frontier selection can work with egocentric obs but adds architectural complexity
### 1.3 Ranking by Suitability for Fast-Sim PPO (table: method × compute cost × PO handling × map requirement × recommendation)
#### 1.3.1 Tier 1 (try first): RE3 or simple position count — near-zero overhead, proven on navigation
#### 1.3.2 Tier 2 (upgrade): E3B-lite — small inverse dynamics model, more robust to noise
#### 1.3.3 Tier 3 (combine): E3B + RND multiplicatively — best empirical performance on contextual MDPs
#### 1.3.4 Avoid: Full NGU, Go-Explore (training budget incompatible with minutes-to-train fast-sim)
### 1.4 PPO Stabilization for Multi-Head Action Spaces with Masking
#### 1.4.1 Root cause: action masking after distribution sampling causes log(0) in KL computation; RLlib's adaptive KL multiplier (kl_coeff *= 1.5) creates runaway explosion
#### 1.4.2 Fix #1: kl_coeff=0.0 — disable KL penalty, rely on PPO clipping alone; consensus across RLlib, Spinning Up, SB3
#### 1.4.3 Fix #2: mask logits with FLOAT_MIN before softmax — zeros probability AND gradient for invalid actions
#### 1.4.4 Fix #3: entropy coefficient schedule — decay from 0.01 to 0.001 over training; sum per-head entropy over masked distributions
#### 1.4.5 Fix #4: log-std clamp to [-5, 2] for unused Gaussian heads; hard masking strongly preferred over penalty
#### 1.4.6 Concrete RLlib config (code block): kl_coeff=0.0, entropy_coeff_schedule, clip_param=0.2, grad_clip=0.5
### 1.5 Behavior Cloning Warm-Start for Exploration Bootstrapping
#### 1.5.1 BC→PPO evidence: PIRLNav achieved 65% ObjectNav (+5% SOTA) with BC pretraining → RL finetuning; naive BC→RL fails without critic warmup
#### 1.5.2 Critical implementation: two-phase regime (critic-only learning first, then actor warmup) prevents catastrophic performance drop
#### 1.5.3 Task-specific demonstrator matters: human demos > frontier exploration > shortest paths for transfer; generic spiral/lawnmower is task-agnostic and transfers poorly
#### 1.5.4 Alternatives: AWAC for sparse-reward dexterous manipulation; IQL for D4RL stitching; Decision Transformer ~9x better finetuning but needs entropy regularization
#### 1.5.5 Verdict: cautiously yes for task-specific scripted searcher with proper implementation; no for generic spiral search

## 2. Where Does the Intelligence Live? (~800 words, 1 table)
### 2.1 Hierarchical and Hybrid Architectures
#### 2.1.1 Options framework: end-to-end learnable, non-stationarity with multiple agents, option discovery remains hard
#### 2.1.2 Feudal/HRL: excellent long-timescale credit assignment via goal conditioning; manager scalability concerns
#### 2.1.3 HTN/GOAP: excellent debuggability and predictability; extensive hand-engineering, limited generalization
#### 2.1.4 LLM-as-planner: best generalization and interpretability; API latency concerns; Voyager, GITM, JARVIS-1 as exemplars
### 2.2 Survey of Comparable Minecraft Agent Architectures (table: system × exploration × where-to-go × learned/scripted)
#### 2.2.1 Voyager: LLM curriculum + code skill library; exploration is LLM-driven curriculum, not learned policy; 3.3x more items than prior SOTA
#### 2.2.2 GITM: LLM decomposer → planner → interface; explore() is scripted BFS/DFS; LLM decides WHEN/WHAT, hardcoded traversal
#### 2.2.3 Plan4MC: LLM offline skill graph + RL Finding-skill; exploration is LEARNED count-based policy; most sample-efficient
#### 2.2.4 DreamerV3: world model + imagined rollouts; exploration learned via entropy-regularized actor; first diamonds from scratch
#### 2.2.5 MineRL BASALT winners: hybrid learned + scripted; ML for perception/control, scripts for task decomposition
### 2.3 Consensus Pattern: "Find a Distant Resource"
#### 2.3.1 Universal three-layer pattern: high-level planner (WHAT/WHERE) → mid-level navigator (waypoint/bearing) → low-level controller (HOW)
#### 2.3.2 Exploration intelligence is SPLIT: strategic direction from planner/producer, tactical execution from learned policy
#### 2.3.3 Every successful system uses this pattern — there is no validated pure end-to-end approach for long-horizon sparse-reward open-world tasks

## 3. The Explorer/Scout Producer (~600 words)
### 3.1 Producing Exploration Targets from Partial Observation
#### 3.1.1 Online occupancy grid: sparse hash-map tracking {unknown, free, occupied, visited_count} — sufficient for 2D Minecraft surface
#### 3.1.2 Wavefront Frontier Detector (WFD): BFS from agent through known-free cells, flag cells adjacent to unknown — O(F) time where F is frontier count
#### 3.1.3 Frontier scoring: simple geometric utility = frontier_size / (path_cost + epsilon); information-gain methods can hurt exploration time
#### 3.1.4 Goal emission: emit frontier centroid as (direction, distance) for downstream reactive controller
### 3.2 Progressive Enhancement Path
#### 3.2.1 Level 1 (no learning): occupancy grid + WFD + geometric scoring — sufficient for initial scout role
#### 3.2.2 Level 2: add count-based visitation bonus to bias toward unvisited frontiers
#### 3.2.3 Level 3: add semantic/LLM priors (biome temperature gradients) to prioritize resource-likely frontiers
#### 3.2.4 Level 4: learned high-level policy (like Plan4MC Finding-skill) if simpler methods plateau
### 3.3 Multi-Agent Information Sharing
#### 3.3.1 Shared map with delta updates: scouts write only changed resource locations; foragers read local map copy
#### 3.3.2 CTDE compatibility: centralized critic conditioned on full map; decentralized actors use local obs + local map
#### 3.3.3 Role-based parameter sharing: shared within roles (all scouts, all foragers), different across roles
#### 3.3.4 Targeted communication: IC3Net-style gating learns WHEN to communicate; beats full broadcast even with perfect observability

## 4. Synthesis and Recommendation (~600 words)
### 4.1 Fork Analysis
#### 4.1.1 Fork A assessment: thin reactive controller + Explorer producer; aligns with field consensus; decision-core IS the Finding-skill pattern; clear sim→real transfer path
#### 4.1.2 Fork B assessment: end-to-end learned search; requires stabilized PPO + exploration bonus + search-requiring arena; BC warm-start unlikely to exceed scripted follower; higher research risk
#### 4.1.3 Staged hybrid: Fork A produces runtime data that becomes Fork B's BC training data; not mutually exclusive
### 4.2 Recommendation: Fork A with Staged Path to B
#### 4.2.1 Immediate: stabilize PPO (kl=0, entropy schedule, proper masking) + build Explorer producer with Level 1 frontier detection
#### 4.2.2 Short-term: add episodic count or RE3 bonus to PPO; collect explorer trajectories as demonstration data
#### 4.2.3 Medium-term: if explorer data is rich, use BC→PPO warm-start to train a unified search policy (Fork B)
### 4.3 Highest-Leverage Next Experiments
#### 4.3.1 Fork A experiment: stabilized PPO + Level-1 scout producer on 2-cluster blind arena; measure held-out clearance with real (non-oracle) frontier bearings
#### 4.3.2 Fork B experiment: RE3 episodic bonus + stabilized PPO on search-requiring arena (fixed-heading must fail); compare to follower baseline
#### 4.3.3 Evidence that would change recommendation: if Fork B experiment achieves >follower clearance with stable training on 3+ seeds, reconsider priority
### 4.4 Concrete Techniques to Try First (settings)
#### 4.4.1 PPO config: kl_coeff=0.0, entropy_coeff_schedule=[[0,0.01],[1000000,0.001]], clip_param=0.2, grad_clip=0.5
#### 4.4.2 Exploration bonus: RE3 with small random encoder (dim=64) + k=4 nearest neighbors; or simple (x,z) position count with beta=0.01/sqrt(N)
#### 4.4.3 Scout producer: 2D occupancy grid (sparse hash-map) + WFD frontier detection + score=size/(distance+1)

# References
## ai_utopia_exploration_report.agent.outline.md
- **Type**: Report outline
- **Description**: This outline file
- **Path**: /mnt/agents/output/ai_utopia_exploration_report.agent.outline.md

## Research Artifacts
- **Type**: Deep research output
- **Description**: 11 dimension files + cross-verification + insight extraction
- **Path**: /mnt/agents/output/research/ai_utopia_dim01.md through ai_utopia_dim11.md, ai_utopia_cross_verification.md, ai_utopia_insight.md
