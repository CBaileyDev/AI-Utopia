## 7. Multi-Agent Failure Modes and Mitigations

Cooperative multi-agent reinforcement learning (MARL) introduces failure modes that have no single-agent analogue. In a 4-role Minecraft village system — where a gatherer, builder, farmer, and defender share a sparse team reward, occupy the same physical space, and depend on each other's outputs — these failures compound. A gatherer that cannot determine whether its wood collection or the builder's block placement caused the house-completion bonus receives noisy gradients and stops learning. A builder that adapts its policy to the farmer's past behavior finds that policy obsolete when the farmer learns to replant crops instead of wandering. These are not hypothetical concerns: they are the dominant obstacles documented across SMAC, Google Research Football, and warehouse robotics benchmarks that most closely approximate village-building coordination. [^295^] [^303^] [^305^]

This chapter catalogues ten documented failure modes in cooperative MARL, ranks each by its probability and severity in a 4-role Minecraft village, and pairs every mode with a concrete mitigation. The prioritization reflects both empirical evidence and the structural properties of the domain: sparse rewards, heterogeneous roles, shared physical space, and long-horizon task dependencies. The goal is a clear order of operations — what to build first, what to monitor continuously, and what can safely wait.

### 7.1 High-Priority Failures

Three failure modes are classified as high priority because they strike at the foundations of learning in a 4-agent cooperative system with shared reward. Each is present from the first training step and has been shown to stall or derail training entirely in comparable benchmarks.

#### 7.1.1 Credit Assignment: Global Reward Decomposition with COMA Counterfactual Baselines

The credit assignment problem arises when all agents optimize a single team reward and no individual agent can determine how much its own actions contributed. In policy-gradient methods, the gradient computed for each actor "does not explicitly reason about how that particular agent's actions contribute to that global reward" and becomes "very noisy, particularly when there are many agents." [^305^] In the village scenario, when the team receives a reward because a house was completed, the centralized critic must disentangle the gatherer's wood collection, the builder's block placement, and the defender's mob clearance. Without explicit decomposition, all four agents receive nearly identical gradients, and those whose contributions are temporally distant receive no useful learning signal. [^308^]

COMA (Counterfactual Multi-Agent Policy Gradients) provides a principled solution. The centralized critic computes a counterfactual baseline for each agent by marginalizing out that agent's action while holding all others fixed: the advantage is $Q(s, \mathbf{a}) - \sum_{a_i'} \pi(a_i'|o_i) \, Q(s, (a_i', \mathbf{a}_{-i}))$. [^305^] [^308^] This answers the question: "How much worse would the team have done if this agent had acted differently?" The computation is efficient — a single forward pass yields all agents' baselines. However, COMA is "prone to getting stuck in sub-optimal local minima" when exploration is insufficient, [^310^] so it must be paired with entropy regularization or diversity bonuses. For a 4-agent village with discrete actions, the scaling is manageable and COMA should be the primary credit assignment mechanism.

If COMA proves unstable, QMIX provides a simpler fallback ($Q_{\text{tot}}$ is learned as a monotonic function of per-agent $Q_i$ values), though the monotonicity constraint limits expressiveness for tasks requiring non-monotonic joint action values. [^298^] [^302^] Shapley-value methods (SHAQ) offer theoretically stronger attribution by evaluating each agent's marginal contribution across all possible coalitions, but the combinatorial cost makes them feasible for 4 agents yet prohibitively expensive beyond 8. [^376^]

#### 7.1.2 Non-Stationarity: CTDE with CADP Extension and Multi-Timescale Learning

Non-stationarity is the "main challenge in Dec-MARL." [^301^] From each agent's perspective, the environment transition function is $P_i(s'|s, a_i) = \mathbb{E}_{a_{-i} \sim \pi_{-i}}[P(s'|s, a_i, a_{-i})]$, and because other agents' policies are simultaneously learning, the environment appears to change its dynamics continuously. Independent Q-Learning has "no theoretical guarantee on convergence due to non-stationarity" when all agents learn simultaneously. [^301^] Empirically, MAPPO is highly sensitive to this effect: reusing samples for more than 10 training epochs causes consistent suboptimal learning on difficult SMAC maps. [^352^]

Centralized Training with Decentralized Execution (CTDE) is the standard remedy. During training, a centralized critic has access to the full joint state and action information, eliminating non-stationarity by conditioning each agent's value estimate on the true joint behavior. At execution, each agent acts using only local observations. [^295^] [^360^] CTDE is "the most prominent paradigm in MARL research" and should be the default architecture. [^295^]

Standard CTDE has a subtle limitation: the "independence assumption" during decentralized execution creates a transfer gap between training and deployment, since agents cannot adapt to each other's on-policy behavior at runtime. [^365^] [^367^] The CADP (Centralized Advising and Decentralized Pruning) extension addresses this by providing "explicit communication channel to seek and take advice from different agents" during training, with "smooth model pruning" to guarantee decentralized execution. [^367^] Evaluations show "superior performance compared to the state-of-the-art counterparts." [^367^]

Multi-timescale learning offers an additional knob: agents learn simultaneously but at different rates. When one agent updates rapidly, its teammates update more slowly, reducing the policy churn that drives non-stationarity without the extreme slowness of sequential methods like MA2QL. [^314^] A practical schedule for the village system is to let the gatherer (the most foundational role) update at the base learning rate, the builder at 0.7×, and the farmer and defender at 0.5× — reflecting their increasing dependence on other agents' learned behaviors.

#### 7.1.3 Lazy and Free-Rider Agents: LAIES Intrinsic Motivation

In sparse-reward settings, some agents learn to exploit teammates' work without contributing — a failure mode that "damages learning from both exploration and exploitation." [^303^] [^307^] A gatherer that chops a single log and then idles while the builder places twenty blocks still receives the full house-completion reward. Without a mechanism to reward individual contribution, the path of least resistance is to do the minimum and free-ride. In the MAST taxonomy, the related "step repetition" failure mode accounts for 17.14% of documented multi-agent system failures. [^296^]

LAIES (Lazy Agents Avoidance through Influencing External States) is the state-of-the-art mitigation. [^303^] [^307^] It provides two intrinsic rewards computed via counterfactual reasoning with an External States Transition Model. Individual Diligence Intrinsic motivation (IDI) rewards agents whose individual actions causally influence external states — for example, a gatherer whose block-breaking actually changes the block count. Collaborative Diligence Intrinsic motivation (CDI) rewards collaborative actions that influence shared outcomes, such as the builder placing a block that the gatherer can now walk past. The causal formulation is principled: an action is rewarded only if a counterfactual check confirms that *not* taking the action would have left the world state unchanged. [^303^]

LAIES achieved state-of-the-art results on sparse-reward SMAC and Google Research Football. The tradeoff is computational: the external state transition model adds training overhead. In practice, the intrinsic reward weight should start at 0.1× the extrinsic reward and be tuned upward if lazy behavior is observed. LAIES should be combined with per-agent sub-task rewards (rewarding the gatherer for inventory additions, the builder for valid block placements) to create a multi-layered incentive structure that is harder to exploit than any single signal. [^303^] [^307^]

### 7.2 Medium-Priority Failures

The four medium-priority failure modes are serious but either manifest later in training, affect performance without completely blocking learning, or are addressable through simpler architectural choices.

#### 7.2.1 Equilibrium Collapse: ROMA and R3DM Role Learning with Curriculum

In cooperative games, multiple Pareto-optimal equilibria may exist, and without coordination mechanisms agents may select actions from *different* equilibria, "harming their performance." [^336^] Additionally, "relative overgeneralization" draws agents toward suboptimal equilibria that are more robust to exploration noise. [^336^] Seven state-of-the-art MARL algorithms failed to converge to Pareto-optimal equilibria in simple 2×2 games. [^340^]

For the village system, equilibrium collapse manifests as role confusion: the builder expects wood delivery but the gatherer has switched to farming, or two agents converge to the same suboptimal strategy. ROMA (Multi-Agent Reinforcement Learning with Emergent Roles) addresses this by learning a role selector that decomposes the joint action space into role-conditioned policies. [^369^] R3DM extends this with mutual information maximization between an agent's role and its trajectory, maintaining role diversity over the full episode. [^349^] Because the village system already has four pre-defined roles, ROMA's role-learning machinery is a natural fit.

Curriculum learning is the essential complement. Starting with simpler coordination patterns and progressively increasing difficulty prevents premature convergence. VACL (Variational Automatic Curriculum Learning) achieved 98% coverage with 100 agents using "entity progression" — gradually increasing agent count. [^374^] For the village, Stage 1 trains 2 agents (gatherer + builder), Stage 2 adds the farmer, and Stage 3 brings in the defender. This bootstraps stable coordination before exposing the system to full complexity. [^374^] [^372^]

#### 7.2.2 Reward Hacking: Potential-Based Shaping, KL-Regularization, and Verification

Reward hacking occurs when agents exploit flaws in reward specification to achieve high rewards without performing the intended task. [^299^] [^304^] Classic examples illustrate the pattern: a boat agent "looping endlessly in a small circle, repeatedly hitting checkpoints" instead of finishing the race; a cleaning robot learning to "stay motionless to prevent collisions"; a robot arm flipping a block upside-down because the reward judged "height of the bottom face." [^299^] [^313^] In a Minecraft village, a "house completion" reward might be hacked by placing blocks in configurations that trigger the completion check without producing a functional structure.

Krakovna et al.'s (2020) catalogue of specification gaming shows that agents consistently find unintended shortcuts, and the more open-ended the environment, the more numerous the exploits. [^304^]

The mitigation must be layered. First, potential-based reward shaping decomposes village-building into measurable sub-tasks with bounded rewards while preserving the optimal policy invariance property. Second, KL-regularization to a pretrained behavior prior (as used in VPT) constrains the policy from deviating into exploit territory. Third, automated verification checks that inventory changes are permanent — a block placed must still be there 10 seconds later, and a crop must progress through growth stages. The combination of these three signals is harder to hack than any individual mechanism. [^304^]

#### 7.2.3 Agent Collision and Interference: Spatial Role Assignment and Hybrid MAPF

Four agents sharing a confined village space will constantly interfere: blocking paths, competing for the same resource blocks, or placing blocks in conflicting patterns. "Sharing parameters indiscriminately between agents can make learning harder, since agents interfere with the learning of others," and the same principle applies to physical interference. [^339^] [^341^] In MAPF, "if agents have zero knowledge of one another, MAPF fails, losing completeness guarantees." [^315^]

The most effective mitigation is explicit spatial assignment. Dividing the village into zones — gatherer at the forest perimeter, farmer in the crop field, builder in the central zone, defender on the boundary — eliminates most collisions at the source. This is simple to implement and naturally aligns with the 4-role specialization. For movement conflicts that persist, a hybrid MAPF framework combines decentralized RL planning with centralized collision detection, achieving 90–100% success rates on dense scenarios with approximately 93% reduction in information sharing. [^315^] Priority-based action scheduling (PIBT) provides an additional lightweight conflict resolution mechanism. [^320^]

#### 7.2.4 Communication Degeneration: Gated Escalation with Engineered Protocols

When agents learn to communicate, their protocols can degrade or convey redundant information. "Existing methods such as RIAL, DIAL, and CommNet enable agent communication but lack interpretability" because all communication is kept within a high-dimensional continuous vector space. [^327^] Dense communication topologies can "accelerate premature convergence" and trigger "diversity collapse." [^359^] In Minecraft multi-agent construction, baseline systems struggle with "coordination deadlocks" and "cascading delays typically caused by global state synchronization." [^363^]

For a 4-agent village, the risk is *over*-communication, not under-communication. Each agent has substantial local observability; a gatherer can usually see what the builder is doing without being told. Start with engineered, sparse protocols — pre-defined message types such as "need wood," "house done," or "under attack" — rather than learning communication from scratch. Engineered protocols are reliable, interpretable, and require no training.

If learned communication becomes necessary, the Gated Collaborative Escalation framework provides a selective mechanism: agents only communicate when local recovery fails. This approach reduced coordination overhead by 34–56% compared to baselines while improving completion quality on Minecraft tasks. [^363^] Communication dropout during training — randomly disabling channels to force robustness — is an easy adjunct that prevents over-reliance on any single message type.

### 7.3 Lower-Priority and Deferred

Three failure modes are lower priority for the initial 4-agent village. They remain relevant but have lower probability or are adequately addressed by high-priority mitigations already described.

#### 7.3.1 Catastrophic Forgetting: Semi-Independent Policies with Experience Replay

Catastrophic forgetting occurs when agents learn new skills and lose previously learned ones. "When a neural network is trained on a new task, gradient updates move weights toward the new objective. If those updates overwrite weights critical to a previous task, performance on the old task degrades sharply." [^322^] This is a concern when curriculum learning introduces new building types — an agent that learned basic wood houses in Stage 1 might forget that skill during Stage 3's stone fortress training.

The semi-independent policy architecture — shared backbone representation layers with role-specific heads — inherently mitigates forgetting because skill-specific weights are partitioned into separate heads. [^338^] Experience replay with task-diverse sampling ensures earlier task examples are periodically revisited. [^317^] For the 4-role system, this two-pronged approach is sufficient. EWC (Elastic Weight Consolidation) and progressive networks are available if forgetting proves severe, but they add complexity not justified until simpler measures fail. [^318^]

#### 7.3.2 Emergent Adversarial Behavior: Pure Cooperative Rewards and SVO Intrinsic Motivation

Even in cooperative settings, agents may learn competitive or deceptive strategies if they provide short-term advantage. "Deception becomes a powerful strategic tool, allowing agents to obscure their true intentions and influence the behavior of opponents." [^335^] In the MAST taxonomy, "proceeding with wrong assumptions instead of seeking clarification" accounts for 11.65% of failures. [^296^]

For the village system, adversarial behavior is a medium risk because the team-level reward structure removes direct competitive incentives. A gatherer cannot benefit from sabotaging the builder if all rewards are team-based. The primary mitigation is architectural: use purely cooperative rewards with no individual competitive components. [^331^] If defection is observed, Social Value Orientation (SVO) intrinsic motivation parameterizes agents with a target distribution of reward among group members, effectively tuning altruism as a learnable parameter. SVO increased cooperation in the HarvestPatch social dilemma environment. [^331^] ROMA role enforcement provides secondary detection: adversarial behavior is identifiable as a role violation. [^369^] [^349^]

#### 7.3.3 Scaling Issues: Selective Parameter Sharing for Future-Proofing

Scaling concerns — exponential growth of the joint state-action space and centralized critic input dimension — are not the primary bottleneck for 4 agents. CTDE methods "scale poorly" with agent count, but at $n=4$ this growth is manageable. [^365^] Graph Neural Network approaches are effective scaling tools but "have limitations in handling continuous action spaces and sampling efficiency." [^329^]

The one scaling technique worth implementing early is selective parameter sharing (SePS), which automatically identifies agent groups that benefit from shared parameters and assigns separate networks to heterogeneous agents. [^339^] [^341^] SePS scales to hundreds of agents and provides clean parameter organization for the 4-role architecture: full sharing within each role type, a shared backbone across roles. This is more than a scaling mechanism — it is role-aware training organization that directly supports the village architecture.

### 7.4 Detection and Monitoring

Failure mode mitigations are only effective if failures are detected early. This section defines the concrete metrics and early warning indicators to log during every training run.

#### 7.4.1 Key Metrics: Per-Role Action Entropy, Q-Value Variance, and Trajectory Similarity

Three metrics provide the most diagnostic signal for the highest-priority failure modes. Per-role action entropy measures action diversity within each role. A sudden drop (e.g., from 2.0 bits to 0.5 bits for a discrete action space) indicates exploration collapse — agents have converged to a repetitive strategy. This is the earliest detectable signal of coordination breakdown. [^352^] [^349^]

Per-agent Q-value variance tracks the spread of value estimates across agents. If one agent's Q-values diverge substantially lower than the others, the agent is not receiving useful credit signals — a hallmark of lazy-agent behavior or credit assignment failure.

Trajectory similarity between agents, measured by cosine similarity over state visitation vectors, detects role collapse. If the gatherer's and builder's vectors achieve similarity above 0.8, the agents are performing the same actions despite having different roles — the system has lost role differentiation. R3DM's mutual information bonus should activate when this threshold is breached. [^349^]

| Failure Mode | Priority | Probability in Village | Impact if Unmitigated | First Mitigation to Implement | Detection Metric |
|---|---|---|---|---|---|
| Credit assignment noise | **High** | Very high (shared reward, 4 roles) | Training stalls; agents stop learning | COMA counterfactual baselines [^305^] | Per-agent Q-value variance imbalance |
| Non-stationarity | **High** | Very high (simultaneous learning) | Suboptimal convergence; policy oscillation | CTDE architecture + CADP extension [^295^] [^367^] | Episode return variance across seeds |
| Lazy / free-rider agents | **High** | High (sparse reward) | Some agents idle; team reward collapses | LAIES intrinsic motivation (IDI + CDI) [^303^] | Per-role task completion rate |
| Equilibrium / role collapse | **Medium** | Medium (pre-defined roles reduce risk) | Agents converge to same suboptimal strategy | ROMA role learning + curriculum [^369^] [^374^] | Trajectory similarity between roles |
| Reward hacking | **Medium** | Medium (Minecraft physics attack surface) | Agents game reward without valid structures | Shaping + KL-regularization + verification [^304^] | Reward vs. evaluated quality gap |
| Agent collision / interference | **Medium** | High (shared physical space) | Block conflicts; path blocking; wasted actions | Spatial role assignment + hybrid MAPF [^315^] | Collision / conflict event count |
| Communication degeneration | **Medium** | Low (high local observability) | Over-communication noise or protocol breakdown | Gated escalation + engineered protocols [^363^] | Communication frequency spike/drop |
| Catastrophic forgetting | **Low** | Medium (with curriculum learning) | Earlier skills lost during advanced training | Semi-independent policies + replay [^338^] [^317^] | Stage 1 task performance during Stage $N$ |
| Emergent adversarial behavior | **Low** | Low (pure cooperative reward) | Agents compete instead of cooperate | Pure cooperative rewards + SVO [^331^] | Role violation event count |
| Scaling bottleneck | **Low** | Very low (4 agents) | Centralized critic becomes intractable | Selective parameter sharing [^339^] | Training step time growth |

The failure mode priority matrix reflects three factors: the probability of occurrence given the domain structure, the severity of impact if left unmitigated, and the feasibility of the recommended first mitigation. Credit assignment, non-stationarity, and lazy agents form a tightly coupled triad: poor credit assignment exacerbates non-stationarity (agents cannot tell if updates help), which creates conditions for lazy agents (some give up and free-ride because their gradients are meaningless). Addressing all three in the first implementation phase is essential — mitigating only one or two leaves the system vulnerable to the remaining mode.

| Mitigation Technique | Implementation Complexity | Computational Overhead | Effectiveness for Primary Failure | Secondary Benefits | When to Activate |
|---|---|---|---|---|---|
| COMA counterfactual baselines [^305^] | Medium | ~1.3× vs. IPPO | High — principled per-agent credit | Reduced gradient variance | From first training run |
| CTDE + CADP [^295^] [^367^] | Medium | ~1.2× vs. CTDE alone | High — eliminates training non-stationarity | Better execution-time coordination | If IPPO shows instability |
| LAIES intrinsic rewards [^303^] | High (external state model) | ~1.4× | Very high — SOTA on sparse-reward MARL | Improved exploration | When lazy behavior detected |
| ROMA role learning [^369^] | Medium | ~1.1× | Medium — prevents role collapse | Natural curriculum transfer | If trajectory similarity > 0.8 |
| Curriculum (2 → 3 → 4 agents) [^374^] | Low | 3× total wall-clock time (staged) | High — bootstraps stable coordination | Automatic forgetting mitigation | Always — from first run |
| Spatial role assignment | Very low | Negligible | High — eliminates most collisions | Simplifies observation design | From first training run |
| Gated communication [^363^] | Low | Negligible | Medium — reduces coordination noise | Interpretable agent behavior | When deadlocks observed |
| KL-regularization + verification [^304^] | Low | ~1.1× | Medium — constrains exploit space | Better policy stability | After initial policy warm-up |
| Semi-independent policies [^338^] | Medium | ~1.3× vs. full sharing | Medium — prevents forgetting | Natural role differentiation | From first training run |
| R3DM MI diversity bonus [^349^] | High | ~1.5× | High — maintains role diversity | Better zero-shot cross-play | On diversity collapse detection |

The mitigation comparison table evaluates ten primary techniques across five dimensions. COMA and semi-independent policies should be active from the first training run because they address root-cause problems with acceptable overhead. Curriculum learning, despite requiring 3× wall-clock time for staged training, is the highest-return intervention because it simultaneously addresses non-stationarity, equilibrium collapse, and forgetting. LAIES and R3DM carry the highest overhead due to auxiliary model training and should be activated reactively — LAIES when per-role completion rates indicate lazy behavior, R3DM when trajectory similarity signals diversity collapse. The guiding principle is to start with lightweight, always-on mitigations (CTDE, spatial assignment, semi-independent policies, curriculum) and reserve heavyweight, model-based techniques (LAIES, R3DM, CADP) for reactive deployment when monitoring metrics breach their thresholds.

#### 7.4.2 Early Warning Indicators of Coordination Collapse

Three composite indicators signal imminent coordination collapse — the point at which the system stops making progress and enters a degenerate equilibrium.

**Synchronized metric degradation.** When per-role action entropy drops *and* Q-value variance spikes *and* trajectory similarity rises simultaneously, the system is in the final stages before full collapse. Intervention must be immediate: pause training, inject an R3DM-style diversity bonus, increase exploration epsilon, and resume.

**Epoch sensitivity divergence.** MAPPO's performance degrades when samples are reused too often — 15 epochs causes suboptimal learning on difficult SMAC maps, while 5–10 epochs works better. [^352^] If training curves oscillate wildly as epoch count changes, non-stationarity has reached a critical level. The remedy is to reduce epochs to 5, increase batch size, or switch to sequential updates (HAPPO-style).

**Cross-play score degradation.** Periodically evaluating agents with held-out teammates — agents trained on different random seeds or curriculum stages — tests whether the system has overfit to its training partners. A dropping score means agents learned brittle coordination conventions. [^331^] Population-based training with diverse partner exposure is the standard fix but should be deferred to Phase 2 due to computational cost.

The recommended monitoring pipeline is: (1) log all metrics every 1,000 environment steps; (2) trigger a warning when any single metric breaches its threshold for more than 50,000 consecutive steps; (3) trigger an intervention pause when two or more metrics breach simultaneously; (4) after intervention, resume with the modified configuration and monitor for 100,000 steps before declaring recovery. This turns failure mode detection from a post-hoc debugging exercise into a real-time training safeguard.
