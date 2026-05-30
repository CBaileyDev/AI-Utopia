# Dim 6: Hierarchical RL & Hybrid Architectures

## Executive Summary

This document surveys hierarchical reinforcement learning (HRL) and hybrid planner+RL architectures for long-horizon open-world multi-agent systems, with focus on a Minecraft village simulation with 5 specialized roles. We cover the Options framework, Feudal Networks, HAC/HIRO, HTN/GOAP planners, LLM-as-planner approaches, and empirical tradeoffs. The key finding is that **a hybrid architecture combining an LLM or HTN high-level planner with learned low-level policies** offers the best balance of sample efficiency, debuggability, and generalization for multi-agent Minecraft villages, consistent with the emergent consensus from Voyager, SayCan, OpenHA, and recent HDRL-for-Minecraft work.

---

## 6.1 Options Framework

### Core Theory

Claim: The Options framework defines temporally extended actions as triples <I, π, β> where I is an initiation set, π is an internal policy, and β is a termination condition, enabling hierarchical decision-making over macro-actions. [^1^]
Source: "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning"
URL: https://dl.acm.org/doi/10.5555/645848.668804
Date: 1999
Excerpt: "Options (Sutton et. al. 1999, Precup 2000) is one among the various frameworks within Hierarchical RL. An option is a temporarily extended action... An option is defined as a tuple <I; π; β>: I: Initiation Set; π: Policy; β: Termination probability"
Context: Foundational paper establishing the mathematical framework for temporal abstraction in RL. Options allow an agent to learn a policy over options rather than primitive actions.
Confidence: high

Claim: The Option-Critic architecture (Bacon et al., 2017) enables fully end-to-end learning of options jointly with a policy-over-options, extending the policy gradient theorem to options and achieving performance comparable to or better than DQN while learning interpretable specialized options. [^2^]
Source: "The Option-Critic Architecture" (AAAI 2017)
URL: https://ojs.aaai.org/index.php/AAAI/article/view/10916/10775
Date: 2017
Excerpt: "The eight options learned in each game are learned fully end-to-end, in tandem with the feature representation, with no prior specification of a subgoal or... using ideas from policy gradient methods, option-critic provides continual option construction, can be used with nonlinear function approximators"
Context: Option-Critic uses actor-critic methods to learn intra-option policies, termination functions, and the policy over options simultaneously. It was the first scalable end-to-end option learning framework.
Confidence: high

Claim: End-to-end option learning on continuous control tasks is feasible but raises questions about when options are beneficial - performance gains are linked to compositional structure in the environment rather than being universal. [^3^]
Source: "Learnings Options End-to-End for Continuous Action Tasks"
URL: https://arxiv.org/pdf/1712.00004
Date: 2017
Excerpt: "Our results also suggest that the increase in performance is not directly linked to the deliberation cost... the increase in performance is related to the compositionality of the environment. In the classic Mujoco environments, using options is not as beneficial as using them in a customized environment with a more obvious division in the state-space."
Context: Klissarov et al. extended Option-Critic to continuous actions with PPO, finding that environments with clear state-space divisions benefit more from options.
Confidence: high

Claim: Recent advances in intra-option learning enable updating all options consistent with current primitive action choices simultaneously, significantly improving performance and data-efficiency. [^4^]
Source: "Flexible Option Learning" (arXiv 2112.03097)
URL: https://arxiv.org/abs/2112.03097
Date: 2021
Excerpt: "We revisit and extend intra-option learning in the context of deep reinforcement learning, in order to enable updating all options consistent with current primitive action choices, without introducing any additional estimates... we obtain significant improvements in performance and data-efficiency across a wide variety of domains."
Context: Builds on Sutton, Precup & Singh's original intra-option learning to make it compatible with deep RL, enabling parallel updates of all relevant options.
Confidence: high

### What is Delegated to Planner vs Policy

In the Options framework:
- **Planner (policy over options)**: Selects which option to execute at each decision point
- **Policy (intra-option policy)**: Executes primitive actions within an option
- **Terminations (β)**: Learned or hand-specified conditions for ending an option
- **Initiation sets (I)**: Define which options are available in which states

### For Minecraft Village Application

Claim: A three-level HDRL agent for Minecraft using Option-Critic for the high-level planner achieves substantially better task completion and sample efficiency than monolithic RL, particularly in long-horizon reward-sparse scenarios. [^5^]
Source: "A Minecraft Agent Based on a Hierarchical Deep Reinforcement Learning Model"
URL: https://www.ijcai.org/proceedings/2022/0452.pdf / https://zenodo.org/records/17474560
Date: 2022/2025
Excerpt: "Results indicate that HDRL substantially enhances task completion rates and sample efficiency compared to monolithic RL agents, particularly in longhorizon and reward-sparse scenarios... A three-level hierarchy—planner (options/subtasks), skill controllers (subgoals), and a pixel-to-action low-level policy."
Context: Panwar et al. designed a specific HDRL agent for Minecraft combining Option-Critic, FuN, HIRO, and HAC approaches across three levels.
Confidence: high

---

## 6.2 Feudal Networks & Hierarchical Value Functions

### Core Theory

Claim: Feudal Networks (FuN) employ a Manager module operating at lower temporal resolution that sets abstract goals in latent space, and a Worker module generating primitive actions at every environment tick, decoupling end-to-end learning across multiple levels. [^6^]
Source: "FeUdal Networks for Hierarchical Reinforcement Learning" (ICML 2017)
URL: https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf
Date: 2017
Excerpt: "Our framework employs a Manager module and a Worker module. The Manager operates at a lower temporal resolution and sets abstract goals which are conveyed to and enacted by the Worker. The Worker generates primitive actions at every tick of the environment... FuN dramatically outperforms a strong baseline agent on tasks that involve long-term credit assignment or memorisation."
Context: Inspired by Dayan & Hinton's 1993 feudal RL proposal. The Manager sets directional goals in a learned latent space, and the Worker interprets these via a goal-conditioned policy.
Confidence: high

Claim: The feudal approach (Manager-Worker decomposition) originated with Dayan and Hinton in 1993 as a way to address long-term credit assignment by having a "boss" (Manager) set tasks for "workers" at different spatial resolutions. [^7^]
Source: "Feudal Reinforcement Learning" (NIPS 1993) - cited in Vezhnevets et al. 2017
URL: https://arxiv.org/abs/1703.01161
Date: 1993/2017
Excerpt: "Our approach is inspired by the feudal reinforcement learning proposal of Dayan and Hinton, and gains power and efficacy by decoupling end-to-end learning across multiple levels -- allowing it to utilise different resolutions of time."
Context: Dayan & Hinton's original feudal RL used spatial hierarchies where managers assigned subgoals to workers based on geographic regions.
Confidence: high

Claim: Feudal approaches represent sub-goals as directions in latent state space which translate into meaningful behavioral primitives, with the decoupled structure facilitating very long timescale credit assignment and encouraging emergence of sub-policies associated with different Manager goals. [^8^]
Source: "FeUdal Networks for Hierarchical Reinforcement Learning" - blog review
URL: https://minkyuchoi-07.github.io/2023/03/20/feudal/
Date: 2017
Excerpt: "FuN represents sub-goals as directions in latent state space which then translate into meaningful behavioural primitives... the decoupled structure of FuN conveys several benefits -- in addition to facilitating very long timescale credit assignment it also encourages the emergence of sub-policies."
Context: The Manager outputs continuous goal vectors that the Worker uses to condition its action selection, learned end-to-end via intrinsic rewards.
Confidence: high

### What is Delegated to Planner vs Policy

In Feudal Networks:
- **Planner (Manager)**: Sets abstract directional goals in latent space at coarse temporal resolution
- **Policy (Worker)**: Generates primitive actions conditioned on current state AND the Manager's goal vector
- **Goal representation**: Learned latent space directions (not explicit symbolic subgoals)

---

## 6.3 HAC, HIRO & Modern Hierarchical RL

### HAC (Hierarchical Actor-Critic)

Claim: HAC (Levy et al., 2017) enables learning multiple levels of policies in parallel by training each level independently of lower levels, using hindsight experience replay to overcome instability - it was the first framework to successfully learn 3-level hierarchies in parallel for continuous state/action spaces. [^9^]
Source: "Learning Multi-Level Hierarchies with Hindsight" (arXiv 1712.00948)
URL: https://arxiv.org/abs/1712.00948
Date: 2017
Excerpt: "Hierarchical agents have the potential to solve sequential decision making tasks with greater sample efficiency than their non-hierarchical counterparts... HAC is the first to successfully learn 3-level hierarchies in parallel in tasks with continuous state and action spaces."
Context: HAC builds on DDPG, UVFA, and HER. Each level is trained as if lower-level policies are already optimal, enabling parallel multi-level learning.
Confidence: high

Claim: HAC's key insight is to train each hierarchy level independently by assuming lower-level policies are already optimal during training, combined with hindsight experience replay that re-labels failed subgoal attempts as successful ones using achieved states. [^10^]
Source: "Hierarchical Actor-Critic" (HRL paper summary)
URL: https://blogs.cuit.columbia.edu/p/hierarchical_actor-critic/
Date: 2017/2019
Excerpt: "The main idea behind HAC is to train each level of the hierarchy independently of the lower levels by training each level as if the lower level policies are already optimal... our framework is the first to successfully learn 3-level hierarchies in parallel."
Context: HAC enables faster learning by dividing work among multiple policies and exploring at higher levels. Tested on gridworld and simulated robotics.
Confidence: high

### HIRO (Data-Efficient HRL)

Claim: HIRO (Nachum et al., 2018) introduces an off-policy correction that enables both higher- and lower-level training from the same replay buffer, achieving substantial data efficiency improvements - learning complex robot behaviors from "only a few million samples, equivalent to a few days of real-time interaction." [^11^]
Source: "Data-Efficient Hierarchical Reinforcement Learning" (NeurIPS 2018)
URL: http://papers.neurips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf
Date: 2018
Excerpt: "We propose to use off-policy experience for both higher- and lower-level training... we introduce an off-policy correction to remedy this challenge... HIRO can be used to learn highly complex behaviors for simulated robots, learning from only a few million samples."
Context: HIRO uses state differences as subgoals (no goal representation learning needed) and introduces a crucial off-policy correction for non-stationarity when lower-level policies change.
Confidence: high

Claim: HIRO's subgoal representation uses state-space differences rather than latent goal vectors, which means the higher-level policy receives meaningful supervision from task reward immediately without needing goal representation training. [^12^]
Source: "Data-Efficient Hierarchical Reinforcement Learning" / The Gradient review
URL: https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
Date: 2018/2019
Excerpt: "The main contribution of HIRO is that the method is very sample efficient compared to previous works thanks to a novel off-policy correction and the fact that the learning algorithm directly uses the state observation as the goal. There is no goal representation, hence no goal representation training needed."
Context: HIRO outperforms previous HRL methods substantially. The off-policy correction addresses the fundamental non-stationarity problem in hierarchical RL.
Confidence: high

### Modern HRL (2022-2025)

Claim: Recent work (RLA, 2025) introduces a principled approach where a high-level "anticipation model" learns to generate optimal waypoints by enforcing geometric consistency on the agent's value function (triangle inequality), with theoretical convergence guarantees to globally optimal policies. [^13^]
Source: "A Hierarchical Approach for Long-Horizon Tasks" (arXiv 2509.05545)
URL: https://arxiv.org/html/2509.05545v1
Date: 2025
Excerpt: "The anticipation model learns by inspecting the agent's value function... a smart waypoint from a starting point to a destination is one that lies on the most direct route... the travel time from the start to the destination is precisely the sum of the time from the start to the waypoint and the time from the waypoint to the destination."
Context: RLA provides a mathematical framework for what constitutes a good subgoal, trains a navigator (anticipation model) and motor controller (low-level policy) separately with principled objectives.
Confidence: high

Claim: DHRL (NeurIPS 2022) uses a graph to decouple high-level and low-level horizons, solving long-horizon tasks that previous HRL methods fail on, with the insight that coupling horizons creates a fundamental tradeoff that degrades either high-level or low-level performance. [^14^]
Source: "DHRL: A Graph-Based Approach for Long-Horizon and Complex Tasks" (NeurIPS 2022)
URL: https://proceedings.neurips.cc/paper_files/paper/2022/file/58b286aea34a91a3d33e58af0586fa40-Paper-Conference.pdf
Date: 2022
Excerpt: "Previous HRL methods cannot solve long-horizon tasks. This is likely due to the coupling of the high-level horizon and the low-level horizon resulting in increasing burden on either the high-level or the low-level policy... DHRL is the only algorithm that can succeed in complex environments (AntMazeComplex and AntMazeBottleneck)."
Context: DHRL achieves 95.1% success on 12x12 maze vs 43-88% for HRAC baselines, by allowing the high-level policy to operate at extended temporal abstraction via a graph.
Confidence: high

Claim: SOL (Scalable Option Learning, 2025) is the first online hierarchical RL algorithm to scale to billions of samples, surpassing flat baselines on NetHack and learning diverse option behaviors coordinated by a controller. [^15^]
Source: "Scalable Option Learning in High-Throughput Environments"
URL: https://arxiv.org/html/2509.00338v3
Date: 2025
Excerpt: "This work introduces, to our knowledge, the first online hierarchical RL algorithm which is able to scale to billions of samples... hierarchical systems execute a sequence of policies... any given slice will likely correspond to several different policies, making parallelization difficult."
Context: SOL addresses the systems-level challenge of parallelizing hierarchical RL training across GPUs, demonstrating HRL at the scale of flat RL for the first time.
Confidence: high

---

## 6.4 HTN & GOAP Planners

### Hierarchical Task Networks (HTN)

Claim: HTN planning solves complex problems by breaking them down into structured subtasks until they become primitive actions, using tasks (abstract/primitive), methods (decomposition rules), operators (primitive actions with preconditions/effects), and task networks (structured dependency graphs). [^16^]
Source: "Hierarchical Task Network (HTN) Planning in AI" (GeeksforGeeks)
URL: https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/
Date: 2026
Excerpt: "Hierarchical Task Network (HTN) Planning is an AI planning approach that solves complex problems by breaking them down into smaller, structured subtasks until they become primitive actions... Tasks: Units of work classified as abstract or primitive. Methods: Rules that describe how an abstract task can be decomposed."
Context: HTN is widely used in game AI (Guerrilla Games' Horizon Zero Dawn/Forbidden West), robotics, and logistics. Requires extensive domain knowledge.
Confidence: high

Claim: Guerrilla Games uses HTN planning in DECIMA engine for Horizon Zero Dawn/Forbidden West NPC high-level decisions, with Prolog-style backtracking over preconditions and in-game debugging visualization for HTN decompositions. [^17^]
Source: "HTN Planning in Decima" (Guerrilla Games, AI and Games Conference 2024)
URL: https://www.guerrilla-games.com/read/htn-planning-in-decima
Date: 2024
Excerpt: "This talk is about DECIMA's implementation of Hierarchical Task Network (HTN) planning, the AI planner technology which drives the NPC's high-level decisions. We explain how our implementation performs backtracking (similar to Prolog) over preconditions."
Context: HTN at Guerrilla handles high-level NPC decision-making with explicit hierarchical decomposition and backtracking when preconditions fail.
Confidence: high

Claim: HTN takes strengths from both behavior trees (careful design power) and planners like GOAP (dynamic adaptability), but with more optimization opportunities than goal-planners because it starts from the root rather than from the goal. [^18^]
Source: Unity Discussion Forum - Fluid HTN
URL: https://discussions.unity.com/t/released-fluid-hierarchical-task-network-planner-ai-free/740036
Date: 2019
Excerpt: "An HTN planner is more like a dynamic behavior tree, in that it starts at present time at the start of a predefined hierarchy... HTN takes strengths from both behavior tree's careful design power and the dynamic adaptability in planners like GOAP."
Context: HTN is characterized as forward-search from root (unlike GOAP's backward search from goal), making it more predictable and optimizable.
Confidence: high

Claim: HTN enables multiple methods for the same compound task, allowing alternative approaches conditioned on world state - e.g., attacking with a tree trunk if available, otherwise throwing boulders. [^19^]
Source: "Exploring HTN Planners through Example" (Game AI Pro)
URL: https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter12_Exploring_HTN_Planners_through_Example.pdf
Date: Unknown
Excerpt: "Compound tasks are where HTN get their 'hierarchical' nature... If he has access to a tree trunk, he may run to his target and use it as a melee weapon... If no tree trunks are available, he can pull large boulders from the ground and toss them."
Context: HTN's method selection based on preconditions provides flexible adaptation without full re-planning from scratch.
Confidence: high

### GOAP (Goal-Oriented Action Planning)

Claim: GOAP, developed by Jeff Orkin for F.E.A.R. in the early 2000s, enables NPCs to dynamically plan action sequences by searching backward from goals to current state using A* through action precondition-effect graphs. [^20^]
Source: "NPC AI planning with GOAP" (Excalibur.js blog)
URL: https://excaliburjs.com/blog/goal-oriented-action-planning/
Date: 2024
Excerpt: "GOAP was developed by Jeff Orkin in the early 2000's while working on the AI system for F.E.A.R. The desire was to generate automated planning sequences for Enemies and NPCs... GOAP can be considered an alternative to classic behavioral trees."
Context: GOAP uses agents, goals, actions (with preconditions/effects), and state to generate plans. Each action has cost, preconditions, and effects.
Confidence: high

Claim: GOAP creates a fully explainable sequence of steps expressed through language, with the planner building a branching tree of available plans and selecting the optimal course based on custom factors like cost. [^21^]
Source: "NPC AI planning with GOAP" / "What is Goap?"
URL: https://goap.crashkonijn.com/readme/theory
Date: 2024
Excerpt: "GOAP begins with the agent evaluating its current state and the desired goal state. The agent then searches through a library of available actions to find a series of actions that will transform the current state into the desired goal state."
Context: GOAP's key advantage is dynamic replanning - if the environment changes, the agent can generate a new plan. It was used in F.E.A.R. and Horizon Zero Dawn.
Confidence: high

Claim: Horizon Zero Dawn transitioned from GOAP to HTN, reflecting a broader industry pattern where GOAP's emergent flexibility is traded for HTN's more predictable, designer-controllable behavior. [^22^]
Source: Unity Discussion Forum / Industry interviews
URL: https://discussions.unity.com/t/released-fluid-hierarchical-task-network-planner-ai-free/740036
Date: 2019
Excerpt: "There is a very insightful interview with Troy Humphreys over on ai game dev, where he goes into how they transitioned from HFSM to GOAP to HTN, and their reasoning behind it... GOAP can find more paths through a problem domain, and be more emergent, but at the potential cost of design ability and performance."
Context: Industry experience shows a progression: FSM → GOAP → HTN, with each transition trading some emergent capability for design control and performance.
Confidence: medium

### What is Delegated to Planner vs Policy in HTN/GOAP

In HTN/GOAP architectures:
- **Planner**: Full high-level planning - decomposes goals into action sequences, selects methods, handles replanning
- **Policy (primitive actions)**: Low-level execution of primitive actions (move, attack, pick up)
- **No learned policy at high level**: Everything above primitive actions is symbolic/planned
- **World state**: Maintained explicitly and updated by action effects

### Advantages and Limitations

Claim: HTN's primary advantages include scalable structure, human-like reasoning, reusability of task hierarchies, and efficient search through domain constraints; its limitations include requiring extensive domain knowledge, limited generalization beyond predefined decompositions, and difficulty in uncertain domains. [^23^]
Source: "Hierarchical Task Network (HTN) Planning in AI"
URL: https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/
Date: 2026
Excerpt: "Advantages: Scalable Structure, Human-Like Reasoning, Reusability, Flexibility, Efficient Search. Limitations: Requires Extensive Domain Knowledge, Limited Generalization, Method Selection Difficulty, Computational Load, Not Ideal for Uncertain Domains."
Context: HTN is best suited for domains where task structure is well-understood and relatively stable.
Confidence: high

---

## 6.5 LLM-as-Planner over Learned Policies

### Voyager (Minecraft)

Claim: Voyager uses GPT-4 as a high-level planner with three components: an automatic curriculum for exploration, an ever-growing skill library of executable code, and iterative prompting with environment feedback - achieving 3.3x more unique items and 15.3x faster tech tree unlocking than prior SOTA in Minecraft. [^24^]
Source: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
URL: https://arxiv.org/abs/2305.16291
Date: 2023
Excerpt: "Voyager consists of three key components: 1) an automatic curriculum that maximizes exploration, 2) an ever-growing skill library of executable code for storing and retrieving complex behaviors, and 3) a new iterative prompting mechanism... obtains 3.3x more unique items, travels 2.3x longer distances, and unlocks key tech tree milestones up to 15.3x faster than prior SOTA."
Context: Voyager is the canonical example of LLM-as-planner with learned low-level skills in Minecraft. Uses code as the action space (not low-level motor commands).
Confidence: high

Claim: Voyager's skill library enables zero-shot transfer to new Minecraft worlds - it can solve novel tasks from scratch using previously learned skills, while other techniques struggle to generalize. [^25^]
Source: "Voyager" (ICLR 2024)
URL: https://openreview.net/forum?id=ehfRiF0R3a
Date: 2023/2024
Excerpt: "Voyager is able to utilize the learned skill library in a new Minecraft world to solve novel tasks from scratch, while other techniques struggle to generalize... The skills developed by Voyager are temporally extended, interpretable, and compositional."
Context: The skill library pattern (persistent store + retrieval by embedding similarity) is a key architectural insight for lifelong learning.
Confidence: high

Claim: The separation between skill acquisition and skill reuse in Voyager demonstrates that you do not need fine-tuning for accumulation - a well-indexed code store plus a capable base model is sufficient for lifelong learning. [^26^]
Source: "Voyager: Skill Libraries as the Foundation for Lifelong AI Agent Learning" (research log)
URL: https://beancount.io/bean-labs/research-logs/2026/05/08/voyager-open-ended-embodied-agent-lifelong-learning
Date: 2023/2026
Excerpt: "Voyager shows that you do not need fine-tuning to get accumulation: a well-indexed code store plus a capable base model is sufficient. That is a strong argument for investing in the indexing and retrieval layer."
Context: This is an important architectural insight - the LLM planner doesn't need gradient updates; in-context learning with skill retrieval is enough.
Confidence: high

### SayCan (Robotics)

Claim: SayCan grounds LLM planning in robotic affordances by combining the LLM's probability that a skill is useful with a learned value function's probability that the skill will succeed, achieving 84% planning success on 101 real-world robotic tasks. [^27^]
Source: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"
URL: https://say-can.github.io/assets/palm_saycan.pdf
Date: 2022
Excerpt: "We use reinforcement learning as a way to learn language-conditioned value functions that provide affordances of what is possible in the world... SayCan achieves 84% planning success rate... the skills and their value functions can act as the language model's 'hands and eyes,' while the language model supplies high-level semantic knowledge."
Context: SayCan is the key reference for LLM+RL grounding. The LLM provides task-grounding (what's useful), value functions provide world-grounding (what's possible).
Confidence: high

Claim: SayCan's critical insight is that the LLM alone lacks contextual grounding - without affordance functions, performance drops from 84% to 67%, demonstrating that learned low-level policies/value functions are essential for physical grounding. [^28^]
Source: "Do As I Can, Not As I Say" - ablation experiments
URL: https://say-can.github.io/
Date: 2022
Excerpt: "Grounding the LLM in the real-world via affordances nearly doubles the performance over the non-grounded baselines... No VF achieves 67% planning success versus PaLM-SayCan's 84%."
Context: This validates the hybrid approach: LLM for high-level reasoning + learned value functions for grounding.
Confidence: high

### OpenHA (Hierarchical Agents in Minecraft)

Claim: OpenHA formalizes a two-level hierarchy where a high-level pretrained LLM-based agentic model generates abstracted actions, and a low-level policy acts as the action tokenizer to produce primitive executable actions - subsuming Vision-Language-Action models as a special case. [^29^]
Source: "OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft"
URL: https://arxiv.org/html/2509.13347v1
Date: 2023-2025
Excerpt: "A high-level pretrained LLM-based agentic model generates an abstracted action based on the task instruction and the current observation; second, a low-level policy acts as the action tokenizer... This two-level procedure can be formally expressed as: A ~ π_AR(·|obs,ins), a ~ π_policy(·|obs,A)"
Context: OpenHA provides a principled framework for hierarchical agents with abstracted actions ranging from symbolic skills to continuous trajectories to latent codes.
Confidence: high

### What is Delegated to Planner vs Policy in LLM-as-Planner

In LLM-as-planner architectures:
- **Planner (LLM)**: High-level task decomposition, subgoal selection, curriculum design, skill retrieval/selection
- **Policy (learned)**: Low-level visuomotor control, goal-conditioned movement, primitive action execution
- **Skill library**: Persistent store of reusable behaviors, indexed by embeddings
- **Value functions/affordances**: Ground LLM decisions in physical possibility

---

## 6.6 Empirical Tradeoffs: Sample Efficiency, Generalization, Debuggability

### Sample Efficiency

Claim: Hierarchical RL achieves improved sample efficiency in long-horizon sparse-reward tasks by reducing the effective decision horizon and enabling faster contraction and variance reduction in value function learning. [^30^]
Source: "A Hierarchical Approach for Long-Horizon Tasks" (RLA, arXiv 2509.05545)
URL: https://arxiv.org/html/2509.05545v1
Date: 2025
Excerpt: "By breaking a long-horizon task (with expected length L) into a series of short-horizon sub-tasks (with expected length K), RLA ensures that its low-level learner operates on problems with a much better (smaller) contraction factor... the convergence gap is inversely related to the expected time to reach the goal."
Context: Mathematical analysis shows HRL's sample efficiency gains come from two sources: faster contraction (better conditioned learning) and variance reduction (shorter-horizon targets).
Confidence: high

Claim: Hybrid imitation-reinforcement learning approaches require significantly fewer samples than hierarchical RL alone, combining the sample efficiency of imitation at the high level with RL exploration at the low level. [^31^]
Source: "Hierarchical Imitation and Reinforcement Learning" (ICML 2018)
URL: http://proceedings.mlr.press/v80/le18a/le18a.pdf
Date: 2018
Excerpt: "Compared to hierarchical RL, the hybrid algorithm requires significantly fewer samples at the LO level... Note that flat Q-learning does not learn anything meaningful in either experimental setting, due to a long planning horizon and sparse rewards."
Context: Hoang Le et al. demonstrate that flat RL fails entirely on long-horizon sparse reward tasks, while hierarchical approaches (both pure HRL and hybrid IL+RL) succeed. The hybrid approach achieves the best sample efficiency.
Confidence: high

Claim: Hierarchical imitation learning outperforms flat imitation learning with significant savings in expert labels, requiring only a fraction of low-level labels after subgoals have been mastered. [^32^]
Source: "Hierarchical Imitation and Reinforcement Learning"
URL: http://users.umiacs.umd.edu/~hal3/docs/daume18ilrl.pdf
Date: 2018
Excerpt: "h-DAgger requires most of its LO-level labels early during training and requests primarily HI-level labels after the subgoals have been mastered. As a result, h-DAgger requires only a fraction of LO-level labels compared to flat DAgger."
Context: The hierarchical approach efficiently queries expert at the right level - low-level labels when learning primitives, high-level labels once subgoals are established.
Confidence: high

Claim: Current HRL methods are typically trained on millions of samples only - several orders of magnitude less than flat RL agents trained on billions - suggesting HRL has not yet realized the full benefits of large-scale training. [^33^]
Source: "Scalable Option Learning in High-Throughput Environments"
URL: https://arxiv.org/html/2509.00338v3
Date: 2025
Excerpt: "existing hierarchical agents are typically trained on millions of samples only -- several orders of magnitude less data. Therefore, hierarchical RL has yet to realize the benefits of large-scale training, which has driven progress in many other areas."
Context: This suggests both a challenge (HRL hasn't been trained at scale) and an opportunity (HRL may benefit enormously from scaling).
Confidence: high

### Generalization and Transfer

Claim: Decomposition into invariant sub-policies over compressed or symbolic state representations allows rapid adaptation to new tasks with similar abstract structure. [^34^]
Source: "Hierarchical Reinforcement Learning" (Emergent Mind review)
URL: https://www.emergentmind.com/topics/hierarchical-reinforcement-learning
Date: 2025
Excerpt: "Transferability: Decomposition into invariant sub-policies over compressed or symbolic state representations allows rapid adaptation to new tasks with similar abstract structure... Skill adaptation methods that permit continued low-level skill updates during transfer avoid final performance plateaus."
Context: HRL's hierarchical structure naturally supports transfer - low-level skills (motor primitives) transfer across tasks, high-level policy adapts to new goals.
Confidence: high

### Debuggability

Claim: Hierarchical RL with options produces more interpretable behavior than flat RL because options naturally cluster behavior - for example, termination events cluster near doorways in the four-rooms domain, intuitively learning doors as subgoals. [^35^]
Source: "The Promise of Hierarchical Reinforcement Learning" (The Gradient)
URL: https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
Date: 2019
Excerpt: "Termination probabilities learnt by the option-critic agent with 4 options... termination events are more likely to occur near the doors, intuitively this means that reaching those doors are learnt as being meaningful sub-goals."
Context: The emergent structure of learned options provides interpretability that flat RL lacks.
Confidence: high

Claim: HTN planners offer superior debuggability through hierarchical decomposition visualization, backtracking flow analysis, and predictable search order from the root - Guerrilla Games uses generated C++ with in-game HTN debugging tools. [^36^]
Source: "HTN Planning in Decima" (Guerrilla Games)
URL: https://www.guerrilla-games.com/read/htn-planning-in-decima
Date: 2024
Excerpt: "We explain how our implementation performs backtracking (similar to Prolog) over preconditions and present a flow visualization which can help understand the backtracking flow... also touch on how we debug our HTN decompositions in-game."
Context: Explicit hierarchical structure makes HTN naturally debuggable - you can trace exactly why a particular decomposition was chosen.
Confidence: high

Claim: SayCan produces fully explainable plans expressed through natural language because each step combines LLM probabilities with value function scores, making it transparent why each skill was selected. [^37^]
Source: "Do As I Can, Not As I Say"
URL: https://arxiv.org/pdf/2204.01691
Date: 2022
Excerpt: "This combination results in a fully explainable sequence of steps that the robot will execute to accomplish an instruction -- an interpretable plan that is expressed through language."
Context: The LLM+value function combination is inherently interpretable since both components output human-readable scores.
Confidence: high

### Tradeoff Summary Table

| Dimension | Flat End-to-End RL | Options/Feudal/HAC/HIRO | HTN/GOAP Planner | LLM-as-Planner + RL |
|-----------|-------------------|------------------------|------------------|---------------------|
| Sample Efficiency | Poor (long horizon) | Good (HAC/HIRO) | N/A (not learned) | Good (skill reuse) |
| Generalization | Poor | Good (skill transfer) | Limited (handcoded) | Excellent (LLM reasoning) |
| Debuggability | Poor (black box) | Moderate (emergent options) | Excellent (explicit hierarchy) | Good (natural language plans) |
| Design Effort | Low | Medium | High (domain modeling) | Medium (skill engineering) |
| Real-time Performance | Fast inference | Fast inference | Fast planning | Slower (LLM calls) |
| Open-world Adaptation | Poor | Moderate | Limited | Excellent |
| Multi-agent Scaling | Hard | Moderate | Moderate | Good (decentralized) |

---

## 6.7 Consensus Pattern for "Find Distant Resource"

### The Universal Pattern

Across all architectures surveyed, the consensus pattern for "find a distant resource" is a **hierarchical waypoint decomposition**:

1. **High level**: Identify the target resource type and select a coarse approach strategy (e.g., "go to forest biome for wood" or "descend to Y-level for diamonds")
2. **Mid level**: Generate intermediate waypoints/subgoals that break the long journey into shorter segments
3. **Low level**: Navigate locally toward each waypoint using visuomotor control

### How Each Architecture Implements This

**Options Framework**:
- High-level policy selects options like [navigate-to-forest], [find-tree], [chop-wood]
- Each option runs until termination condition (reaching biome, finding tree, filling inventory)
- Intra-option policies handle local navigation

Claim: In the Options framework for Minecraft, skills like "finding objects" and "navigating to various locations" are naturally represented as temporally extended actions, with a controller determining when to reuse learned skills. [^38^]
Source: "A Deep Hierarchical Approach to Lifelong Learning in Minecraft" (EWRL 2016)
URL: https://ewrl.wordpress.com/wp-content/uploads/2016/11/ewrl13-2016-submission_20.pdf
Date: 2016
Excerpt: "In Minecraft, tasks, which can also be interpreted as skills include, building houses, finding objects and navigating to various locations in the game... learning skills and when to reuse the skills is key to accumulating knowledge, increasing exploration, efficiently solving tasks."
Context: Early work on deep hierarchical RL in Minecraft identified navigation and resource-finding as core skills to be learned as reusable options.
Confidence: high

**Feudal Networks**:
- Manager sets directional goals in latent space pointing toward the resource
- Worker executes movement primitives conditioned on the directional goal
- Manager updates direction at coarse temporal resolution

**HAC/HIRO**:
- High-level policy outputs subgoal states (waypoints) toward the resource
- Low-level policy learns to reach each subgoal
- Hindsight experience replay re-labels failed subgoal attempts

Claim: HIRO's use of state observations directly as goals means the high-level policy naturally generates waypoint sequences toward distant targets without needing hand-designed subgoal representations. [^39^]
Source: "Data-Efficient Hierarchical Reinforcement Learning"
URL: http://papers.neurips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf
Date: 2018
Excerpt: "The higher-level policy receives a meaningful supervision signal from the task reward at the outset... subgoals are represented as differences in state space, enabling efficient reuse of trajectories."
Context: HIRO's state-space goals make it particularly natural for navigation - the high-level policy literally picks intermediate states to visit.
Confidence: high

**HTN/GOAP**:
- Planner decomposes [ObtainResource] into [NavigateToBiome] → [LocateResource] → [HarvestResource]
- Methods handle different resource types with different precondition-effect chains
- Replanning occurs if path is blocked or resource depleted

**LLM-as-Planner (Voyager)**:
- LLM generates a plan: "go to forest" → "find tree" → "chop wood"
- Skill library retrieves previously learned [navigate], [find], [chop] code
- Iterative prompting refines the plan based on environment feedback

Claim: Voyager's automatic curriculum generates progressively harder exploration tasks calibrated to the agent's current state, enabling systematic distant resource discovery - e.g., learning to collect sand in desert before digging for iron. [^40^]
Source: "Voyager: An Open-Ended Embodied Agent"
URL: https://arxiv.org/abs/2305.16291
Date: 2023
Excerpt: "For example, the agent learns to collect sand and cactus in a desert before digging for iron... the automatic curriculum takes into account the exploration progress and the agent's state to maximize exploration."
Context: The curriculum acts as an emergent high-level planner for resource discovery, using LLM reasoning to sequence exploration goals.
Confidence: high

**Modern HRL (RLA/DHRL)**:
- Anticipation model generates waypoints satisfying geometric optimality (triangle inequality)
- Subgoals are points on the shortest path to the distant resource
- Low-level policy handles navigation between waypoints

Claim: RLA's anticipation model trained to enforce geometric consistency on the value function naturally produces waypoints that lie on optimal shortest paths, with theoretical convergence to optimal waypoint selection. [^41^]
Source: "A Hierarchical Approach for Long-Horizon Tasks"
URL: https://arxiv.org/html/2509.05545v1
Date: 2025
Excerpt: "A smart waypoint from a starting point to a destination is one that lies on the most direct route. Such a waypoint has a unique geometric property: the travel time from the start to the destination is precisely the sum of the time from the start to the waypoint and the time from the waypoint to the destination."
Context: This provides a principled mathematical foundation for what constitutes a good subgoal in navigation tasks.
Confidence: high

### Recommended Architecture for Multi-Agent Minecraft Village

Based on the survey, the recommended architecture for a multi-agent Minecraft village with 5 specialized roles combines:

1. **High-level**: LLM or HTN planner per role (farmer, miner, builder, guard, trader) that generates task plans from village goals
2. **Mid-level**: Learned options/skills for common subtasks (navigate, gather, craft, fight, trade) shared across roles via a skill library
3. **Low-level**: Goal-conditioned visuomotor policies for primitive actions (move, look, click, place)
4. **Coordination**: Hierarchical multi-agent framework with role specialization (ROMA-style emergent roles) at the high level

Claim: Hierarchical multi-agent RL with task-level coordination learns significantly faster than flat methods by enabling agents to learn coordination at the level of subtasks rather than primitive actions. [^42^]
Source: "Hierarchical Multi-Agent Reinforcement Learning" (JAAMAS 2006)
URL: https://mohammadghavamzadeh.github.io/PUBLICATIONS/jaamas06.pdf
Date: 2006
Excerpt: "The use of hierarchy speeds up learning in multi-agent domains by making it possible to learn coordination skills at the level of subtasks instead of primitive actions... coordination at high-level provides significant advantage over flat methods by preventing agents from getting confused by low-level details."
Context: Ghavamzadeh et al.'s Cooperative HRL algorithm outperformed industrial heuristics like "first come first serve" in AGV scheduling.
Confidence: high

Claim: ROMA (Role-Oriented Multi-Agent RL) demonstrates that sub-task specialization via emergent role discovery improves team performance, with the performance gap increasing for larger populations. [^43^]
Source: "ROMA: Multi-Agent Reinforcement Learning with Emergent Roles" (ICML 2020)
URL: https://proceedings.mlr.press/v119/wang20f/wang20f.pdf
Date: 2020
Excerpt: "The emergence of role is more likely to improve the labor efficiency in larger populations... such sub-task specialization can indeed improve team performance."
Context: ROMA uses information-theoretic objectives to discover emergent roles, with agents learning to specialize on different sub-tasks automatically.
Confidence: high

Claim: Multi-role RL frameworks that embed role assignment, specialization, and coordination achieve superior win rates, sample efficiency, and scalability across complex multi-agent environments. [^44^]
Source: "Multi-Role RL Frameworks" (Emergent Mind)
URL: https://www.emergentmind.com/topics/multi-role-reinforcement-learning-framework
Date: 2025
Excerpt: "Multi-role reinforcement learning frameworks formalize the division and adaptation of agent behaviors via distinct roles, embedding role assignment, specialization, and coordination mechanisms directly into the learning process."
Context: Modern approaches decouple high-level role planning from low-level execution, with policy parameterization conditioned on learned role embeddings.
Confidence: medium

---

## 6.8 Key Recommendations for AI Utopia Minecraft Village

### Decision: (A) Thin Reactive Controller + Smart Producers vs (B) End-to-End Learned Search

**Recommendation: Hybrid (A+B) - LLM/HTN Planner + Learned Low-Level Policies**

Rationale:
1. **Long-horizon resource tasks** require planning (find → navigate → harvest → return → craft), which is poorly handled by pure end-to-end RL
2. **5 specialized roles** benefit from explicit role decomposition at the high level (HTN/LLM) with shared learned low-level skills
3. **Sample efficiency** is critical - pure RL would require billions of steps; hybrid approaches with behavioral priors (MineRL/VPT) dramatically reduce this
4. **Debuggability** is essential for multi-agent coordination - LLM/HTN provides interpretable plans
5. **Generalization** across world seeds and task variations is best achieved by separating planning (transferable reasoning) from control (learned adaptation)

### Architecture Blueprint

```
Village Goals ("Build a house", "Gather food", "Defend village")
    |
    v
[LLM/HTN Planner per Role]  <- High-level: task decomposition, role coordination
    |
    v
[Skill Library]  <- Mid-level: reusable options (navigate, gather, craft, fight)
    |
    v
[Goal-Conditioned Low-Level Policy]  <- Low-level: visuomotor control at ~20Hz
    |
    v
[Minecraft Environment]  <- Observations, rewards
```

### Why Not Pure Options/Feudal/HAC?

- **Options alone**: Struggle with non-stationarity when multiple agents interact; option discovery is still an open problem
- **Feudal alone**: Manager scalability issues with many goals; latent goal space may not align with semantic task structure
- **HAC/HIRO alone**: Excellent for single-agent continuous control, but don't naturally handle multi-role coordination or discrete high-level reasoning

### Why Not Pure HTN/GOAP?

- **HTN alone**: Requires extensive hand-engineering of decomposition methods; limited adaptation to novel situations
- **GOAP alone**: Performance issues with large action spaces; harder to optimize than HTN
- **Neither learns**: Cannot improve low-level execution through experience; misses opportunity for skill acquisition

### The Consensus Answer

The research consensus strongly favors **hybrid architectures** that combine:
- **Symbolic/explicit planning at the high level** (LLM, HTN, or curriculum)
- **Learned reusable skills at the mid level** (options, skill library)
- **Learned visuomotor control at the low level** (goal-conditioned policies)

This pattern appears in Voyager, SayCan, OpenHA, JueWu-MC, and the HDRL Minecraft agent - all successful recent approaches to long-horizon open-world tasks.

---

## Sources

[^1^] Sutton, R. S., Precup, D., & Singh, S. (1999). "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial Intelligence, 112, 181-211. https://dl.acm.org/doi/10.5555/645848.668804

[^2^] Bacon, P. L., Harb, J., & Precup, D. (2017). "The Option-Critic Architecture." AAAI 2017. https://ojs.aaai.org/index.php/AAAI/article/view/10916/10775

[^3^] Klissarov, M., Bacon, P. L., Harb, J., & Precup, D. (2017). "Learnings Options End-to-End for Continuous Action Tasks." arXiv:1712.00004. https://arxiv.org/pdf/1712.00004

[^4^] Klissarov, M., & Precup, D. (2021). "Flexible Option Learning." arXiv:2112.03097. https://arxiv.org/abs/2112.03097

[^5^] Panwar, A. (2022/2025). "A Minecraft Agent Based on a Hierarchical Deep Reinforcement Learning Model." IJCAI 2022 / IJITEE 2025. https://www.ijcai.org/proceedings/2022/0452.pdf

[^6^] Vezhnevets, A. S., et al. (2017). "FeUdal Networks for Hierarchical Reinforcement Learning." ICML 2017. https://proceedings.mlr.press/v70/vezhnevets17a/vezhnevets17a.pdf

[^7^] Dayan, P., & Hinton, G. E. (1993). "Feudal Reinforcement Learning." NIPS 1993. Referenced in Vezhnevets et al. 2017. https://arxiv.org/abs/1703.01161

[^8^] Vezhnevets et al. (2017). Feudal Networks summary. Blog review at https://minkyuchoi-07.github.io/2023/03/20/feudal/

[^9^] Levy, A., et al. (2017). "Learning Multi-Level Hierarchies with Hindsight." arXiv:1712.00948. https://arxiv.org/abs/1712.00948

[^10^] Levy, A., Platt, R., & Saenko, K. (2017). "Hierarchical Actor-Critic." https://blogs.cuit.columbia.edu/p/hierarchical_actor-critic/

[^11^] Nachum, O., Gu, S., Lee, H., & Levine, S. (2018). "Data-Efficient Hierarchical Reinforcement Learning." NeurIPS 2018. http://papers.neurips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf

[^12^] The Gradient. (2019). "The Promise of Hierarchical Reinforcement Learning." https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/

[^13^] Anonymous. (2025). "A Hierarchical Approach for Long-Horizon Tasks" (RLA). arXiv:2509.05545. https://arxiv.org/html/2509.05545v1

[^14^] Lee, S., et al. (2022). "DHRL: A Graph-Based Approach for Long-Horizon and Complex Hierarchical Reinforcement Learning." NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/58b286aea34a91a3d33e58af0586fa40-Paper-Conference.pdf

[^15^] Anonymous. (2025). "Scalable Option Learning in High-Throughput Environments" (SOL). arXiv:2509.00338. https://arxiv.org/html/2509.00338v3

[^16^] GeeksforGeeks. (2026). "Hierarchical Task Network (HTN) Planning in AI." https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/

[^17^] Verweij, T. J. (2024). "HTN Planning in Decima." Guerrilla Games. https://www.guerrilla-games.com/read/htn-planning-in-decima

[^18^] Trefall, P. (2019). Unity Discussion Forum - Fluid HTN. https://discussions.unity.com/t/released-fluid-hierarchical-task-network-planner-ai-free/740036

[^19^] Game AI Pro. "Exploring HTN Planners through Example." https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter12_Exploring_HTN_Planners_through_Example.pdf

[^20^] Excalibur.js. (2024). "NPC AI planning with GOAP." https://excaliburjs.com/blog/goal-oriented-action-planning/

[^21^] CrashKonijn. (2024). "What is Goap?" https://goap.crashkonijn.com/readme/theory

[^22^] Unity Discussion Forum. (2019). https://discussions.unity.com/t/released-fluid-hierarchical-task-network-planner-ai-free/740036

[^23^] GeeksforGeeks. (2026). HTN Planning in AI. https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-task-network-htn-planning-in-ai/

[^24^] Wang, G., et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv:2305.16291. https://arxiv.org/abs/2305.16291

[^25^] Wang, G., et al. (2023/2024). Voyager. ICLR 2024. https://openreview.net/forum?id=ehfRiF0R3a

[^26^] Research log analysis. (2026). "Voyager: Skill Libraries as the Foundation for Lifelong AI Agent Learning." https://beancount.io/bean-labs/research-logs/2026/05/08/voyager-open-ended-embodied-agent-lifelong-learning

[^27^] Brohan, A., et al. (2022). "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." arXiv:2204.01691. https://say-can.github.io/assets/palm_saycan.pdf

[^28^] SayCan project page. (2022). https://say-can.github.io/

[^29^] Zhong et al. (2025). "OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft." arXiv:2509.13347. https://arxiv.org/html/2509.13347v1

[^30^] RLA paper. (2025). arXiv:2509.05545. https://arxiv.org/html/2509.05545v1

[^31^] Le, H. M., et al. (2018). "Hierarchical Imitation and Reinforcement Learning." ICML 2018. http://proceedings.mlr.press/v80/le18a/le18a.pdf

[^32^] Le, H. M., et al. (2018). Hierarchical Imitation and Reinforcement Learning (extended version). http://users.umiacs.umd.edu/~hal3/docs/daume18ilrl.pdf

[^33^] SOL paper. (2025). arXiv:2509.00338. https://arxiv.org/html/2509.00338v3

[^34^] Emergent Mind. (2025). "Hierarchical Reinforcement Learning." https://www.emergentmind.com/topics/hierarchical-reinforcement-learning

[^35^] The Gradient. (2019). The Promise of HRL. https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/

[^36^] Verweij, T.J. (2024). HTN Planning in Decima. Guerrilla Games. https://www.guerrilla-games.com/read/htn-planning-in-decima

[^37^] Brohan et al. (2022). SayCan. https://arxiv.org/pdf/2204.01691

[^38^] Mankowitz, D. J., et al. (2016). "A Deep Hierarchical Approach to Lifelong Learning in Minecraft." EWRL 2016. https://ewrl.wordpress.com/wp-content/uploads/2016/11/ewrl13-2016-submission_20.pdf

[^39^] Nachum et al. (2018). HIRO. http://papers.neurips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf

[^40^] Wang et al. (2023). Voyager. https://arxiv.org/abs/2305.16291

[^41^] RLA paper. (2025). arXiv:2509.05545. https://arxiv.org/html/2509.05545v1

[^42^] Ghavamzadeh, M., Mahadevan, S., & Makar, R. (2006). "Hierarchical Multi-Agent Reinforcement Learning." JAAMAS. https://mohammadghavamzadeh.github.io/PUBLICATIONS/jaamas06.pdf

[^43^] Wang, J., et al. (2020). "ROMA: Multi-Agent Reinforcement Learning with Emergent Roles." ICML 2020. https://proceedings.mlr.press/v119/wang20f/wang20f.pdf

[^44^] Emergent Mind. (2025). "Multi-Role RL Frameworks." https://www.emergentmind.com/topics/multi-role-reinforcement-learning-framework

### Additional References

- Dietterich, T. G. (2000). "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition." JAIR. https://arxiv.org/pdf/cs/9905014
- Forghieri thesis (2025). "Hierarchical Reinforcement Learning for Large Scale..." https://theses.hal.science/tel-05513147/
- H-TD3 paper (2025). "Hierarchical reinforcement learning method for long-horizon path planning." Aerospace Science and Technology. https://www.sciencedirect.com/science/article/abs/pii/S1270963825001464
- TBC-HRL (2025). "A Bio-Inspired Framework for Stable and Sample-Efficient HRL." PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12650010/
- Multi-Resolution Skills for HRL (NeurIPS 2025). https://openreview.net/forum?id=lnTrBYewkG
- RD-HRL (ICLR 2026). "Generating Reliable Sub-Goals for Long-Horizon Sparse-Reward Tasks." https://iclr.cc/virtual/2026/poster/10011487
- Int-HRL. "Intention-based hierarchical reinforcement learning." https://pmc.ncbi.nlm.nih.gov/articles/PMC12313806/
- JueWu-MC (IJCAI 2022). "Playing Minecraft with Sample-efficient Hierarchical KL." https://www.ijcai.org/proceedings/2022/0452.pdf
- OpenAI. (2019). "Emergent Tool Use From Multi-Agent Autocurricula." https://jxwuyi.weebly.com/uploads/2/5/1/1/25111124/hide-seek.pdf
- Hierarchical RL with Augmented Step-Level Transitions for LLM Agents (2026). https://arxiv.org/html/2604.05808v2
- CoGHP. "Chain-of-Goals Hierarchical Policy for Long-Horizon Offline Goal-Conditioned RL." https://arxiv.org/html/2602.03389v1
- Hierarchical RL-based Mapless Navigation. https://orca.cardiff.ac.uk/id/eprint/159543/
- Offline Goal-Conditioned RL with Latent States as Actions (HIQL, NeurIPS 2023). https://neurips.cc/virtual/2023/poster/71099
- SG-Safe. "Enhancing Safe Exploration Through Subgoal Guidance." https://www.mdpi.com/2227-7080/14/3/146
- LLM-augmented HRL for autonomous driving (2025). https://www.sciencedirect.com/science/article/abs/pii/S0957417425023541
- Hierarchical RL Survey (2022). "Hierarchical Reinforcement Learning: A Survey and Open Research Challenges." https://www.mdpi.com/2504-4990/4/1/9
- Hierarchical MARL with Skill Discovery (2019). https://arxiv.org/abs/1912.03558
- HRL with Advantage-Based Auxiliary Rewards (2019). https://arxiv.org/abs/1910.04450
- Computational evidence for hierarchically structured RL in the brain (PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC7703642/
