## Dim 10: Multi-Agent Information Sharing & CTDE Patterns

---

### 10.1 CTDE Framework Review (MAPPO, MADDPG, QMIX)

#### Finding 1: CTDE Definition and Core Taxonomy

```
Claim: CTDE (Centralized Training with Decentralized Execution) is a dominant paradigm in cooperative multi-agent RL where full global information is exploited during training to facilitate coordination, credit assignment, and stability, while decentralized execution mandates that each agent acts solely on its own local observations or action-observation history. CTDE methods fall into two main classes: value function factorization methods (VDN, QMIX, QPLEX) and centralized critic actor-critic methods (MADDPG, COMA, MAPPO). [^1^]
Source: A First Introduction to Cooperative Multi-Agent Reinforcement Learning
URL: https://www.ccs.neu.edu/home/camato/publications/IntroMARL.pdf
Date: 2024
Excerpt: "This text is an introduction to CTDE MARL... the two main classes of CTDE methods: value function factorization methods and centralized critic actor-critic methods. Value function factorization methods include VDN, QMIX, and QPLEX approaches, while centralized critic methods include MADDPG, COMA, and MAPPO."
Context: Comprehensive survey/tutorial paper on CTDE frameworks by Christopher Amato, Northeastern University.
Confidence: high
```

#### Finding 2: MAPPO - Multi-Agent Proximal Policy Optimization

```
Claim: MAPPO is a scalable on-policy deep RL algorithm for cooperative multi-agent systems that builds on PPO with CTDE. It uses decentralized actor networks (often with parameter sharing) and a centralized critic that receives global state information to provide low-variance advantage estimates. MAPPO has become a principal baseline in cooperative MARL, outperforming many off-policy value-decomposition baselines in domains like SMAC, autonomous traffic, and swarm robotics. [^2^]
Source: MAPPO: Multi-Agent Policy Optimization (emergentmind summary)
URL: https://www.emergentmind.com/topics/multi-agent-proximal-policy-optimization-mappo
Date: 2021/2025
Excerpt: "MAPPO is a scalable, on-policy deep reinforcement learning algorithm for cooperative multi-agent systems that builds upon the Proximal Policy Optimization (PPO) framework with a centralized training/decentralized execution (CTDE) paradigm. In MAPPO, each agent independently optimizes a local policy based on its observation while leveraging shared training signals provided by global, centralized critics."
Context: Summary of Yu et al. (2021) MAPPO paper. MAPPO uses clipped surrogate objective and GAE for advantage estimation.
Confidence: high
```

#### Finding 3: MADDPG - Multi-Agent Deep Deterministic Policy Gradient

```
Claim: MADDPG addresses non-stationarity in multi-agent learning by using centralized critics conditioned on the global state and joint actions of all agents, while each agent maintains a decentralized actor that acts on local observations only. The centralized critic provides low-variance gradients for policy updates during training, and is discarded at execution time. MADDPG enables agents to learn emergent coordination strategies, efficient communication protocols, and cooperative behaviors that non-communicative baselines cannot discover. [^3^]
Source: MADDPG: Multi-Agent Deep Deterministic Policy Gradient (Emergent Mind)
URL: https://www.emergentmind.com/topics/multi-agent-deep-deterministic-policy-gradient-maddpg
Date: 2017/2025
Excerpt: "The fundamental innovation in MADDPG is the adoption of centralized critics and decentralized actors. During training, each agent's critic is conditioned on the global state (or aggregate of all agents' observations) and the joint action vector (all agents' actions), while each actor only observes its own local information."
Context: Summary of Lowe et al. (2017) NeurIPS paper. MADDPG is one of the foundational CTDE algorithms.
Confidence: high
```

#### Finding 4: QMIX - Monotonic Value Function Factorization

```
Claim: QMIX trains decentralized policies in a centralized end-to-end fashion by employing a mixing network that estimates joint action-values as a monotonic combination of per-agent values. The monotonicity constraint (enforced via non-negative weights in the mixing network) guarantees consistency between centralized training and decentralized policies, allowing tractable maximization of the joint action-value during off-policy learning. QMIX significantly outperforms IQL, VDN, and COMA on the StarCraft Multi-Agent Challenge (SMAC). [^4^]
Source: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (ICML 2018)
URL: https://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Date: 2018
Excerpt: "QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. We structurally enforce that the joint-action value is monotonic in the per-agent values, which allows tractable maximisation of the joint action-value in off-policy learning, and guarantees consistency between the centralised and decentralised policies."
Context: Rashid et al., ICML 2018. 4,383 citations. One of the most influential CTDE papers.
Confidence: high
```

#### Finding 5: SMAC - The StarCraft Multi-Agent Challenge as CTDE Benchmark

```
Claim: The StarCraft Multi-Agent Challenge (SMAC) was introduced alongside QMIX as a benchmark for deep multi-agent RL, featuring decentralized micromanagement in StarCraft II where each unit is an independent agent. SMAC has become the standard evaluation environment for CTDE algorithms, testing partial observability, cooperative coordination, and decentralized execution. [^5^]
Source: The StarCraft Multi-Agent Challenge (AAMAS 2019)
URL: https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2019/pdfs/p2186.pdf
Date: 2019
Excerpt: "Partially observable, cooperative, multi-agent learning problems are of particular interest... This gives rise to the paradigm of centralised training with decentralised execution, which has been well-studied in the planning community."
Context: Samvelyan et al., AAMAS 2019. Introduced alongside PyMARL codebase.
Confidence: high
```

---

### 10.2 Shared Map / Blackboard Architectures

#### Finding 6: Shared Map for Multi-Agent Pathfinding with Grid Memory

```
Claim: A shared map architecture for multi-agent cooperative pathfinding uses incremental map updates (delta updates) where each agent transmits only newly observed blocked/free locations to teammates, who merge these into their local copies of the shared map. This avoids full synchronization overhead while ensuring agents have timely access to environmental data. The shared map is combined with per-agent grid memory that accumulates local observations over time. [^6^]
Source: Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps
URL: https://arxiv.org/html/2503.22162v1
Date: 2025
Excerpt: "Through a decentralized approach, the shared map is updated incrementally. Each agent periodically transmits its own local updates to teammates, who then merge these local changes into their versions of the shared map. This maintains a distributed structure without introducing excessive communication overhead."
Context: Uses APPO (Asynchronous PPO) with grid memory for obstacle information. Fuse() operation merges local updates into global map.
Confidence: high
```

#### Finding 7: Blackboard Architecture as Shared Memory for Multi-Agent Systems

```
Claim: The blackboard architecture serves as a central database where agents post intermediate computing results. Autonomous agents continuously monitor this public space for new information and independently decide when to contribute based on their capabilities. This decouples agents from each other (they need not know about other agents' existence), provides transparency through centralized visibility, and scales well because new agents can be added without system redesign. [^7^]
Source: Understanding Shared Memory In Multi-Agent Systems
URL: https://jumpcloud.com/it-index/understanding-shared-memory-in-multi-agent-systems
Date: 2026
Excerpt: "A blackboard architecture serves as a central database where agents post their intermediate computing results. Autonomous agents continuously monitor this public space for new information. They independently decide when to contribute based on their specific capabilities and programmed instructions. This decentralized approach eliminates the need for a rigid central coordinator."
Context: Technical overview of shared memory architectures for multi-agent systems.
Confidence: high
```

#### Finding 8: LLM-Based Blackboard Architecture for Multi-Agent Collaboration

```
Claim: In LbMAS (LLM-based Multi-Agent Systems), the blackboard serves as a shared public space where agents communicate solely through writing and reading from the blackboard - no direct agent-to-agent communication occurs. This removes the need for individual memory modules since all agent messages are stored centrally, reducing prompt lengths and enabling more discussions under token constraints. [^8^]
Source: Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture
URL: https://arxiv.org/html/2507.01701v1
Date: 2025
Excerpt: "In LbMAS agents communicate solely through the blackboard without any direct contact; in other words the blackboard is responsible for all agent communication and agents decide on their own what to write on the blackboard. With the availability of blackboard, we remove the memory module commonly existing in LLM-based agents."
Context: Recent work on blackboard architectures for LLM multi-agent systems. Applicable by analogy to RL agent shared state.
Confidence: medium
```

#### Finding 9: Shared Recurrent Memory Transformer (SRMT) for Multi-Agent Pathfinding

```
Claim: The Shared Recurrent Memory Transformer (SRMT) extends memory transformers to multi-agent settings by pooling and globally broadcasting individual working memories, enabling agents to implicitly exchange information and coordinate actions without explicit communication protocols. SRMT consistently outperforms RL baselines in bottleneck navigation tasks, especially under sparse rewards, and generalizes effectively to corridors 1000+ cells long - far beyond training scenarios. [^9^]
Source: SRMT: Shared Memory for Multi-agent Lifelong Pathfinding
URL: https://arxiv.org/abs/2501.13200
Date: 2025
Excerpt: "We propose the Shared Recurrent Memory Transformer (SRMT) which extends memory transformers to multi-agent settings by pooling and globally broadcasting individual working memories, enabling agents to exchange information implicitly and coordinate their actions... SRMT consistently outperforms a variety of reinforcement learning baselines, especially under sparse rewards, and generalizes effectively to longer corridors than those seen during training."
Context: Sagirova et al., NeurIPS 2024. Inspired by Global Workspace Theory (Baars, 1988).
Confidence: high
```

---

### 10.3 Emergent Communication (CommNet, TarMAC, MAGIC)

#### Finding 10: CommNet - Learning Multi-Agent Communication with Backpropagation

```
Claim: CommNet is a neural network architecture that enables collaborative decision-making among multiple agents by allowing them to communicate via continuous vectors (hidden states). Each agent broadcasts its hidden state as a message; incoming messages from other agents are averaged and used as input for the next layer. The entire system is end-to-end differentiable and can be trained via standard backpropagation combined with RL algorithms. However, this averaging approach may face scalability issues with large numbers of agents. [^10^]
Source: Learning Multiagent Communication with Backpropagation (NeurIPS 2016)
URL: https://arxiv.org/abs/1605.07736
Date: 2016
Excerpt: "We explore a simple neural model, called CommNet, that uses continuous communication for fully cooperative tasks. The model consists of multiple agents and the communication between them is learned alongside their policy... agents are able to learn to communicate amongst themselves, yielding improved performance over non-communicative agents and baselines."
Context: Sukhbaatar, Szlam, and Fergus, NeurIPS 2016. 1,777 citations. Pioneering work in emergent communication.
Confidence: high
```

#### Finding 11: DIAL - Differentiable Inter-Agent Learning (Foerster et al.)

```
Claim: DIAL (Differentiable Inter-Agent Learning) exploits centralized training by allowing real-valued messages to pass between agents during learning, treating communication actions as bottleneck connections between agents. Gradients can be backpropagated through the communication channel, yielding a system that is end-to-end trainable even across agents. During decentralized execution, real-valued messages are discretized to the allowed communication actions. This is the first work to demonstrate end-to-end learning of communication protocols in complex environments with sequences and raw images. [^11^]
Source: Learning to Communicate with Deep Multi-Agent Reinforcement Learning (NeurIPS 2016)
URL: https://arxiv.org/abs/1605.06676
Date: 2016
Excerpt: "DIAL exploits the fact that, during learning, agents can backpropagate error derivatives through (noisy) communication channels. Hence, this approach uses centralised learning but decentralised execution... gradients can be pushed through the communication channel, yielding a system that is end-to-end trainable even across agents."
Context: Foerster, Assael, de Freitas, and Whiteson, NeurIPS 2016. Introduced both RIAL and DIAL.
Confidence: high
```

#### Finding 12: IC3Net - Individualized Controlled Continuous Communication

```
Claim: IC3Net extends CommNet by adding a gating mechanism that controls when each agent communicates (learned as a discrete decision), and uses individualized rewards per agent rather than shared team rewards. This addresses credit assignment issues and scales better than CommNet. IC3Net can be applied to cooperative, semi-cooperative, and competitive settings, unlike CommNet which was restricted to fully cooperative tasks. [^12^]
Source: Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks (ICLR 2019)
URL: https://arxiv.org/abs/1812.09755
Date: 2018/2019
Excerpt: "IC3Net controls continuous communication with a gating mechanism and uses individualized rewards for each agent to gain better performance and scalability while fixing credit assignment issues."
Context: Singh, Jain, and Sukhbaatar, ICLR 2019. 510 citations. Builds on CommNet.
Confidence: high
```

#### Finding 13: TarMAC - Targeted Multi-Agent Communication

```
Claim: TarMAC enables agents to learn what messages to send AND to whom to address them. Each message contains a signature (intended recipient) and a value (actual message). Receiving agents compute attention weights for every incoming message, aggregate them, and use as input for the local actor alongside local observations. TarMAC follows CTDE and is trained with a centralized critic. It can be combined with IC3Net's gating mechanism for additional efficiency. [^13^]
Source: TarMAC: Targeted Multi-Agent Communication (ICML 2019)
URL: https://proceedings.mlr.press/v97/das19a/das19a.pdf
Date: 2019
Excerpt: "TarMAC can be easily combined with IC3Net, thus extending its applicability to mixed and competitive settings."
Context: Das et al., ICML 2019. 706 citations. Key contribution: recipient-targeted messaging via attention.
Confidence: high
```

#### Finding 14: MAGIC - Multi-Agent Graph-Attention Communication and Teaming

```
Claim: MAGIC is a graph-attention communication protocol that simultaneously addresses "when" and "whom" and "how" to communicate. It uses: (1) a Scheduler with graph attention encoder and differentiable attention mechanism to decide when to communicate and whom to address, and (2) a Message Processor using Graph Attention Networks (GATs) with dynamic directed graphs to process messages. MAGIC outperforms baselines across all tested domains, achieving ~10.5% increase in reward in the most challenging domain (Google Research Football), while communicating 27.4% more efficiently. [^14^]
Source: Multi-Agent Graph-Attention Communication and Teaming (AAMAS 2021)
URL: https://dl.acm.org/doi/10.5555/3463952.3464065
Date: 2021
Excerpt: "We propose a novel multi-agent reinforcement learning algorithm, Multi-Agent Graph-attention Communication (MAGIC)... Our method outperforms baselines across all domains, achieving ~10.5% increase in reward in the most challenging domain. We also show MAGIC communicates 27.4% more efficiently on average than baselines."
Context: Niu, Paleja, and Gombolay, AAMAS 2021 (oral). Also ICCV 2021 Mair2 Workshop best paper.
Confidence: high
```

---

### 10.4 Scout-Forager Coordination

#### Finding 15: Biological Inspiration for Scout-Forager Coordination

```
Claim: Multi-agent foraging systems have drawn inspiration from biological behaviors including bee dance communication (sharing direction and distance vectors to food sources) and ant colony pheromone-based foraging. In computational models, scouts typically search for resources and communicate findings to foragers/recruiters via information vectors. Some systems use centralized communication systems where scouts share location matrices with harvesters. [^15^]
Source: Multi-Agent Foraging: state-of-the-art and research challenges
URL: https://link.springer.com/article/10.1186/s40294-016-0041-8
Date: 2017
Excerpt: "Alers et al. (2011) propose a foraging algorithm inspired by the biological bees' dance behavior. Agents share information about previous search experience as information vector (direction and distance toward food source)... Geuther et al. (2012) propose a dual agent Multi-Robot System for solving a foraging objective. They use scouts and harvesters to harvest energy positioned in clustered regions."
Context: Survey of multi-agent foraging approaches. Biological inspirations for scout-forager coordination.
Confidence: high
```

#### Finding 16: Blackboard-Based Collaborative Information Foraging

```
Claim: A blackboard communication model has been successfully applied to multi-agent information foraging, where foraging agents browse an environment and communicate their partial solutions through a shared blackboard. A blackboard-handler agent manages the shared space and sorts solutions by relevance. When an agent's proposed solution is not competitive, it abandons its current path and restarts from a new location defined by the best solutions on the blackboard. [^16^]
Source: A Collaborative Multi-Agent Approach to Web Information Foraging
URL: https://ceur-ws.org/Vol-1911/20.pdf
Date: 2017
Excerpt: "The collaboration feature is ensured through the blackboard communication model, which offers real-time communication and results sharing between the agents... If the proposed solution by the foraging agent is a top solution then the agent continues surfing on the same path, else the agent abandons its current surfing path and starts surfing again from a new Web page defined randomly."
Context: Web information foraging multi-agent system using blackboard architecture.
Confidence: medium
```

#### Finding 17: SCoUT - Scalable Communication for Large Populations

```
Claim: SCoUT (Scalable Communication via Utility-guided Temporal Grouping) addresses communication in large-population MARL by grouping agents via Gumbel-Softmax sampling and using counterfactual communication advantages for credit assignment. Each agent has a three-headed policy: environment action, send decision, and recipient selection. At execution time, all centralized training components are discarded and only the per-agent policy runs, preserving decentralized execution. SCoUT scales to hundreds of agents (100v100 in Battle, 1.6x-12x more agents than prior work). [^17^]
Source: SCoUT: Scalable Communication via Utility-Guided Temporal Grouping in Multi-Agent Reinforcement Learning
URL: https://arxiv.org/abs/2603.04833
Date: 2026
Excerpt: "We introduce SCoUT which addresses both these challenges via temporal and agent abstraction within traditional MARL. During training, SCoUT resamples soft agent groups every K environment steps via Gumbel-Softmax... At execution time, all centralized training components are discarded and only the per-agent policy is run, preserving decentralized execution."
Context: Vora et al., 2026. Scales to hundreds of agents. Highly relevant for large multi-agent villages.
Confidence: high
```

---

### 10.5 Spatial Memory Sharing

#### Finding 18: Global Workspace Theory Applied to Multi-Agent Shared Memory

```
Claim: The Shared Recurrent Memory Transformer (SRMT) is explicitly inspired by Global Workspace Theory (Baars, 1988), which suggests that independent functional modules in the brain cooperate by broadcasting information through a global workspace. In SRMT, agents act as independent modules with shared memory, learning emergent task-dependent communication to improve coordination and avoid deadlocks. The shared memory consists of a globally accessible, ordered sequence of all agents' memory vectors for the current time step. [^18^]
Source: SRMT: Shared Memory for Multi-agent Lifelong Pathfinding
URL: https://arxiv.org/html/2501.13200v1
Date: 2025
Excerpt: "The global workspace theory suggests that in the brain, there are independent functional modules that can cooperate by broadcasting information through a global workspace. Inspired by this theory, we consider the agents in MAPF as independent modules with shared memory and propose a Shared Recurrent Memory Transformer (SRMT) as a mechanism for exchanging information to improve coordination and avoid deadlocks."
Context: NeurIPS 2024 paper. Explicitly connects shared memory architectures to neuroscientific theory.
Confidence: high
```

#### Finding 19: Grid Memory for Incremental Obstacle Information

```
Claim: EPOM introduces a grid memory for managing obstacle information in multi-agent pathfinding. Unlike standard APPO, this grid memory is updated alongside the RNN hidden state and stores incremental obstacle data from the global map. It scales to different map sizes and ensures agents do not rely on outdated obstacle information. When an agent observes changes, it writes these updates into its own grid memory, expanding the grid as the agent moves. [^19^]
Source: Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps
URL: https://arxiv.org/html/2503.22162v1
Date: 2025
Excerpt: "EPOM introduces a grid memory for managing obstacle information. Unlike standard APPO, this grid memory is updated alongside the RNN hidden state and stores incremental obstacle data from the global map. It scales to different map sizes and ensures agents do not rely on outdated obstacle information."
Context: Grid memory is a per-agent spatial memory structure that accumulates global map information incrementally.
Confidence: high
```

#### Finding 20: MAMBA - Model-Based MARL with Communication

```
Claim: MAMBA is a pure MARL approach that uses communication within a Model-Based Reinforcement Learning framework, featuring discrete communication and decentralized execution. A 3-layer transformer serves as the communication block with its outputs used by agents to update their world models and make action predictions. Each agent maintains its own version of the world model, which can be updated through communication. QPLEX similarly provides inter-agent communication through multi-agent Q-learning with centralized end-to-end training. [^20^]
Source: SRMT: Shared Memory for Multi-agent Lifelong Pathfinding (Related Work section)
URL: https://arxiv.org/html/2501.13200v1
Date: 2025
Excerpt: "MAMBA is a pure MARL approach that uses communication and centralized training within a Model-Based Reinforcement Learning framework, featuring discrete communication and decentralized execution. A 3-layer transformer serves as the communication block with its outputs used by the agents to update their world models and make action predictions."
Context: Related work in SRMT paper. MAMBA (Egorov & Shpilman, 2022) uses transformers for inter-agent communication.
Confidence: high
```

---

### 10.6 Simplest Effective Pattern for Scout->Forager Flow

#### Finding 21: Shared Map with Delta Updates as Simplest Pattern

```
Claim: The simplest effective pattern for scout-to-forager information flow is a shared map with delta updates: when an agent discovers new information (blocked/free locations), it shares only the delta (changed cells) rather than the full map. Other agents merge these deltas into their local map copies. This pattern: (1) avoids full synchronization overhead, (2) works with decentralized execution, (3) naturally handles spatial information like resource locations, and (4) can be combined with any CTDE training algorithm. [^21^]
Source: Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps
URL: https://arxiv.org/html/2503.22162v1
Date: 2025
Excerpt: "The shared map is updated incrementally. Each agent periodically transmits its own local updates to teammates, who then merge these local changes into their versions of the shared map. This maintains a distributed structure without introducing excessive communication overhead."
Context: The Fuse() operation merges local updates. Path blockages are shared and all agents adjust their planning.
Confidence: high
```

#### Finding 22: Parameter Sharing with CTDE Enables Experience Sharing

```
Claim: Parameter sharing is typically employed in CTDE, allowing all agents to use the same policy network parameters. This significantly reduces the number of required parameters, speeds up training, and facilitates experience sharing among agents during centralized training, which helps in learning robust and stable policies. However, agents sharing the same parameters tend to learn homogeneous behaviors, which can limit diversity needed for exploration. For scout-forager coordination, this suggests using role-specific parameter sharing (shared within roles, different across roles). [^22^]
Source: Toward Efficient Multi-Agent Exploration with Diversity-Driven Communication
URL: https://openreview.net/pdf/dd89f301bd928d0b44b44bf9d4d9a194fe80f1e4.pdf
Date: 2024
Excerpt: "The parameter sharing technique is typically employed, allowing all agents to use the same policy network parameters to make action decisions. This approach significantly reduces the number of required parameters, thereby lowering computational complexity and speeding up the training process. Additionally, parameter sharing facilitates the experience sharing among agents during centralized training."
Context: Exploration-focused MARL paper. Discusses trade-offs of parameter sharing for diverse agent behaviors.
Confidence: high
```

#### Finding 23: Communication Schedulers Outperform Full Communication

```
Claim: MAGIC's ablation studies demonstrate that even when agents have unlimited vision (can observe the full state), utilizing a complete graph for communication performs much worse than utilizing a scheduler that gives precise and targeted communication. This confirms that deciding "when" and "whom" to communicate is critical for effective scout-forager information flow - simply broadcasting all information is not optimal. [^23^]
Source: Multi-Agent Graph-Attention Communication and Teaming (AAMAS 2021)
URL: https://yaruniu.com/assets/pdf/magic_mair2.pdf
Date: 2021
Excerpt: "It is interesting to note that even with unlimited vision, utilizing a complete graph for communications performs much worse than utilizing a scheduler that gives precise and targeted communication."
Context: Ablations in MAGIC paper. Full communication graph underperforms targeted communication even with perfect observability.
Confidence: high
```

#### Finding 24: AlphaStar's Multi-Agent League Training for Robust Coordination

```
Claim: AlphaStar uses a novel multi-agent learning algorithm with a "league" of continually adapting agents. The league consists of main agents (train against all past players), main exploiters (train against main agents), and league exploiters (train against all past players). New competitors are dynamically added by branching from existing competitors. The final agent is sampled from the Nash distribution of the league. This creates robust policies that perform well against diverse strategies. The league approach is applicable to any multi-agent domain requiring robust coordination. [^24^]
Source: Grandmaster level in StarCraft II using multi-agent reinforcement learning (Nature)
URL: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
Date: 2019
Excerpt: "A continuous league was created, with the agents of the league - competitors - playing games against each other... New competitors were dynamically added to the league, by branching from existing competitors; each agent then learns from games against other competitors."
Context: Vinyals et al., Nature 2019. AlphaStar achieved Grandmaster level in StarCraft II. League training ensures robustness.
Confidence: high
```

---

### Summary of Key Recommendations

Based on the research findings, the following patterns are recommended for implementing scout-forager information sharing in a CTDE multi-agent system:

1. **Architecture Pattern**: Use CTDE (centralized critic, decentralized actors) with MAPPO or QMIX as the base algorithm. The centralized critic can access the shared map during training; agents act only on local observations + shared map during execution.

2. **Shared Information Structure**: Implement a shared spatial map (grid-based) where scouts write discovered resource locations as delta updates. Each cell contains: (a) terrain type, (b) resource presence/type, (c) timestamp of last observation. The map is stored as part of the centralized state during training and replicated locally during execution.

3. **Communication Mechanism**: Use targeted communication (TarMAC-style attention or MAGIC-style scheduler) rather than broadcast. Scouts should learn when to communicate (gating mechanism like IC3Net) and which foragers should receive the information.

4. **Training Strategy**: Use parameter sharing within roles (all scouts share parameters, all foragers share parameters) but separate parameters across roles. This maintains diversity while enabling experience sharing.

5. **Simplest Viable Pattern**: For an initial implementation, the simplest effective pattern is: (a) a shared map updated via delta messages, (b) a CTDE critic that conditions on the full map, (c) decentralized actors that condition on local observation + local map copy, (d) scout agents write resource locations to the shared map; forager agents read from it.

---

### Sources

[^1^]: Christopher Amato, "A First Introduction to Cooperative Multi-Agent Reinforcement Learning," Northeastern University, 2024. https://www.ccs.neu.edu/home/camato/publications/IntroMARL.pdf

[^2^]: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," 2021 / Emergent Mind summary, 2025. https://www.emergentmind.com/topics/multi-agent-proximal-policy-optimization-mappo

[^3^]: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments," NeurIPS 2017. https://www.emergentmind.com/topics/multi-agent-deep-deterministic-policy-gradient-maddpg

[^4^]: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning," ICML 2018. https://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf

[^5^]: Samvelyan et al., "The StarCraft Multi-Agent Challenge," AAMAS 2019. https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2019/pdfs/p2186.pdf

[^6^]: "Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps," arXiv 2025. https://arxiv.org/html/2503.22162v1

[^7^]: "Understanding Shared Memory In Multi-Agent Systems," JumpCloud, 2026. https://jumpcloud.com/it-index/understanding-shared-memory-in-multi-agent-systems

[^8^]: "Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture," arXiv 2025. https://arxiv.org/html/2507.01701v1

[^9^]: Sagirova, Kuratov, and Burtsev, "SRMT: Shared Memory for Multi-agent Lifelong Pathfinding," NeurIPS 2024. https://arxiv.org/abs/2501.13200

[^10^]: Sukhbaatar, Szlam, and Fergus, "Learning Multiagent Communication with Backpropagation," NeurIPS 2016. https://arxiv.org/abs/1605.07736

[^11^]: Foerster, Assael, de Freitas, and Whiteson, "Learning to Communicate with Deep Multi-Agent Reinforcement Learning," NeurIPS 2016. https://arxiv.org/abs/1605.06676

[^12^]: Singh, Jain, and Sukhbaatar, "Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks," ICLR 2019. https://arxiv.org/abs/1812.09755

[^13^]: Das et al., "TarMAC: Targeted Multi-Agent Communication," ICML 2019. https://proceedings.mlr.press/v97/das19a/das19a.pdf

[^14^]: Niu, Paleja, and Gombolay, "Multi-Agent Graph-Attention Communication and Teaming," AAMAS 2021. https://dl.acm.org/doi/10.5555/3463952.3464065

[^15^]: "Multi-Agent Foraging: state-of-the-art and research challenges," Springer 2017. https://link.springer.com/article/10.1186/s40294-016-0041-8

[^16^]: "A Collaborative Multi-Agent Approach to Web Information Foraging," CEUR-WS 2017. https://ceur-ws.org/Vol-1911/20.pdf

[^17^]: Vora et al., "SCoUT: Scalable Communication via Utility-Guided Temporal Grouping in Multi-Agent Reinforcement Learning," arXiv 2026. https://arxiv.org/abs/2603.04833

[^18^]: Sagirova et al., "SRMT: Shared Memory for Multi-agent Lifelong Pathfinding," NeurIPS 2024. https://arxiv.org/html/2501.13200v1

[^19^]: "Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps," arXiv 2025. https://arxiv.org/html/2503.22162v1

[^20^]: Egorov and Shpilman, "MAMBA," 2022 / cited in SRMT related work. https://arxiv.org/html/2501.13200v1

[^21^]: "Cooperative Hybrid Multi-Agent Pathfinding Based on Shared Exploration Maps," arXiv 2025. https://arxiv.org/html/2503.22162v1

[^22^]: "Toward Efficient Multi-Agent Exploration with Diversity-Driven Communication," OpenReview 2024. https://openreview.net/pdf/dd89f301bd928d0b44b44bf9d4d9a194fe80f1e4.pdf

[^23^]: Niu, Paleja, and Gombolay, "MAGIC," AAMAS 2021 / ICCV Mair2 Workshop. https://yaruniu.com/assets/pdf/magic_mair2.pdf

[^24^]: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," Nature 2019. https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
