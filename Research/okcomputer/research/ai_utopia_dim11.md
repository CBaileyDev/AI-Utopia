## Dim 11: Persistent Multi-Agent Worlds

This document synthesizes research findings on persistent multi-agent worlds, their memory architectures, exploration strategies, and role specialization patterns. The research covers Generative Agents (AI Town), AI-Economist, Voyager, and comparable systems, with focus on implications for a persistent multi-agent Minecraft village.

---

### 11.1 Generative Agents / AI Town

#### 11.1.1 Core Architecture

The Generative Agents framework by Park et al. (2023) introduces a novel agent architecture that combines large language models with mechanisms for synthesizing and retrieving relevant information to condition behavior [^1^].

**Claim:** The generative agent architecture comprises three main components: a memory stream (long-term episodic memory), reflection (higher-level inference synthesis), and planning (hierarchical action generation) [^1^].
**Source:** Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023), UIST 2023 / arXiv:2304.03442
**URL:** https://arxiv.org/abs/2304.03442
**Date:** 2023
**Excerpt:** "At the center of our architecture is the memory stream, a database that maintains a comprehensive record of an agent's experience. From the memory stream, records are retrieved as relevant to plan the agent's actions and react appropriately to the environment. Records are recursively synthesized into higher and higher-level reflections that guide behavior."
**Context:** This architecture was instantiated in Smallville, a 2D sprite-based sandbox environment with 25 agents, running on Phaser with a dedicated server managing world state.
**Confidence:** High

**Claim:** Memory retrieval uses a composite scoring function combining recency (exponentially decaying freshness), importance (LLM-rated 1-10), and relevance (cosine similarity of embeddings), all normalized and weighted equally [^2^].
**Source:** Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
**URL:** https://arxiv.org/abs/2304.03442
**Date:** 2023
**Excerpt:** "The score of each memory item depends on three factors: Recency: exponentially decaying freshness score; Importance: rated by LLM on scale 1-10; Relevance: cosine similarity between embeddings of query and memory item."
**Context:** This retrieval mechanism is critical because agents produce large streams of memories that would otherwise exceed prompt capacity.
**Confidence:** High

**Claim:** Reflections are created periodically (2-3 times per simulated day) when combined importance scores exceed a threshold. The LLM generates 3 high-level questions from the 100 most recent memories, then produces 5 insights with supporting evidence citations [^3^].
**Source:** Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
**URL:** https://3dvar.com/Park2023Generative.pdf
**Date:** 2023
**Excerpt:** "Reflections are created periodically, especially when the combined importance scores of the agent's recent events exceed a certain threshold... To create a reflection via the LLM, the 100 most recent memories are taken and sent to the LLM with the prompt: 'Given only the information above, what are the 3 most salient high-level questions we can answer?'"
**Context:** Reflections can recursively build on previous reflections, forming hierarchical memory trees.
**Confidence:** High

#### 11.1.2 Planning Architecture

**Claim:** Planning follows a top-down recursive decomposition: daily plans (5-8 points) are generated first, then refined to hourly resolution, then to 5-15 minute intervals just-in-time [^4^].
**Source:** Inside Smallville: How AI Agents Built a Town (Medium analysis)
**URL:** https://medium.com/@noahgothacked/inside-smallville-how-ai-agents-built-a-town-and-planned-a-party-0dd129b69d10
**Date:** 2024
**Excerpt:** "The approach to generating plans is top-down and recursive. Initially, a broad plan is generated for the day with 5-8 points, which is then refined and detailed. These elements are first detailed at an hourly resolution and later refined to intervals of 5-15 minutes."
**Context:** Plans are stored in the memory stream and retrieved when relevant. Agents can react and update plans based on environmental observations.
**Confidence:** High

**Claim:** The full architecture outperformed all ablation conditions by a standardized effect size of d=8.16 compared to the no-memory baseline, with all pairwise differences significant at p<0.001 [^1^].
**Source:** Generative Agents (Park et al., 2023), UIST 2023
**URL:** https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
**Date:** 2023
**Excerpt:** "Comparing the condition representing prior work to the full architecture produces a standardized effect size of d=8.16, or eight standard deviations. A Kruskal-Wallis test confirms overall statistical significance (H(4)=150.29, p<0.001)."
**Context:** Ablation conditions removed observation, reflection, and planning components progressively. The full architecture scored mu=29.89 vs. mu=21.21 for the fully ablated baseline.
**Confidence:** High

#### 11.1.3 AI Town Implementation

**Claim:** AI Town (a16z-infra) is a MIT-licensed, deployable starter kit implementing generative agents in JavaScript/TypeScript with Convex for real-time state management, vector search, and PixiJS for rendering [^5^].
**Source:** AI Town GitHub / Grokipedia
**URL:** https://github.com/a16z-infra/ai-town
**Date:** 2023
**Excerpt:** "AI Town stands out through its use of JavaScript and TypeScript, providing greater accessibility for web developers. The project relies on Convex for real-time state management, vector search, and transaction handling, paired with PixiJS for client-side rendering."
**Context:** AI Town uses Llama 3 as the default model via Ollama, with support for OpenAI, Together.ai, and other APIs. The project has 9,600+ GitHub stars.
**Confidence:** High

**Claim:** AI Town's agent loop operates on ticks where agents assess game state and schedule actions. Simple rule-based triggers handle immediate decisions (proximity conversations), while complex reasoning is offloaded asynchronously to LLM calls [^5^].
**Source:** AI Town Technical Architecture
**URL:** https://grokipedia.com/page/AI_Town
**Date:** 2026
**Excerpt:** "Decision-making operates within a tick-based agent loop, where agents assess the game state and schedule actions. Simple rule-based triggers handle immediate decisions, such as starting a conversation upon proximity to another agent, while complex reasoning is offloaded asynchronously to LLM calls."
**Context:** The persistent memory system uses vector embeddings stored in a database. After each conversation, the LLM summarizes the exchange and embeds it for storage.
**Confidence:** High

**Claim:** The sandbox environment is represented as a tree data structure with containment relationships (e.g., "stove" child of "kitchen" rendered as "there is a stove in the kitchen"). Agents build individual subgraphs as they navigate [^6^].
**Source:** Generative Agents (Park et al., 2023) - Paper PDF
**URL:** https://3dvar.com/Park2023Generative.pdf
**Date:** 2023
**Excerpt:** "We represent the sandbox environment -- areas and objects -- as a tree data structure, with an edge in the tree indicating a containment relationship in the sandbox world. We convert this tree into natural language to pass to the generative agents."
**Context:** Agents are not omniscient -- their environment tree may get out of date and is updated when they re-enter areas.
**Confidence:** High

---

### 11.2 AI-Economist

#### 11.2.1 Two-Level RL Framework

**Claim:** The AI-Economist uses a two-level deep reinforcement learning framework where (inner loop) self-interested agents learn utility-maximizing behaviors, and (outer loop) a social planner learns tax policies to optimize social welfare [^7^].
**Source:** The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning (Zheng et al., 2022), Science Advances
**URL:** https://www.science.org/doi/10.1126/sciadv.abk2607
**Date:** 2022
**Excerpt:** "In the inner loop, self-interested workers perform labor, receive income, and pay taxes. They learn through trial-and-error how to adapt their behavior to maximize utility. In the outer loop, tax policies are adapted in order to optimize the social objective."
**Context:** This creates a highly non-stationary learning environment where agents must continuously adapt to changing incentives.
**Confidence:** High

**Claim:** The framework improves the equality-productivity trade-off by 16% compared to the prominent Saez tax framework, with even larger gains over the US Federal income tax and free market [^8^].
**Source:** Salesforce Blog - The AI Economist
**URL:** https://www.salesforce.com/blog/the-ai-economist/
**Date:** 2024
**Excerpt:** "Our experiments show the AI Economist can improve the trade-off between equality and productivity by 16%, compared to a prominent tax framework proposed by Emmanuel Saez, with even larger gains over an adaptation of the US Federal income tax and the free market."
**Context:** The learned tax policies feature higher top tax rates and lower rates for middle incomes, and are robust to emergent tax gaming strategies.
**Confidence:** High

#### 11.2.2 Gather-Trade-Build Environment

**Claim:** The Gather-Trade-Build (GTB) economy is a two-dimensional spatial simulation where agents gather resources (stone/wood), trade with each other, and build houses, simulating 10 tax years of 100 days each [^7^].
**Source:** Science Advances (Zheng et al., 2022)
**URL:** https://www.science.org/doi/10.1126/sciadv.abk2607
**Date:** 2022
**Excerpt:** "Gather-Trade-Build features multiple heterogeneous economic agents in a two-dimensional spatial environment. Productivity and income elasticity emerge as the result of the strategic behavior of multiple agents, rather than from statistical assumptions."
**Context:** Agents have 50 unique actions available including movement, gathering, trading (buy/sell), and building. Observations include egocentric spatial windows (11x11), endowments, and skill levels.
**Confidence:** High

#### 11.2.3 Emergent Role Specialization

**Claim:** AI agents in GTB learn emergent specialization as builders or traders depending on their build-skill. High build-skill agents focus on building houses, while lower-skill agents specialize in gathering and trading [^9^].
**Source:** The AI Economist (Zheng et al., 2022)
**URL:** https://qiniu.pattern.swarma.org/attachment/The%20AI%20Economist-%20Taxation%20policy%20design%20via%20two-level%20deep%20multiagent%20reinforcement%20learning.pdf
**Date:** 2022
**Excerpt:** "Agents specialize as builders (blue agent) or gathers (others) depending on their build-skill... The highest build-skill agent chooses to do the most work and earns larger income."
**Context:** This specialization emerges from the shape of agent rewards and economic incentives, not from explicit role assignments. Even with only 4-10 agents, clear division of labor develops.
**Confidence:** High

**Claim:** Agents learn emergent tax-gaming strategies -- high-income agents move labor and income between tax years to shift income to low-rate brackets, a behavior prohibitively complex for theory-driven methods to derive [^7^].
**Source:** Science Advances (Zheng et al., 2022)
**URL:** https://www.science.org/doi/10.1126/sciadv.abk2607
**Date:** 2022
**Excerpt:** "High-income agents learn to avoid taxes by moving labor and thus income between tax years to move more income to low-rate brackets. This can reduce the overall tax paid in comparison to earning a constant amount each year."
**Context:** This demonstrates that RL agents can discover temporal behavioral strategies that analytical approaches cannot capture.
**Confidence:** High

#### 11.2.4 WarpDrive GPU Framework

**Claim:** WarpDrive is an end-to-end GPU multi-agent RL framework that eliminates CPU-GPU data copying and achieves 100x+ speedup over CPU counterparts, running 2.9M environment steps/second with 2000 environments and 1000 agents [^10^].
**Source:** WarpDrive: Fast End-to-End Deep Multi-Agent RL on a GPU (Lan et al., 2022), JMLR
**URL:** https://github.com/salesforce/warp-drive
**Date:** 2022
**Excerpt:** "WarpDrive yields 2.9 million environment steps/second with 2000 environments and 1000 agents (at least 100x faster than a CPU version) in a 2d-Tag simulation... It is extremely efficient as it avoids back-and-forth data copying between the CPU and the GPU."
**Context:** Each CUDA block runs one environment, each thread simulates one agent. Interactions between agents use shared block memory. Supports 1-1024 agents per environment.
**Confidence:** High

#### 11.2.5 Training Strategies

**Claim:** Three learning curricula stabilize two-level RL: (1) annealing utility cost of labor in phase one, (2) annealing maximum marginal tax in phase two, and (3) carefully balanced entropy regularization to promote coadaptation [^7^].
**Source:** Science Advances (Zheng et al., 2022)
**URL:** https://www.science.org/doi/10.1126/sciadv.abk2607
**Date:** 2022
**Excerpt:** "We use three learning curricula and two training phases to stabilize two-level RL... Agents should not face substantial utility costs that discourage exploration early during learning, and the agents and social planner should be encouraged to gradually explore and coadapt."
**Context:** Training uses PPO on mini-batches from 30 parallel environment replicas, with 6000 transitions sampled per iteration for the planner.
**Confidence:** High

---

### 11.3 Memory Architectures for Persistent Agents

#### 11.3.1 CoALA Framework

**Claim:** The CoALA (Cognitive Architectures for Language Agents) framework organizes language agents along three dimensions: information storage (memory modules), action space, and decision-making loop, formalizing four memory types: working, episodic, semantic, and procedural [^11^].
**Source:** Cognitive Architectures for Language Agents (Sumers et al., 2023), arXiv:2309.02427
**URL:** https://ar5iv.labs.arxiv.org/html/2309.02427
**Date:** 2023
**Excerpt:** "Language agents explicitly organize information into multiple memory modules, each containing a different form of information. These include short-term working memory and several long-term memories: episodic, semantic, and procedural."
**Context:** CoALA draws from cognitive science (Soar architecture) and maps existing agents (ReAct, SayCan, Voyager, Generative Agents) into this unified framework.
**Confidence:** High

**Claim:** In CoALA, working memory maintains active information for the current decision cycle; episodic memory stores experience from earlier cycles (history events, trajectories); semantic memory stores world knowledge; procedural memory contains both implicit knowledge (LLM weights) and explicit knowledge (agent code) [^11^].
**Source:** CoALA Framework (Sumers et al., 2023)
**URL:** https://arxiv.org/html/2309.02427v3
**Date:** 2023
**Excerpt:** "Episodic memory stores experience from earlier decision cycles... Semantic memory stores an agent's knowledge about the world and itself... Procedural memory contains two forms: implicit knowledge stored in LLM weights, and explicit knowledge written in the agent's code."
**Context:** This taxonomy provides the theoretical foundation for designing memory systems in LLM-based agents.
**Confidence:** High

#### 11.3.2 MemGPT / Letta Tiered Memory

**Claim:** MemGPT (now Letta) implements a tiered memory architecture modeled on OS memory hierarchy: core memory (always in-context, like RAM), recall memory (searchable conversation history, like disk cache), and archival memory (long-term vector store, like cold storage). Agents actively manage tiers through explicit function calls [^12^].
**Source:** Agent Memory Architectures: Patterns and Trade-offs (Atlan)
**URL:** https://atlan.com/know/agent-memory-architectures/
**Date:** 2026
**Excerpt:** "The canonical MemGPT/Letta implementation uses three tiers: core memory (always in-context, like RAM), recall memory (searchable conversation history, like disk cache), and archival memory (long-term vector store, like cold storage). Agents actively manage their own tier assignments through explicit function calls."
**Context:** Mem0's LOCOMO benchmark shows in-context memory reaches 72.9% accuracy at 17.12s p95, while selective retrieval reaches 66.9% at 1.44s with 90% fewer tokens.
**Confidence:** High

#### 11.3.3 Voyager Skill Library

**Claim:** Voyager uses an ever-growing skill library of executable code indexed by embedding vectors of natural-language descriptions. Successful programs are stored and retrieved via similarity search (top-5 relevant skills) for new tasks [^13^].
**Source:** Voyager: An Open-Ended Embodied Agent with Large Language Models (Wang et al., 2023), ICLR 2024
**URL:** https://arxiv.org/abs/2305.16291
**Date:** 2023
**Excerpt:** "A skill library of executable code for storing and retrieving complex behaviors... Each skill is indexed by the embedding of its description, which can be retrieved in similar situations in the future."
**Context:** Voyager discovered 63 unique items across 160 prompting iterations (3.3x more than prior SOTA), unlocked wooden-tier tech-tree milestones 15.3x faster, and was the only method to reach diamond-tier in Minecraft.
**Confidence:** High

**Claim:** Voyager's skill library is transferable -- giving accumulated skills to AutoGPT improved its zero-shot generalization from 0/3 to 1-2/3 success. Removing the automatic curriculum dropped discovered-item count by 93%; removing self-verification dropped performance by 73% [^14^].
**Source:** Voyager Open-Ended Embodied Agent Analysis
**URL:** https://beancount.io/bean-labs/research-logs/2026/05/08/voyager-open-ended-embodied-agent-lifelong-learning
**Date:** 2026
**Excerpt:** "Removing the automatic curriculum dropped discovered-item count by 93%. Removing self-verification dropped performance by 73%. The skill library matters most in later stages -- early on it helps little; at 80+ iterations, agents without it plateau."
**Context:** This demonstrates the critical importance of all three components (curriculum, skill library, iterative prompting) working together.
**Confidence:** High

#### 11.3.4 Chroma Vector Store for Agent Memory

**Claim:** Chroma is used as an embedded vector database for agent memory, storing embeddings with HNSW for fast approximate nearest neighbor search, with p50 query latency of 20ms on warm cache for 100k vectors [^15^].
**Source:** AI Agent Chroma Storage Guide (Fastio)
**URL:** https://fast.io/resources/ai-agent-chroma-storage/
**Date:** 2026
**Excerpt:** "Chroma indexes documents into vector collections using HNSW for fast approximate nearest neighbors search... Official benchmarks show p50 query latency of 20ms on warm cache for 100k vectors (384 dimensions), with p99 at 57ms."
**Context:** Chroma integrates into 90k+ open-source codebases with 8M monthly downloads. Supports local execution (sub-100ms retrieval) with Python, JS/TS, and Rust clients.
**Confidence:** High

#### 11.3.5 SAMEP Protocol

**Claim:** SAMEP (Secure Agent Memory Exchange Protocol) enables persistent, secure, semantically searchable memory sharing among AI agents with cryptographic access controls, achieving 73% reduction in redundant computations and 89% improvement in context relevance scores [^16^].
**Source:** SAMEP: A Secure Protocol for Persistent Context Sharing Across AI Agents (arXiv:2507.10562)
**URL:** https://arxiv.org/abs/2507.10562
**Date:** 2025
**Excerpt:** "SAMEP implements a distributed memory repository with vector-based semantic search, cryptographic access controls (AES-256-GCM), and standardized APIs compatible with existing agent communication protocols (MCP, A2A)."
**Context:** Addresses three challenges: persistent context preservation, secure multi-agent collaboration with fine-grained access control, and efficient semantic discovery of relevant historical context.
**Confidence:** Medium (recent preprint)

---

### 11.4 Role Specialization Patterns

#### 11.4.1 CrewAI Role-Based Framework

**Claim:** CrewAI uses a role-playing approach where each agent has a role (title/purpose), goal, and backstory. Coordination models include sequential (pipeline) or hierarchical (manager agent delegates to specialists) [^17^].
**Source:** CrewAI vs AutoGen comparison (Pec Collective)
**URL:** https://pecollective.com/blog/ai-agent-frameworks-compared/
**Date:** 2026
**Excerpt:** "CrewAI uses a role-playing approach. Each agent has a role, a goal, and a backstory that shapes its behavior. Agents are assigned tasks and can delegate to each other. The coordination model is either sequential or hierarchical."
**Context:** CrewAI has 47,000+ GitHub stars and 100K+ developers. Memory supports short-term, entity, and vector memory with Chroma/Qdrant backends.
**Confidence:** High

#### 11.4.2 CAMEL Role-Playing Framework

**Claim:** CAMEL (Communicative Agents for "Mind" Exploration) uses inception prompting to guide chat agents toward task completion via role-playing (AI User + AI Assistant), supporting societies of up to 1M agents for studying emergent behaviors [^18^].
**Source:** CAMEL (Li et al., 2023), NeurIPS 2023
**URL:** https://github.com/camel-ai/camel
**Date:** 2023
**Excerpt:** "Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of a society of agents."
**Context:** CAMEL has 13,800+ GitHub stars and 100+ contributors, supporting up to 1M agents, stateful memory, and multiple benchmarks.
**Confidence:** High

#### 11.4.3 AutoGen Conversational Agents

**Claim:** AutoGen v0.4 uses an asynchronous, event-driven architecture (actor model) for multi-agent conversations, with built-in GroupChat patterns (RoundRobin, Selector), tracing, OpenTelemetry, and human-in-the-loop support [^19^].
**Source:** AutoGen vs CrewAI: Multi-Agent Frameworks Compared
**URL:** https://agent.nexus/blog/autogen-vs-crewai
**Date:** 2025
**Excerpt:** "AutoGen v0.4 uses asynchronous, event-driven architecture (actor model) for concurrency and distributed execution. Layered design: Core, AgentChat, and Extensions. Cross-language support, observability, and AutoGen Studio for drag-and-drop UI."
**Context:** Microsoft is merging AutoGen with Semantic Kernel into the Microsoft Agent Framework. AutoGen has ~38,000 GitHub stars.
**Confidence:** High

#### 11.4.4 Emergent Specialization in RL

**Claim:** RODE (Role-Oriented DEcomposition) decomposes joint action spaces into restricted role action spaces by clustering actions based on their effects on the environment and other agents, reducing execution and training complexity [^20^].
**Source:** Imagine, Initialize, and Explore (IIE) paper, arXiv:2402.17978
**URL:** https://arxiv.org/html/2402.17978v2
**Date:** 2024
**Excerpt:** "RODE decomposes joint action spaces into restricted role action spaces by clustering actions based on their effects on the environment and other agents. Low-level role-based policies explore and learn within these restricted action-observation spaces."
**Context:** IIE extends Go-Explore to multi-agent settings by using an imagination model to initialize agents at promising interaction states, outperforming BC, goal-conditioned policies, and generative models.
**Confidence:** High

**Claim:** Multi-agent shepherding with hierarchical DQN and parameter sharing demonstrates >50% reduction in settling time with emergent specialization [^21^].
**Source:** Cooperative RL in Multi-Agent Systems (Emergent Mind)
**URL:** https://www.emergentmind.com/topics/cooperative-reinforcement-learning-rl
**Date:** 2026
**Excerpt:** "Multi-Agent Shepherding: Hierarchical DQN, parameter sharing: >50% reduction in settling time, emergent specialization."
**Context:** This demonstrates that emergent role specialization can arise from shared policy networks with agent-specific observations.
**Confidence:** Medium

#### 11.4.5 Role Assignment Patterns

**Claim:** Role assignment can be static (predefined, like factory robot arms) or dynamic (adapting to conditions via auction-based bidding or contract net protocols). Dynamic approaches use agents bidding on tasks based on cost/capability [^22^].
**Source:** How do multi-agent systems use role assignment? (Milvus)
**URL:** https://milvus.io/ai-quick-reference/how-do-multiagent-systems-use-role-assignment
**Date:** 2026
**Excerpt:** "Role assignment can be static or dynamic. Static roles are predefined. Dynamic role assignment adapts to changing conditions, like a swarm of delivery drones reassigning roles based on real-time wind patterns. Algorithms like auction-based bidding or contract net protocols are often used."
**Context:** Developers must balance flexibility and stability -- overhead from frequent role changes can reduce efficiency.
**Confidence:** High

---

### 11.5 Exploration in Persistent Multi-Agent Worlds

#### 11.5.1 Go-Explore and Variants

**Claim:** Go-Explore decomposes exploration into two phases: returning to a previously promising state (via archive sampling) and exploring from there. This alleviates detachment and derailment problems in intrinsic motivation methods [^23^].
**Source:** Curiosity-driven Exploration in Sparse-reward Multi-agent Environments (arXiv:2302.10825)
**URL:** https://arxiv.org/pdf/2302.10825
**Date:** 2023
**Excerpt:** "Go-Explore starts the exploration from the achieved state which alleviates the problems of detachment and derailment for intrinsic motivation methods. The Go-Explore method has two phases: exploration and robustify."
**Context:** I-Go-Explore extends this to multi-agent settings by adding a post-exploration phase at the end of each episode, improving training efficiency.
**Confidence:** High

#### 11.5.2 Curiosity-Driven Exploration (ICM)

**Claim:** The Intrinsic Curiosity Module (ICM) provides intrinsic rewards based on prediction error (difference between predicted and actual next state), composed of an encoder, forward model, and inverse model [^23^].
**Source:** Curiosity-driven Exploration in Sparse-reward Multi-agent MARL
**URL:** https://arxiv.org/pdf/2302.10825
**Date:** 2023
**Excerpt:** "ICM is composed of three parts: encoder, forward model, and inverse model. The intrinsic reward is the prediction error which is the difference between the predicted next state and the actual next state."
**Context:** In multi-agent settings, each agent has a private intrinsic reward module for decentralized execution. Combined with MADDPG for centralized training.
**Confidence:** High

#### 11.5.3 Frontier-Based Exploration

**Claim:** MemoryMesh uses delta-encoded map sharing with a goal-broadcast coordination protocol for multi-robot search. Goal exclusion zones penalize frontiers near other agents' announced goals, achieving 0.698 coverage on 150x150 maps with only 208KB communication (294x reduction vs. full-map sharing) [^24^].
**Source:** MemoryMesh: Shared Episodic Spatial Memory for Ground-Air Search (OpenReview)
**URL:** https://openreview.net/forum?id=PwiDtJ1Icn
**Date:** 2026
**Excerpt:** "MemoryMesh combines delta-encoded map sharing with a goal-broadcast coordination protocol. Goal exclusion zones penalize frontiers near other agents' announced goals, persistent goal pursuit reduces oscillation, and role-aware scoring biases aerial agents toward open areas."
**Context:** Three coordination mechanisms emerge: goal exclusion zones, persistent goal pursuit, and role-aware scoring. Robust under 20% sensor noise, 70% UAV occlusion, and 40% packet loss.
**Confidence:** Medium (preprint)

#### 11.5.4 Shared Spatial Memory

**Claim:** A predictive coding framework enables multi-agent shared spatial memory through learned emergent symbols. Agents build 2D bird's-eye-view maps via a Grid Cell Network (LSTM path integrator) and Transformer-based visual processing, achieving 44.7% faster task completion with adaptive communication [^25^].
**Source:** Shared Spatial Memory Through Predictive Coding (arXiv:2511.04235)
**URL:** https://arxiv.org/html/2511.04235v2
**Date:** 2025
**Excerpt:** "The multi-agent cooperative navigation task features multiple agents with egocentric vision input exploring a 3D environment to find a hidden target. They coordinate by building and sharing a 2D bird's-eye-view map via learned, emergent symbols."
**Context:** The framework develops emergent social place cells encoding partner agent locations, with communication bandwidth as low as 4-128 bits/step.
**Confidence:** Medium (preprint)

#### 11.5.5 Cooperative Multi-Agent Exploration (CMAE)

**Claim:** CMAE (Cooperative Exploration for Multi-Agent Deep RL) decouples target and exploration policies with restricted space exploration, achieving 47.7% success rate on 3m-sparse (vs. 18-20% for baselines). Without restricted space exploration, success drops to 0.4% [^26^].
**Source:** Cooperative Exploration for Multi-Agent Deep RL (Liu et al., ICML 2021)
**URL:** https://ioujenliu.github.io/papers/cmae_icml21.pdf
**Date:** 2021
**Excerpt:** "Without restricted space exploration, i.e., by directly exploring the full state space, the success rate drops to 0.4%. This demonstrates that restricted space exploration and policy decoupling are essential to CMAE's success."
**Context:** The key insight is that agents should explore in restricted subspaces rather than the full joint action space.
**Confidence:** High

#### 11.5.6 Self-Motivated Multi-Agent Exploration

**Claim:** SMMAE adaptively adjusts exploration probability based on uncertainty in the multi-agent system -- when uncertainty between agents' actions and others' observations is low, agents explore individually; when high, agents explore less and learn cooperation first [^27^].
**Source:** Self-Motivated Multi-Agent Exploration (AAMAS 2023)
**URL:** https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p476.pdf
**Date:** 2023
**Excerpt:** "When the uncertainty between agents' actions and others' observations is limited, the agents should explore individually to jump out of the local optimum. When the uncertainty is high, agents should explore less and learn how to cooperate first before exploring more."
**Context:** Uses mutual information to measure correlation between each agent's action and observations of other agents, with cross-entropy loss as criterion.
**Confidence:** High

#### 11.5.7 Decentralized Frontier Exploration

**Claim:** Decentralized frontier-based strategies achieve 7.8%, 15.3%, and 32.6% less exploration time for 2, 3, and 5 robots respectively compared to coordinated strategies, by having robots choose frontier points that minimize cost functions accounting for other robots' positions [^28^].
**Source:** Decentralized Strategy for Cooperative Multi-Robot Exploration (IFAC 2020)
**URL:** https://ifatwww.et.uni-magdeburg.de/ifac2020/media/pdfs/0748.pdf
**Date:** 2020
**Excerpt:** "Using our decentralized strategy, it took 7.8%, 15.3% and 32.6% less time for two, three and five mobile robots respectively to explore the environment compared to coordinated strategy."
**Context:** Each robot navigates to frontier points that minimize a cost function rather than simply choosing the closest frontier, reducing overlap.
**Confidence:** High

---

### 11.6 Multi-Agent Coordination Architectures

#### 11.6.1 Blackboard Architecture

**Claim:** The blackboard architecture enables agents to share knowledge through a central memory space without direct agent-to-agent communication. Agents independently monitor the blackboard, contribute when they can add value, and selection emerges based on current content [^29^].
**Source:** Blackboard Architecture in Agentic AI (DataFlair)
**URL:** https://data-flair.training/blogs/blackboard-architecture-in-agentic-ai/
**Date:** 2026
**Excerpt:** "Agents don't directly communicate with each other. Instead, they share and receive information from the blackboard. This design shifts decision-making from a single coordinator to a distributed model."
**Context:** Advantages include collaboration, flexibility (new agents added without redesign), decoupling, transparency, and scalability. Challenges include coordination overhead, conflict resolution, and performance issues with many agents.
**Confidence:** High

#### 11.6.2 Coordination Patterns

**Claim:** Four primary architectural patterns for multi-agent coordination exist: (1) hierarchical orchestration (coordinator manages task allocation), (2) peer-to-peer (message passing without central control), (3) blackboard (shared knowledge base), and (4) dual-tier memory (short-term + long-term) [^30^].
**Source:** Multi-agent systems: Why coordinated AI beats going solo (Redis Blog)
**URL:** https://redis.io/blog/multi-agent-systems-coordinated-ai/
**Date:** 2026
**Excerpt:** "Multi-agent systems use four primary architectural patterns: Hierarchical orchestration, Peer-to-peer coordination, Blackboard architecture, and Memory architecture with short-term and long-term tiers."
**Context:** Each pattern has trade-offs: hierarchical systems create coordinator bottlenecks; peer-to-peer adds latency; blackboards need careful state management.
**Confidence:** High

#### 11.6.3 Communication Protocols

**Claim:** Production multi-agent systems typically combine three communication mechanisms: shared memory for fast structured state exchange, message queues for task distribution and fault tolerance, and direct calls for tightly coupled orchestrator-agent relationships [^31^].
**Source:** Agent Coordination: How Multi-Agent AI Systems Work Together (Tacnode)
**URL:** https://tacnode.io/post/multi-agent-coordination
**Date:** 2026
**Excerpt:** "In practice, production multi-agent systems combine all three: shared memory for fast structured state exchange, message queues for task distribution and fault tolerance, direct calls for tightly coupled orchestrator-agent relationships."
**Context:** Shared memory risks race conditions; message passing introduces serialization overhead but is more fault-tolerant; direct calls are natural but harder to scale.
**Confidence:** High

---

### 11.7 Lessons for Minecraft Village Architecture

#### 11.7.1 Memory Architecture Recommendations

Based on the research, a persistent multi-agent Minecraft village should implement a tiered memory architecture:

1. **Working Memory (In-Context):** Current observations, active goals, immediate context for the decision cycle (analogous to CoALA working memory)
2. **Episodic Memory (Chroma Vector Store):** Timestamps, locations, events, conversations with recency/importance/relevance scoring (analogous to Generative Agents' memory stream) [^1^]
3. **Semantic Memory (Vector Store + Knowledge Graph):** Facts about the world, item recipes, location types, agent relationships (analogous to CoALA semantic memory) [^11^]
4. **Procedural Memory (Skill Library):** Executable code/functions for tasks, indexed by embedding (analogous to Voyager's skill library) [^13^]
5. **Reflection Memory:** Higher-level inferences and insights synthesized from experiences (analogous to Generative Agents' reflections) [^1^]

#### 11.7.2 Exploration Strategy Recommendations

1. **Frontier-Based Exploration:** Use frontier detection (boundary between explored/unexplored) with decentralized coordination and goal broadcasting to minimize overlap [^24^][^28^]
2. **Curiosity-Driven Exploration:** Add ICM-based intrinsic rewards for novel block types, biomes, and structures discovered [^23^]
3. **Role-Aware Exploration:** Bias exploration based on agent roles -- builders explore for resources, farmers explore for water/arable land, traders explore for villages [^24^]
4. **Adaptive Communication:** Use information bottleneck methods to only share spatial information when it reduces uncertainty, minimizing bandwidth [^25^]

#### 11.7.3 Role Specialization Recommendations

1. **Pre-defined Initial Roles:** Start with explicit roles (builder, farmer, miner, trader) with associated goals and backstories (CrewAI pattern) [^17^]
2. **Emergent Sub-specialization:** Allow roles to evolve based on skill development and economic incentives (AI-Economist pattern) [^7^]
3. **Dynamic Task Assignment:** Use auction-based or contract-net protocols for dynamic task reassignment when conditions change [^22^]
4. **Shared Economic Memory:** Maintain a village ledger (shared blackboard) of resource stocks, trades, and needs visible to all agents [^29^]

#### 11.7.4 Persistent World Architecture

1. **Tick-Based Simulation:** Run agents on a tick loop with simple rule-based triggers for immediate decisions and async LLM calls for complex reasoning (AI Town pattern) [^5^]
2. **State Persistence:** Use a database (like Convex) with real-time subscriptions, vector search, and transactional guarantees for world state and agent memory [^5^]
3. **Environment Tree:** Represent the Minecraft world as a hierarchical tree (village -> building -> room -> chest -> item) rendered into natural language for LLM context [^6^]
4. **GPU Acceleration:** Consider WarpDrive-style GPU parallelization if running many environment replicas for training [^10^]

#### 11.7.5 Key Open Challenges

1. **Context Window Management:** With many agents and long-running simulations, managing what fits in the LLM context window remains the core challenge [^1^]
2. **Memory Deduplication:** Voyager-style skill libraries need deduplication and conflict resolution when similar memories accumulate [^13^]
3. **Coordination Overhead:** As agent count grows, coordination costs increase -- need efficient blackboard/message-passing hybrid [^30^]
4. **Stability of Two-Level Learning:** The AI-Economist shows that simultaneous learning of agents and environment rules is highly non-stationary and requires careful curriculum design [^7^]
5. **Evaluation Complexity:** Long-running persistent worlds are difficult to evaluate -- need continuous monitoring of emergent metrics (trade volume, building quality, social graph density)

---

### Sources

[^1^]: Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*. https://arxiv.org/abs/2304.03442

[^2^]: Same as [^1^], memory retrieval scoring section.

[^3^]: Same as [^1^], reflection generation section.

[^4^]: Medium analysis of Smallville planning. https://medium.com/@noahgothacked/inside-smallville-how-ai-agents-built-a-town-and-planned-a-party-0dd129b69d10

[^5^]: AI Town Technical Architecture. https://grokipedia.com/page/AI_Town / https://github.com/a16z-infra/ai-town

[^6^]: Park et al. (2023), environment tree representation. https://3dvar.com/Park2023Generative.pdf

[^7^]: Zheng, S., Trott, A., Srinivasa, S., et al. (2022). "The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning." *Science Advances*. https://www.science.org/doi/10.1126/sciadv.abk2607

[^8^]: Salesforce Blog on AI Economist. https://www.salesforce.com/blog/the-ai-economist/

[^9^]: AI Economist GTB emergent specialization visualization. https://qiniu.pattern.swarma.org/attachment/The%20AI%20Economist-%20Taxation%20policy%20design%20via%20two-level%20deep%20multiagent%20reinforcement%20learning.pdf

[^10^]: Lan, T., Srinivasa, S., Wang, H., & Zheng, S. (2022). "WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU." *JMLR*. https://github.com/salesforce/warp-drive

[^11^]: Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2023). "Cognitive Architectures for Language Agents." *arXiv:2309.02427*.

[^12^]: Agent Memory Architectures: Patterns and Trade-offs. https://atlan.com/know/agent-memory-architectures/

[^13^]: Wang, G., Xie, Y., Jiang, Y., et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." *ICLR 2024*. https://arxiv.org/abs/2305.16291

[^14^]: Voyager analysis blog. https://beancount.io/bean-labs/research-logs/2026/05/08/voyager-open-ended-embodied-agent-lifelong-learning

[^15^]: AI Agent Chroma Storage Guide. https://fast.io/resources/ai-agent-chroma-storage/

[^16^]: Masoor, H. et al. (2025). "SAMEP: A Secure Protocol for Persistent Context Sharing Across AI Agents." *arXiv:2507.10562*.

[^17^]: CrewAI Framework Analysis. https://pecollective.com/blog/ai-agent-frameworks-compared/

[^18^]: Li, G., Hammoud, H. A. A. K., Itani, H., et al. (2023). "CAMEL: Communicative Agents for Mind Exploration of Large Scale Language Model Society." *NeurIPS 2023*. https://github.com/camel-ai/camel

[^19^]: AutoGen vs CrewAI Comparison. https://agent.nexus/blog/autogen-vs-crewai

[^20^]: IIE Paper (Imagine, Initialize, Explore). https://arxiv.org/html/2402.17978v2

[^21^]: Cooperative RL in Multi-Agent Systems survey. https://www.emergentmind.com/topics/cooperative-reinforcement-learning-rl

[^22^]: Multi-agent role assignment. https://milvus.io/ai-quick-reference/how-do-multiagent-systems-use-role-assignment

[^23^]: Curiosity-driven Exploration in MARL. https://arxiv.org/pdf/2302.10825

[^24^]: MemoryMesh: Shared Episodic Spatial Memory. https://openreview.net/forum?id=PwiDtJ1Icn

[^25^]: Shared Spatial Memory Through Predictive Coding. https://arxiv.org/html/2511.04235v2

[^26^]: Liu, I., Zhu, H., & Wang, Y. (2021). "Cooperative Exploration for Multi-Agent Deep Reinforcement Learning." *ICML 2021*.

[^27^]: Self-Motivated Multi-Agent Exploration. *AAMAS 2023*. https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p476.pdf

[^28^]: Decentralized Strategy for Cooperative Multi-Robot Exploration. *IFAC 2020*.

[^29^]: Blackboard Architecture in Agentic AI. https://data-flair.training/blogs/blackboard-architecture-in-agentic-ai/

[^30^]: Redis Blog: Multi-agent systems coordination. https://redis.io/blog/multi-agent-systems-coordinated-ai/

[^31^]: Tacnode: Agent Coordination. https://tacnode.io/post/multi-agent-coordination

---

### Search Log

| # | Search Query | Results | Key Findings |
|---|-------------|---------|-------------|
| 1 | Generative Agents Park et al 2023 memory stream reflection planning architecture | 5 results | Memory stream, retrieval scoring, reflection hierarchy, planning decomposition |
| 2 | AI Town generative agents implementation architecture detailed | 6 results | Convex backend, PixiJS rendering, tick-based loop, vector embeddings |
| 3 | AI-Economist Zheng et al multi-agent economic simulation coordination | 4 results | Two-level RL, GTB environment, emergent specialization, tax gaming |
| 4 | Persistent multi-agent RL environments long-term | 1 result | Long-term behavior influencing via limiting policy objectives |
| 5 | LLM-based agents memory architectures MemGPT embodied agents | 2 results | Tiered memory patterns, MemGPT/Letta, CoALA framework |
| 6 | Multi-agent curiosity-driven exploration embodied AI decentralized | 0 results (refined later) | |
| 7 | Shared spatial memory multi-agent navigation coordination map | 3 results | MemoryMesh, predictive coding BEV maps, topological memory |
| 8 | Emergent role specialization multi-agent systems LLM agents | 0 results (refined later) | |
| 9 | Persistent world simulation 24/7 multi-agent architecture | 0 results (refined later) | |
| 10 | Cognitive Architectures for Agents memory episodic semantic procedural | 5 results | CoALA framework, Soar architecture, four memory types, multi-agent core |
| 11 | LLM multi-agent role assignment specialization dynamic division labor | 1 result | Static vs dynamic role assignment, auction-based bidding |
| 12 | Curiosity-driven exploration multi-agent decentralized ICM | 2 results | I-Go-Explore, ICM in MARL with MADDPG/COMA |
| 13 | Persistent agent simulation 24/7 autonomous world Minecraft agents | 0 results | |
| 14 | Voyager Minecraft agent memory skill library lifelong learning | 7 results | Skill library, automatic curriculum, iterative prompting, zero-shot transfer |
| 15 | Multi-agent coordination memory sharing communication protocols | 2 results | Shared memory, message passing, direct communication, SAMEP |
| 16 | Emergent specialization multi-agent deep RL division labor | 1 result | RODE, cooperative MARL benchmarks |
| 17 | Decentralized frontier-based exploration multi-robot coordination | 1 result | Decentralized vs coordinated strategies, coverage ratios |
| 18 | LlamaIndex agent memory vector store ChromaDB persistence | 2 results | Chroma persistence patterns, vector storage |
| 19 | AutoGen CrewAI multi-agent framework role assignment patterns | 8 results | Role-based vs conversational, CrewAI/AutoGen/LangGraph comparison |
| 20 | Go-Explore exploration algorithm multi-agent efficient exploration | 4 results | IIE (Imagine Initialize Explore), Go-Explore variants, MAVEN, EMC |
| 21 | AI Economist Gather-Trade-Build emergent specialization trader builder | 1 result | GTB emergent specialization visualization |
| 22 | Multi-agent frontier exploration coverage path planning decentralized | 3 results | Decentralized coverage, curiosity-based MARL for CPP |
| 23 | Reflexion self-reflective agents memory improvement iterative | 4 results | SAGE framework, episodic memory buffer, iterative refinement |
| 24 | CAMEL multi-agent communication role-playing framework | 5 results | CAMEL inception prompting, role-playing, up to 1M agents |
| 25 | Persistent simulation world agents 24/7 autonomous running | 1 result | Google Agent Runtime, sessions, memory bank, long-running agents |
| 26 | WarpDrive GPU multi-agent RL Salesforce framework | 7 results | 100x speedup, 2.9M steps/sec, end-to-end GPU MARL |
| 27 | AI Town Convex backend state management vector search | 1 result | Convex features, real-time queries, vector search |
| 28 | Humanoid Agents Wang 2023 generative agents memory architecture | 3 results | System 1 thinking, Generative Agents improvements |

Total: 28 independent searches across 15+ distinct query topics, yielding primary sources from arXiv, NeurIPS, ICML, UIST, JMLR, Science Advances, AAMAS, and industry blogs.
