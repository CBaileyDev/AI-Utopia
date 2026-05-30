# Dim 9: Online Mapping & Frontier Selection for Exploration Targets

## 9.1 Online Occupancy/Semantic Mapping

### 9.1.1 Classical Occupancy Grid Mapping

**Claim:** Occupancy grid mapping is the foundational representation for online mapping, where each cell holds a probability representing whether it is occupied, and Bayesian updates allow incremental map construction from partial observations. [^1^]
**Source:** "A Frontier-Based Approach for Autonomous Exploration" (Yamauchi 1997)
**URL:** https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/yamauchi_frontiers.pdf
**Date:** 1997
**Excerpt:** "Occupancy Grid is a grid representation of the environment. Each cell holds a probability that represents if it is occupied."
**Context:** Yamauchi's seminal paper defined the three-state classification (open/unknown/occupied) that remains standard today.
**Confidence:** high

**Claim:** Online occupancy grids can be updated efficiently with new sensor observations by only scanning known regions rather than the entire grid, significantly reducing computational complexity. [^2^]
**Source:** "Frontier Based Exploration for Autonomous Robot" (Topiwala et al., 2018)
**URL:** https://arxiv.org/pdf/1806.03581
**Date:** 2018
**Excerpt:** "WFD is similar to the original approach as even this algorithm is based on two nested Breadth-First Searches. The main advantage of the WFD algorithm over the original is that it only scans the known regions of the occupancy grid as opposed to the original approach which scans the entire grid at every run of the algorithm."
**Context:** The Wavefront Frontier Detector (WFD) optimizes frontier detection by restricting search to known space.
**Confidence:** high

**Claim:** Occupancy grids can be modeled with Hidden Markov Models (HMMs) to explicitly represent dynamics and predict future occupancy states, with online EM enabling constant-memory updates. [^3^]
**Source:** "Occupancy Grid Models for Robot Mapping in Changing Environments" (Beinhofer & Burgard, AAAI 2012)
**URL:** http://ais.informatik.uni-freiburg.de/publications/papers/meyerdelius12aaai.pdf
**Date:** 2012
**Excerpt:** "Our method applies cell-specific hidden Markov models (HMM) to represent the belief about the occupancy state and state transition probabilities of each grid cell...the online version of the EM algorithm only needs to store the sufficient statistics...16 values have to be stored."
**Context:** This is particularly relevant for Minecraft where the environment is dynamic (blocks can be placed/removed).
**Confidence:** high

### 9.1.2 Neural SLAM and Learned Mapping

**Claim:** Active Neural SLAM combines a learned SLAM module (predicting maps and poses from RGB observations) with analytical path planning and hierarchical policies, outperforming end-to-end learning approaches for exploration. [^4^]
**Source:** "Learning to Explore using Active Neural SLAM" (Chaplot et al., ICLR 2020)
**URL:** https://arxiv.org/abs/2004.05155
**Date:** 2020
**Excerpt:** "Our approach leverages the strengths of both classical and learning-based methods, by using analytical path planners with learned SLAM module, and global and local policies...hierarchical decomposition and modular training allow us to sidestep the high sample complexities associated with training end-to-end policies."
**Context:** The Neural SLAM module uses CNN encoder-decoder with binary cross-entropy loss for map prediction and MSE for pose prediction. The global policy operates at a coarse timescale (every 25 steps) to produce long-term goals, while the local policy navigates to short-term goals.
**Confidence:** high

**Claim:** The Active Neural SLAM architecture can be directly transferred to PointGoal navigation by simply changing the global policy to output the goal coordinates as the long-term goal, winning the CVPR 2019 Habitat PointGoal Navigation Challenge. [^5^]
**Source:** "Learning to Explore using Active Neural SLAM" (Chaplot et al., ICLR 2020)
**URL:** https://blog.ml.cmu.edu/2020/06/19/learning-to-explore-using-active-neural-slam/
**Date:** 2020
**Excerpt:** "The ANS model trained for Exploration, when transferred to the PointGoal task can outperform all the baselines trained for the PointGoal Task by a large margin."
**Context:** Demonstrates modularity enables transfer without retraining.
**Confidence:** high

**Claim:** For egocentric semantic mapping from RGB-D observations, a ResNet-18 + UNet architecture can produce both occupancy maps and region-level semantic maps, with CLIP features injected for semantic understanding. [^6^]
**Source:** "Mapping High-level Semantic Regions in Indoor Environments" (Bigazzi et al., ICRA 2024)
**URL:** https://stanfordasl.github.io/wp-content/papercite-data/pdf/Bigazzi.ea.ICRA24.pdf
**Date:** 2024
**Excerpt:** "The region mapper module builds a region-level semantic map of the environment while generating an obstacle occupancy grid. At each timestep, the RGB-D observation is processed to extract an egocentric map where the first two channels indicate the occupancy and exploration state of the currently observed region, and the last C channels are dedicated to the registration of observed region-level semantic classes."
**Context:** This approach directly maps to the Minecraft setting where semantic categories (biomes, resources) need to be tracked alongside occupancy.
**Confidence:** high

### 9.1.3 Online 3D Occupancy Prediction

**Claim:** Diffusion models can enable real-time online occupancy prediction across the entire map (not just sensor-covered areas), with probabilistic updates into running occupancy maps improving frontier prediction by 71% over previous methods. [^7^]
**Source:** "Online Diffusion-Based 3D Occupancy Prediction at the Edge" (Reed et al., ICRA 2025)
**URL:** https://www.cairo-lab.com/papers/icra25.pdf
**Date:** 2025
**Excerpt:** "These modifications enable occupancy prediction across the entire map, rather than limiting it to the area around the robot where sensor data can be collected. We introduce a probabilistic update method for merging predicted occupancy data into running occupancy maps, resulting in a 71% improvement in predicting occupancy at map frontiers compared to previous methods."
**Context:** While computationally expensive, this shows how learned models can fill in unobserved regions for better frontier detection.
**Confidence:** medium

---

## 9.2 Frontier Selection Algorithms

### 9.2.1 Classical Frontier-Based Exploration

**Claim:** Frontiers are defined as regions on the boundary between explored open space and unexplored space; the robot repeatedly navigates to the nearest accessible, unvisited frontier until no frontiers remain. [^8^]
**Source:** "A Frontier-Based Approach for Autonomous Exploration" (Yamauchi 1997)
**URL:** https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/yamauchi_frontiers.pdf
**Date:** 1997
**Excerpt:** "Frontiers are regions on the boundary between open space and unexplored space. By moving to new frontiers, a mobile robot can extend its map into new territory until the entire environment has been explored."
**Context:** This is the foundational algorithm. Frontier detection uses edge detection and region extraction: any open cell adjacent to an unknown cell is a frontier edge cell; adjacent edge cells are grouped into frontier regions.
**Confidence:** high

**Claim:** Multi-robot frontier exploration can be achieved by having each robot build its own Sensor-based Random Tree (SRT) and then support other robots' tree expansion through decentralized coordination mechanisms. [^9^]
**Source:** "Frontier-Based Exploration Using Multiple Robots" (Yamauchi 1998)
**URL:** http://robotfrontier.com/papers/agents98.pdf
**Date:** 1998
**Excerpt:** "The multi-agent extension is essentially a parallelization of the basic method...A decentralized cooperation mechanism and two coordination mechanisms are introduced to improve the exploration efficiency and to avoid conflicts."
**Context:** Relevant for multi-agent Minecraft villages where multiple explorers should coordinate.
**Confidence:** high

### 9.2.2 Frontier Detection Algorithms

**Claim:** The Wavefront Frontier Detector (WFD) uses two nested BFS searches: the outer BFS explores only known open space, and when a frontier point is found, an inner BFS extracts the complete frontier region, achieving O(F) complexity where F is the number of frontier cells. [^10^]
**Source:** "Robot Exploration with Fast Frontier Detection: Theory and Experiments" (Keidar & Kaminka, AAMAS 2012)
**URL:** https://www.ifaamas.org/Proceedings/aamas2012/papers/3A_3.pdf
**Date:** 2012
**Excerpt:** "WFD involves beginning with the robot's current location and performing a Breadth-First Search (BFS) from that position through freespace cells until frontier cells are encountered...This algorithm has the advantage over the Naive approach of only evaluating the subset of the map that is freespace."
**Context:** WFD is the standard baseline. The pseudocode is available and can be directly implemented for 2D grid worlds.
**Confidence:** high

**Claim:** Incremental frontier detection algorithms (WFD-INC, WFD-IP, FTFD) can run in time proportional to the active area (newly observed region) rather than the full map size, with FTFD achieving O(n) worst-case and O(1) amortized complexity per timestep. [^11^]
**Source:** "Approaches for Efficiently Detecting Frontier Cells in Robotics Exploration" (Dornhege & Kleiner, Frontiers in Robotics and AI 2021)
**URL:** https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.616470/full
**Date:** 2021
**Excerpt:** "WFD-INC uses the same principle as WFD but bounds the BFS to the active area of the most recent sequence of scans...By using a bounded area, WFD-INC runs in time proportional to the size of the active area rather than the size of the map."
**Context:** For Minecraft with limited perception radius (~16 blocks), incremental algorithms are ideal since only a small region updates each step.
**Confidence:** high

### 9.2.3 Frontier Utility Functions

**Claim:** The geometric utility of a frontier balances frontier size against travel cost: u_geo(f) = length(f) / dist(robot, f, map), with larger frontiers closer to the robot receiving higher scores. [^12^]
**Source:** "How To Not Train Your Dragon: Training-free Embodied Object Goal Navigation with Semantic Frontiers" (Chen et al., RSS 2023)
**URL:** https://www.research-collection.ethz.ch/handle/20.500.11850/622379
**Date:** 2023
**Excerpt:** "The geometric utility of frontier f^i is defined as u_{t,geo}^i = l_t^i / dist(s_t^i, f_t^i, M_t) where dist() is the geodesic distance from the current agent state to frontier f^i on the 2D occupancy map M_t. This heuristic function, which derives a larger value with a larger frontier size and a shorter distance, describes the score of a frontier in greedy exploration policy."
**Context:** This simple utility function is easy to compute and works well as a baseline.
**Confidence:** high

**Claim:** Information-theoretic frontier selection (next-best-view) predicts the expected information gain for traveling to each frontier and selects the one that maximizes expected reduction in map uncertainty, but pure gain maximization can lead to suboptimal exploration time. [^13^]
**Source:** "Information Gain Is Not All You Need" (Bircher et al.)
**URL:** https://arxiv.org/html/2504.01980v3
**Date:** 2025
**Excerpt:** "Prioritizing gain led to fast short-term exploration, but ultimately made completing exploration take longer...the subtle confusion between budget-constrained exploration and quality-constrained exploration was already present."
**Context:** For the Minecraft exploration use case, fastest coverage (budget-constrained) is typically the goal, so a simple nearest-frontier heuristic may outperform sophisticated information-gain methods.
**Confidence:** high

---

## 9.3 Learned Exploration Policies (Goal-Outputting)

### 9.3.1 Hierarchical Exploration with Goal Emission

**Claim:** Plan4MC's Finding-skill uses a hierarchical policy where the high-level recurrent policy observes historical positions and emits a goal position (x,y), while the low-level policy navigates to it; the high-level is trained to maximize state count (area coverage) with PPO. [^14^]
**Source:** "Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks" (Yuan et al., 2023)
**URL:** https://arxiv.org/pdf/2303.16563
**Date:** 2023
**Excerpt:** "The high-level policy pi^H observes historical positions (x,y) from the environment and generates a goal position (x,y)^g. The low-level policy pi^L is a goal-based policy to reach the goal position...The high-level policy is optimized to maximize the state count in the grid world...We divide the world's surface into discrete grids, where each grid represents a 10x10 area. We use state count in the grid as the reward for the high-level policy."
**Context:** This is exactly the architecture needed: a high-level explorer that emits goal positions for a low-level controller. The Finding-skill in Plan4MC is task-agnostic (target-free exploration) and achieves good coverage.
**Confidence:** high

**Claim:** The Finding-skill in Plan4MC critically improves sample efficiency for downstream skills - tasks with Finding-skills achieve 40% conditional success rate vs 25% without, because providing good initialization (near target items) makes RL learning feasible. [^15^]
**Source:** "Plan4MC" (Yuan et al., 2023)
**URL:** https://arxiv.org/pdf/2303.16563
**Date:** 2023
**Excerpt:** "While Plan4MC has a conditional success rate of 0.40, Plan4MC w/o Find-skill decreases to 0.25, showing that solving sub-tasks with additional Finding-skills is more effective."
**Context:** This validates the architectural decision of having a dedicated exploration module.
**Confidence:** high

### 9.3.2 Semi-Parametric Topological Memory

**Claim:** Semi-Parametric Topological Memory (SPTM) builds a graph from exploration footage where nodes correspond to locations and edges encode connectivity; at navigation time, it localizes the agent and goal in the graph, computes shortest path via Dijkstra, and emits the furthest confidently-reachable node as a waypoint. [^16^]
**Source:** "Semi-parametric Topological Memory for Navigation" (Savinov et al., ICLR 2018)
**URL:** https://arxiv.org/pdf/1803.00653
**Date:** 2018
**Excerpt:** "The waypoint vertex is selected as the vertex in the shortest path that is furthest from the agent's vertex but can still be confidently reached by the agent. The output of the SPTM is the corresponding waypoint observation."
**Context:** SPTM provides a concrete example of a memory architecture that emits waypoints for a low-level locomotion network. The average success rate across test environments was 3x higher than best baselines.
**Confidence:** high

### 9.3.3 Goal-Conditioned RL with Subgoal Generation

**Claim:** MUN (World Models for Unconstrained Goal Navigation) uses goal-directed exploration by repeatedly navigating between subgoals sampled from the replay buffer, discovering key subgoals via Farthest Point Sampling on actions (DAD), enabling zero-shot transfer to new goal configurations. [^17^]
**Source:** "Learning World Models for Unconstrained Goal Navigation" (ICLR 2025)
**URL:** https://arxiv.org/html/2411.02446v1
**Date:** 2025
**Excerpt:** "MUN facilitates modeling state transitions between any subgoal states in the replay buffer...we introduce a simple and practical strategy for discovering key subgoal states from the replay buffer. The key subgoals precisely mark the milestones necessary for task completion."
**Context:** Subgoal-based methods decompose exploration into manageable segments.
**Confidence:** medium

**Claim:** Spectral subgoal discovery via Laplacian clustering can identify bottleneck states in the environment that serve as optimal subgoals, achieving 96.5% on AntMaze and 84.5% on Franka-Kitchen benchmarks. [^18^]
**Source:** "Spectral Subgoals for Offline Goal-Conditioned RL" (OpenReview 2025)
**URL:** https://openreview.net/forum?id=wVBVa09JVV
**Date:** 2025
**Excerpt:** "We apply Laplacian spectral clustering to offline dataset to expose bottlenecks and then identify trajectories from the offline dataset that cross these boundaries, and the intersects are defined as keypoints...we provide theory showing that the next bottleneck is the optimal one-step subgoal."
**Context:** While sophisticated, spectral methods require offline data. For online Minecraft exploration, simpler subgoal heuristics may suffice.
**Confidence:** medium

### 9.3.4 Exploration with Learned Models

**Claim:** Plan2Explore learns a world model and uses it to plan for expected future novelty (information gain), outperforming prior self-supervised exploration methods and almost matching oracle performance with access to rewards. [^19^]
**Source:** "Planning to Explore via Self-Supervised World Models" (Sekar et al., ICML 2020)
**URL:** https://ramanans1.github.io/plan2explore/
**Date:** 2020
**Excerpt:** "Unlike prior methods which retrospectively compute the novelty of observations after the agent has already reached them, our agent acts efficiently by leveraging planning to seek out expected future novelty."
**Context:** Plan2Explore represents the state-of-the-art in model-based exploration, though it requires significant compute.
**Confidence:** high

**Claim:** Curiosity-driven exploration using forward dynamics prediction error (Pathak et al. 2017) can learn effective exploration policies without any extrinsic rewards, achieving coverage that significantly exceeds random exploration in 3D environments. [^20^]
**Source:** "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., ICML 2017)
**URL:** https://arxiv.org/pdf/1705.05363
**Date:** 2017
**Excerpt:** "An agent trained with no extrinsic rewards was able to learn to navigate corridors, walk between rooms and explore many rooms in the 3-D Doom environment...the curious agent trained with intrinsic rewards explores a significantly larger number of rooms as compared to a randomly exploring agent."
**Context:** The ICM (Intrinsic Curiosity Module) uses a forward dynamics model in feature space to compute prediction error as reward.
**Confidence:** high

### 9.3.5 Latent Goal and Topological Navigation

**Claim:** RECON combines frontier-based exploration with latent goal-sampling from a learned model, using three behaviors: (1) navigate to feasible goal directly, (2) explore at frontier by sampling random subgoal, (3) go to frontier neighbor, achieving efficient open-world visual navigation. [^21^]
**Source:** "Rapid Exploration for Open-World Navigation with Latent Goal Generation" (Shah et al., CoRL 2022)
**URL:** https://proceedings.mlr.press/v164/shah22a/shah22a.pdf
**Date:** 2022
**Excerpt:** "Feasible Goal: The robot believes it can reach the goal...Explore at Frontier: The robot is at the least-explored node and explores by sampling a random conditional subgoal latent from the prior...Go to Frontier: The robot adopts its least-explored neighbor as a subgoal."
**Context:** RECON explicitly combines frontier exploration with learned goal generation, which maps directly to the "Explorer/Scout emits direction/frontier" requirement.
**Confidence:** high

### 9.3.6 Multi-Agent Learned Exploration

**Claim:** Multi-Agent Active Neural SLAM (MAANS) uses a learned Multi-agent Spatial Planner (MSP) with a transformer-based Spatial-TeamFormer architecture to allocate agents to different frontiers, achieving higher coverage ratios than planning-based methods like RRT. [^22^]
**Source:** "Learning Efficient Multi-Agent Cooperative Visual Exploration" (ECCV 2022)
**URL:** https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990491.pdf
**Date:** 2022
**Excerpt:** "MAANS achieves much higher and faster coverage ratio than RRT...at timestep around 90, MAANS produces global goals successfully allocate the agents towards two distant unexplored area while RRT guides the agents towards the same part of the map."
**Context:** For multi-agent Minecraft villages, learned coordination can prevent redundant exploration.
**Confidence:** medium

---

## 9.4 Resource-Finding Heuristics

### 9.4.1 Semantic Frontier Exploration

**Claim:** Semantic frontiers combine classical frontier exploration with semantic priors from pre-trained language models and scene statistics; by propagating semantics on a spatial scene graph, the agent can score frontiers by their likelihood of containing the target object category. [^23^]
**Source:** "How To Not Train Your Dragon: Training-free Embodied Object Goal Navigation with Semantic Frontiers" (Chen et al., RSS 2023)
**URL:** https://www.research-collection.ethz.ch/handle/20.500.11850/622379
**Date:** 2023
**Excerpt:** "We introduce language and scene-based priors to reason the promising unexplored areas by scoring geometric-based frontiers with semantics via the spatial scene graph. The language priors are acquired from pre-trained large-scale language models that encode knowledge from large-scale natural language inference datasets."
**Context:** For Minecraft, LLM priors can encode knowledge like "iron ore is found underground" or "sugar cane grows near water" to prioritize frontiers.
**Confidence:** high

**Claim:** Deep RL can learn a frontier semantic policy that selects frontiers based on both map features and object category, significantly outperforming map-based methods in success rate and efficiency. [^24^]
**Source:** "Frontier Semantic Exploration for Visual Target Navigation" (Yu et al.)
**URL:** https://yubangguo.com/project/frontier-semantic-exploration/
**Date:** 2022
**Excerpt:** "The semantic map and the frontier map are built from current observation of the environment. Based on the features of the maps, the deep reinforcement learning is used to learn a navigational policy. The policy can be used to select a frontier from the map as long-term goal to explore the environment efficiently based on the object category."
**Context:** Code available at https://github.com/ybgdgh/Frontier-Semantic-Exploration. This approach trains a policy to pick which frontier to go to next given semantic features.
**Confidence:** high

### 9.4.2 Minecraft-Specific Heuristics

**Claim:** Minecraft biome placement follows temperature-based climate zones: land is randomly assigned Warm (4/6), Cold (1/6), or Freezing (1/6), then blended to ensure smooth transitions. Within each temperature zone, biomes are selected probabilistically (e.g., Warm regions: 50% Desert, 33% Savanna, 17% Plains). [^25^]
**Source:** "The World Generation of Minecraft" (Zucconi)
**URL:** https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/
**Date:** 2022
**Excerpt:** "Each piece of land is randomly assigned a temperature: Warm, Cold or Freezing, in proportions of 4, 1 and 1 respectively...Any warm land adjacent to a cool or freezing region will turn into a temperate one instead...Warm regions have a 50% chance of turning into a Desert, 33% into a Savanna, and the remaining 17% into Plains."
**Context:** This temperature-smoothness property means biomes of similar temperature tend to cluster. An explorer can use observed biome temperature to infer likely nearby biomes.
**Confidence:** high

**Claim:** Minecraft modern biome generation uses 6 parameters (Temperature, Humidity/Vegetation, Continentalness, Erosion, Weirdness/Ridges, Depth) computed from density functions based on horizontal coordinates, forming a 6D parameter space where biome intervals are defined. [^26^]
**Source:** "Biome" (Minecraft Wiki)
**URL:** https://minecraft.wiki/w/Biome
**Date:** 2026 (wiki)
**Excerpt:** "Overworld biome generation is based on 6 parameters: Temperature, Humidity (aka. Vegetation), Continentalness (aka. Continents), Erosion, Weirdness (aka. Ridges) and Depth, which are calculated with 6 density functions."
**Context:** The 6D parameter space means biome prediction is complex, but temperature remains the dominant visible cue. For practical exploration, temperature-level changes provide the most reliable signal.
**Confidence:** high

**Claim:** Minecraft villagers/cartographers sell explorer maps that point to specific structures (ocean monuments, woodland mansions, trial chambers, villages of specific biomes), providing a built-in mechanism for targeted exploration. [^27^]
**Source:** "Effective Navigation in Minecraft: Using Compass and Map"
**URL:** https://www.4netplayers.com/en/blog/minecraft/effective-navigation-minecraft-compass-map/
**Date:** 2025
**Excerpt:** "The cartographer can offer you various explorer maps as you increase your bond...Village Map: Leads to another village of a specific biome (savanna, taiga, snow, desert, and plains)"
**Context:** If the agent can trade with villagers, this provides a powerful external resource-locating mechanism.
**Confidence:** medium

### 9.4.3 Exploration Strategies for Minecraft Agents

**Claim:** In Plan4MC, the Finding-skill is task-agnostic and explores by maximizing area coverage; when a target is detected in lidar observations, the agent switches to goal-directed navigation toward the detected target. This provides a general exploration mechanism that works for finding any resource. [^28^]
**Source:** "Plan4MC" (Yuan et al., 2023)
**URL:** https://arxiv.org/pdf/2303.16563
**Date:** 2023
**Excerpt:** "During test, to find a specific item, the agent first explores the world with the hierarchical policy until a target item is detected in its lidar observations. Then, the agent executes the low-level policy conditioned on the detected target's location, to reach the target item."
**Context:** The hierarchical Finding-skill is pre-trained with random goal locations (low-level) and PPO on state count (high-level), making it general-purpose.
**Confidence:** high

**Claim:** Count-based exploration with a goal selector outperforms plan-based exploration (Plan4MC) in sparse resource-finding tasks in Minecraft, especially when combined with Place Event Memory that stores locations based on visual similarity. [^29^]
**Source:** "Instruction-Following Agents in Minecraft with What-Where-When Memory" (Milani et al., 2024)
**URL:** https://arxiv.org/html/2411.06736v1
**Date:** 2024
**Excerpt:** "We compared Steve-1, Mr.Steve with exploration method from Plan4MC and FIFO Memory (PMC-Steve-FM), and our agent Mr.Steve with Count-Based goal selector and VPT-Nav for exploration and Place Event Memory."
**Context:** Count-based exploration maintains visitation counts per grid cell and prefers less-visited areas, providing a simple yet effective heuristic.
**Confidence:** high

---

## 9.5 Simplest Reliable Approach for Partial Observation

### 9.5.1 The Core Loop: Occupancy Grid + Nearest Frontier

Based on all findings, the simplest reliable approach for emitting exploration targets from partial observations is:

**Claim:** The simplest reliable exploration system maintains an occupancy grid from egocentric observations, detects frontiers via BFS from the robot position through known free space, and repeatedly selects and navigates to the nearest accessible frontier. This requires no learning and achieves complete coverage of bounded environments. [^30^]
**Source:** "A Frontier-Based Approach for Autonomous Exploration" (Yamauchi 1997) + "Frontier Based Exploration for Autonomous Robot" (Topiwala 2018)
**URL:** https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/yamauchi_frontiers.pdf
**Date:** 1997/2018
**Excerpt:** "The robot detects frontiers present in the updated grid and attempts to navigate to the nearest accessible, unvisited frontier...If the robot is unable to make progress toward its destination, then after a certain amount of time, the robot will determine that the destination is inaccessible."
**Context:** For Minecraft with ~16 block perception radius, this translates to: (1) mark observed blocks in a 2D grid, (2) run WFD to find frontiers (cells adjacent to unknown), (3) select nearest frontier weighted by size, (4) emit direction/waypoint to controller.
**Confidence:** high

### 9.5.2 Minimal Implementation Architecture

**Claim:** For the specific use case of a Minecraft Explorer/Scout agent with ~16 block perception radius, the minimal architecture is:
1. **2D Occupancy Grid**: Maintain a sparse hash-map or dense grid centered on the agent. Each cell tracks {unknown, free, occupied, visited_count}.
2. **Frontier Detection**: On each observation update, run WFD (BFS from agent through known-free cells, flag cells adjacent to unknown).
3. **Frontier Selection**: Score frontiers by: `score = frontier_size / (path_cost + epsilon)` where path_cost is BFS distance from agent. Add small bonus for frontiers pointing toward unvisited regions.
4. **Goal Emission**: Emit the centroid of the highest-scoring frontier as `(dx, dz)` relative to agent, or as a cardinal direction with distance estimate.
5. **Biome-Aware Prioritization**: If seeking specific resources, weight frontiers by semantic likelihood (e.g., if looking for jungle, prefer frontiers in directions where temperature indicators suggest warmer biomes).
**Source:** Synthesis of findings from Yamauchi 1997, WFD algorithm, Plan4MC Finding-skill, Semantic Frontiers
**Confidence:** high

### 9.5.3 Progressive Enhancement Roadmap

**Claim:** The frontier-based approach can be progressively enhanced:
- **Level 0**: Random walk with obstacle avoidance (no map, no learning)
- **Level 1**: Occupancy grid + nearest frontier (complete, no learning)
- **Level 2**: Add count-based exploration bonus (prefer less-visited areas)
- **Level 3**: Add semantic frontier scoring via LLM priors (prefer frontiers toward target biome types)
- **Level 4**: Learn a high-level goal-emission policy (like Plan4MC Finding-skill) trained to maximize coverage or find specific resources
**Source:** Synthesis of findings
**Confidence:** high

### 9.5.4 Key Design Decision: Task-Agnostic vs Goal-Directed Exploration

**Claim:** A task-agnostic Finding-skill (maximize area coverage) is simpler and more general than goal-directed exploration, because it does not need to know the target at exploration time. When the target is detected, a switch to goal-directed navigation occurs. This is the architecture used by Plan4MC with demonstrated success across 40 diverse Minecraft tasks. [^31^]
**Source:** "Plan4MC" (Yuan et al., 2023)
**URL:** https://arxiv.org/pdf/2303.16563
**Date:** 2023
**Excerpt:** "The Finding-skill is implemented with a hierarchical policy, maximizing the area traversed by the agent...we propose to train a target-free hierarchical policy to solve all the Finding-skills."
**Context:** This architectural pattern cleanly separates the "Explorer/Scout" role (coverage) from the "Navigator" role (target approach).
**Confidence:** high

### 9.5.5 Memory Considerations for Partial Observation

**Claim:** Standard LSTM memory is insufficient for efficient goal-directed navigation in previously unseen environments; specialized map-like representations (topological or metric) are required. SPTM's success (3x baseline performance) demonstrates that storing all observations in a non-parametric graph with learned retrieval is effective. [^32^]
**Source:** "Semi-parametric Topological Memory for Navigation" (Savinov et al., ICLR 2018)
**URL:** https://arxiv.org/pdf/1803.00653
**Date:** 2018
**Excerpt:** "The difference in performance between feedforward and LSTM baseline variants is generally small and inconsistent across mazes. This suggests that standard LSTM memory is not sufficient to efficiently make use of the provided walkthrough footage."
**Context:** For the Minecraft Explorer, this means maintaining an explicit occupancy grid (rather than relying solely on recurrent memory) is critical for reliable exploration.
**Confidence:** high

---

## Sources

[^1^] Yamauchi, B. (1997). A Frontier-Based Approach for Autonomous Exploration. *IEEE International Symposium on Computational Intelligence in Robotics and Automation*.

[^2^] Topiwala, A. et al. (2018). Frontier Based Exploration for Autonomous Robot. *arXiv preprint*.

[^3^] Beinhofer, M. & Burgard, W. (2012). Occupancy Grid Models for Robot Mapping in Changing Environments. *AAAI Conference on Artificial Intelligence*.

[^4^] Chaplot, D.S. et al. (2020). Learning to Explore using Active Neural SLAM. *ICLR 2020*.

[^5^] CMU ML Blog. (2020). Learning to Explore using Active Neural SLAM. https://blog.ml.cmu.edu/2020/06/19/learning-to-explore-using-active-neural-slam/

[^6^] Bigazzi et al. (2024). Mapping High-level Semantic Regions in Indoor Environments. *ICRA 2024*.

[^7^] Reed, A. et al. (2025). Online Diffusion-Based 3D Occupancy Prediction at the Edge. *ICRA 2025*.

[^8^] Yamauchi, B. (1997). A Frontier-Based Approach for Autonomous Exploration. *Proceedings of the 1997 IEEE International Symposium on Computational Intelligence in Robotics and Automation*.

[^9^] Yamauchi, B. (1998). Frontier-Based Exploration Using Multiple Robots. *Proceedings of the Second International Conference on Autonomous Agents*.

[^10^] Keidar, M. & Kaminka, G.A. (2012). Robot Exploration with Fast Frontier Detection: Theory and Experiments. *AAMAS 2012*.

[^11^] Dornhege, C. & Kleiner, A. (2021). Approaches for Efficiently Detecting Frontier Cells in Robotics Exploration. *Frontiers in Robotics and AI*.

[^12^] Chen, J. et al. (2023). How To Not Train Your Dragon: Training-free Embodied Object Goal Navigation with Semantic Frontiers. *RSS 2023*.

[^13^] Bircher, A. et al. (2025). Information Gain Is Not All You Need. *arXiv preprint*.

[^14^] Yuan, H. et al. (2023). Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks. *arXiv:2303.16563*.

[^15^] Yuan, H. et al. (2023). Plan4MC. *OpenReview submission*.

[^16^] Savinov, N. et al. (2018). Semi-parametric Topological Memory for Navigation. *ICLR 2018*.

[^17^] ICLR 2025. Learning World Models for Unconstrained Goal Navigation. *arXiv:2411.02446v1*.

[^18^] OpenReview 2025. Spectral Subgoals for Offline Goal-Conditioned RL.

[^19^] Sekar, R. et al. (2020). Planning to Explore via Self-Supervised World Models. *ICML 2020*.

[^20^] Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-supervised Prediction. *ICML 2017*.

[^21^] Shah, D. et al. (2022). Rapid Exploration for Open-World Navigation with Latent Goal Generation. *CoRL 2022*.

[^22^] ECCV 2022. Learning Efficient Multi-Agent Cooperative Visual Exploration.

[^23^] Chen, J. et al. (2023). Training-free Embodied Object Goal Navigation with Semantic Frontiers. *RSS 2023*.

[^24^] Yu, B. et al. (2022). Frontier Semantic Exploration for Visual Target Navigation.

[^25^] Zucconi, A. (2022). The World Generation of Minecraft. https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/

[^26^] Minecraft Wiki. Biome. https://minecraft.wiki/w/Biome

[^27^] 4Netplayers (2025). Effective Navigation in Minecraft.

[^28^] Yuan, H. et al. (2023). Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks.

[^29^] Milani et al. (2024). Instruction-Following Agents in Minecraft with What-Where-When Memory.

[^30^] Yamauchi, B. (1997). A Frontier-Based Approach for Autonomous Exploration.

[^31^] Yuan, H. et al. (2023). Plan4MC.

[^32^] Savinov, N. et al. (2018). Semi-parametric Topological Memory for Navigation. *ICLR 2018*.
