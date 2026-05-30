# Dim 8: Minecraft RL Benchmarks & Learning Agents

> Research Date: 2025-07-25
> Scope: MineDojo/MineCLIP, DreamerV3, MineRL BASALT, VPT, low-level RL policies, exploration strategies
> Context: Multi-agent Minecraft Java 1.21.1 village project using Ray RLlib PPO + LSTM

---

## 8.1 MineDojo & MineCLIP

### Overview

MineDojo is a comprehensive benchmark and framework for building open-ended embodied agents in Minecraft, developed by Fan et al. (2022) at NVIDIA. It provides a unified simulator API, thousands of benchmarking tasks, and an internet-scale knowledge base (YouTube videos, Wiki pages, Reddit threads) for training and evaluating agents.

**Claim:** MineDojo defines thousands of benchmarking tasks across two categories: Programmatic tasks (automatically evaluable) and Creative tasks (requiring learned evaluation metrics) [^1^].
**Source:** MINEDOJO: Building Open-Ended Embodied Agents with Internet-Scale Knowledge (NeurIPS 2022 Datasets and Benchmarks Track)
**URL:** https://arxiv.org/abs/2206.08853
**Date:** 2022
**Excerpt:** "MINEDOJO offers a set of simulator APIs help researchers develop generally capable, open-ended agents in Minecraft... We define thousands of benchmarking tasks, which are divided into two categories: 1) Programmatic tasks that can be automatically assessed based on the ground-truth simulator states; and 2) Creative tasks that do not have well-defined or easily-automated success criteria."
**Context:** MineDojo builds upon MineRL codebase with unified observation/action spaces, support for all three Minecraft worlds (Overworld, Nether, End), and convenient APIs for configuring initial conditions.
**Confidence:** high

### MineCLIP Architecture

MineCLIP is a contrastive video-language model trained to correlate video snippets with natural language descriptions. It serves as a learned reward function eliminating the need for hand-engineered dense rewards.

**Claim:** MineCLIP uses separate text and video encoders with a temporal aggregator, trained via InfoNCE objective on 640K video-text pairs from YouTube [^2^].
**Source:** MINEDOJO paper (NeurIPS 2022)
**URL:** https://arxiv.org/pdf/2206.08853
**Date:** 2022
**Excerpt:** "MINECLIP is composed of a separate text encoder that embeds a language goal and a video encoder that embeds a moving window of 16 consecutive frames with 160 x 256 resolution. Our neural architecture has a similar design as CLIP4Clip, where the text encoder uses OpenAI CLIP's pretrained text encoder, and the video encoder is factorized into a frame-wise image encoder and a temporal aggregator."
**Context:** Two variants exist: MineCLIP[avg] (average pooling, fast but temporally agnostic) and MineCLIP[attn] (transformer layers, slower but better reward signal).
**Confidence:** high

**Claim:** MineCLIP achieves strong agreement with human judgment on Creative tasks, with F1 scores of 98.7-100% on task success classification [^3^].
**Source:** MINEDOJO paper (NeurIPS 2022)
**URL:** https://arxiv.org/pdf/2206.08853
**Date:** 2022
**Excerpt:** "Table 2: MINECLIP agrees well with the ground-truth human judgment on the Creative tasks we consider. Numbers are F1 scores between MINECLIP's binary classification of tasks success and human labels."
**Context:** Creative tasks include "Find Nether Portal", "Find Ocean", "Dig Hole", and "Lay Carpet" -- tasks without easily automated success criteria.
**Confidence:** high

### MineCLIP for RL Training

**Claim:** MineCLIP agents are trained with PPO + Self-Imitation Learning (SIL), using MineCLIP as the sole reward source for Creative tasks and combined with binary success rewards for Programmatic tasks [^4^].
**Source:** MINEDOJO paper (NeurIPS 2022)
**URL:** https://proceedings.neurips.cc/paper_files/paper/2022/file/74a67268c5cc5910f64938cac4526a90-Paper-Datasets_and_Benchmarks.pdf
**Date:** 2022
**Excerpt:** "Policy networks of all methods share the same architecture and are trained by PPO+Self-Imitation... Ours(Attn): our agent trained with the MINECLIP[attn] reward model. For Programmatic tasks, we also add the final success condition as a binary reward. For Creative tasks, MINECLIP is the only source of reward."
**Context:** The agent (MineAgent) takes raw pixels as input and predicts discrete control actions. MineCLIP eliminates manual reward engineering.
**Confidence:** high

**Claim:** MineCLIP agents demonstrate strong zero-shot visual generalization to unseen terrains, weathers, and lighting conditions, with relative performance drops of only 0.8-32.9% compared to 67.6-77% for naive CLIP approaches [^5^].
**Source:** MINEDOJO paper (NeurIPS 2022)
**URL:** https://arxiv.org/pdf/2206.08853
**Date:** 2022
**Excerpt:** "Table 3: MINECLIP agents have stronger zero-shot visual generalization ability to unseen terrains, weathers, and lighting. Numbers outside parentheses are percentage success rates averaged over 3 seeds, while those inside parentheses are relative performance changes."
**Context:** This shows MineCLIP's domain adaptation capability from noisy YouTube videos to clean simulator frames.
**Confidence:** high

### Exploration in MineDojo

**Claim:** MineDojo uses MineCLIP as a dense reward signal that guides exploration toward language-specified goals without hand-engineered shaping rewards [^6^].
**Source:** MINEDOJO paper (NeurIPS 2022)
**URL:** https://arxiv.org/pdf/2206.08853
**Date:** 2022
**Excerpt:** "Agents developed in popular RL benchmarks often rely on meticulously crafted dense and task-specific reward functions to guide random explorations. However, these rewards are hard or even infeasible to define for our diverse and open-ended tasks in MINEDOJO."
**Context:** The exploration is effectively goal-conditioned: the agent explores toward behaviors that increase MineCLIP similarity between current trajectory and language goal. There is no explicit exploration bonus; exploration emerges from pursuing the MineCLIP reward in a sparse environment.
**Confidence:** high

### CLIP4MC: RL-Friendly Vision-Language Model

**Claim:** CLIP4MC improves upon MineCLIP by incorporating task completion degree into the training objective, making it more "RL-friendly" for distinguishing state importance during exploration [^7^].
**Source:** Reinforcement Learning Friendly Vision-Language Model for Minecraft (ECCV 2024)
**URL:** https://arxiv.org/abs/2303.10571
**Date:** 2023
**Excerpt:** "Simply utilizing the similarity between the video snippet and the language prompt is not RL-friendly since standard VLMs may only capture the similarity at a coarse level. To achieve RL-friendliness, we incorporate the task completion degree into the VLM training objective, as this information can assist agents in distinguishing the importance between different states."
**Context:** CLIP4MC provides denser, more informative reward signals that better guide exploration during RL training.
**Confidence:** high

---

## 8.2 DreamerV3 on Minecraft

### Overview

DreamerV3 (Hafner et al., 2023) is a general model-based RL algorithm that learns a world model and improves behavior by imagining future scenarios. It is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula.

**Claim:** DreamerV3 is the first algorithm to collect diamonds in Minecraft from scratch without human data or manually crafted curricula, requiring 20+ minutes of farsighted planning with sparse rewards [^8^].
**Source:** Mastering Diverse Domains through World Models (ICLR 2023)
**URL:** https://arxiv.org/abs/2301.04104
**Date:** 2023
**Excerpt:** "Applied out of the box, Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula. This achievement has been posed as a significant challenge in artificial intelligence that requires exploring farsighted strategies from pixels and sparse rewards in an open world."
**Context:** The achievement was made with a SINGLE set of hyperparameters used across all 150+ tasks in 8 different benchmarks.
**Confidence:** high

### DreamerV3 Exploration Mechanism

**Claim:** DreamerV3 uses percentile-based return normalization (5th to 95th percentile range) instead of standard deviation normalization, which is critical for exploration under sparse rewards [^9^].
**Source:** Mastering Diverse Domains through World Models (ICLR 2023)
**URL:** https://ar5iv.labs.arxiv.org/html/2301.04104
**Date:** 2023
**Excerpt:** "When rewards are sparse, the return standard deviation is generally small and this approach would amplify the noise contained in near-zero returns, resulting in an overly deterministic policy that fails to explore. Therefore, we propose to scale down large returns without scaling up small returns... we divide returns by their scale S... S = Per(R_t^lambda, 95) - Per(R_t^lambda, 5)"
**Context:** This normalization technique, combined with a fixed entropy regularizer (eta = 3e-4), enables the same exploration strength across both dense and sparse reward environments without domain-specific tuning.
**Confidence:** high

**Claim:** DreamerV3's exploration is driven by entropy regularization combined with robust world model predictions -- the agent explores by imagining future scenarios in its learned world model rather than explicit exploration bonuses [^10^].
**Source:** Mastering Diverse Domains through World Models (ICLR 2023)
**URL:** https://danijar.com/project/dreamerv3/
**Date:** 2023
**Excerpt:** "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios. Robustness techniques based on normalization, balancing and transformations enable stable learning across domains."
**Context:** The key robustness techniques include: symlog transformations for value/reward prediction, KL balancing with free bits for world model regularization, percentile-based return normalization for actor learning, and unimix categorical smoothing. These enable stable exploration from pixels.
**Confidence:** high

### Diamond Environment Standardized Benchmark

**Claim:** The Diamond Environment used by DreamerV3 features 12 milestones leading up to collecting a diamond, with sparse rewards and a 25-action discrete action space [^11^].
**Source:** Diamond Env GitHub repository (danijar/diamond_env)
**URL:** https://github.com/danijar/diamond_env
**Date:** 2023
**Excerpt:** "In the Diamond Env, the agent plays Minecraft to accomplish 12 milestones leading up to collecting a diamond just from sparse rewards, which poses an exploration challenge. Moreover, each episode plays out in a unique randomly generated 3D world, requiring agents to generalize."
**Context:** The environment uses 64x64x3 image observations, 391-dimensional inventory vectors, and a flat categorical action space with 25 actions including crafting and equipment actions.
**Confidence:** high

### How DreamerV3 Handles Exploration

**Claim:** DreamerV3 learns all policy and critic updates purely from imagined rollouts generated by its world model -- it does not need additional environment interactions for policy improvement, making exploration highly sample-efficient [^12^].
**Source:** Emergent Mind - DreamerV3 analysis
**URL:** https://www.emergentmind.com/topics/dreamerv3
**Date:** 2026 (analysis of 2023 paper)
**Excerpt:** "The central innovation in DreamerV3 is policy improvement solely 'in imagination' -- all actor and critic updates proceed using imagined rollouts generated by the world model, with no further real-environment interaction."
**Context:** The agent learns from both real and imagined experience. The world model (RSSM) predicts future states, rewards, and continuation flags. The actor learns to maximize predicted returns from imagined trajectories, with entropy regularization ensuring exploration.
**Confidence:** high

---

## 8.3 MineRL BASALT Winners

### Competition Overview

The MineRL BASALT (Benchmark for Agents that Solve Almost-Lifelike Tasks) competition at NeurIPS 2022 focused on fine-tuning foundation models from human feedback for tasks with hard-to-specify reward functions. Tasks included FindCave, MakeWaterfall, AnimalPen, and BuildVillageHouse.

**Claim:** The top 3 teams in BASALT 2022 were GoUp (1st), UniTeam (2nd), and voggite (3rd). No submission matched human performance [^13^].
**Source:** A Retrospective of the MineRL BASALT 2022 Competition (PMLR 2023)
**URL:** https://proceedings.mlr.press/v220/milani23a/milani23a.pdf
**Date:** 2023
**Excerpt:** "The top three teams were GoUp, UniTeam, and voggite. GoUp achieved higher performance on all tasks but FindCave... None of the submissions matched human performance and thus no submission reached the 100k USD milestone award."
**Context:** Evaluation used human judges via Amazon Mechanical Turk with pairwise comparisons and TrueSkill scoring.
**Confidence:** high

### 1st Place: GoUp (Hybrid ML + Scripting Approach)

**Claim:** GoUp's winning solution divided each task into ML-solvable and script-solvable parts, using fine-tuned VPT for movement, YOLOv5 for target detection, MobileNet for placement detection, and a finite state machine for execution flow [^14^].
**Source:** A Retrospective of the MineRL BASALT 2022 Competition
**URL:** https://proceedings.mlr.press/v220/milani23a/milani23a.pdf
**Date:** 2023
**Excerpt:** "The team found that all of the four tasks consist of the same flow. The agent walks around, searches for a target (e.g., a cave), then solves the task. They identified the targets in each task by training several classifiers and object detection models... Taking the AnimalPen task as an example, the solution contains a fine-tuned VPT model for moving the agent, a fine-tuned YOLOv5 detector for detecting animals, a fine-tuned MobileNet detector for identifying fence placement location, and a finite state machine that controls the executing flow."
**Context:** GoUp's approach explicitly scripts the high-level task decomposition (walk -> search -> detect -> execute) while using ML for perception and low-level control. The exploration (walking/searching) was primarily driven by the VPT foundation model's learned behavior priors.
**Confidence:** high

### 2nd Place: UniTeam (Search-Based Behavioral Cloning)

**Claim:** UniTeam proposed searching the VPT latent space for the most similar past expert situations and copying the corresponding actions, achieving strong performance through simplicity [^15^].
**Source:** A Retrospective of the MineRL BASALT 2022 Competition
**URL:** https://proceedings.mlr.press/v220/milani23a/milani23a.pdf
**Date:** 2023
**Excerpt:** "UniTeam proposed a search-based behavioral cloning approach, which aims to reproduce an expert's behavior by copying relevant actions from relevant situations in the demonstration dataset. They defined a situation as a subset of consecutive frames and actions within a recorded trajectory, which they encoded using the VPT network."
**Context:** This approach performs nearest-neighbor search in VPT latent space to find reference situations. When the L1 distance between current and reference trajectories exceeds a threshold, a new search is performed. The "exploration" is essentially retrieval of expert exploration patterns.
**Confidence:** high

### 3rd Place: voggite (Improved BC with Action Reweighting)

**Claim:** voggite focused on improving the behavioral cloning baseline with action reweighting and trigger-action heuristics for state transitions [^16^].
**Source:** A Retrospective of the MineRL BASALT 2022 Competition
**URL:** https://proceedings.mlr.press/v220/milani23a/milani23a.pdf
**Date:** 2023
**Excerpt:** "They observed that certain actions serve as signals that trigger higher-level changes in state. For example, the use of a bucket to pour water to complete the waterfall task signals that the agent should begin to climb down the mountain to take a scenic picture. They manually encoded these trigger actions along with the change in action distribution."
**Context:** voggite's approach uses manually encoded trigger actions for high-level state transitions, with plans to incorporate this into a hierarchical RL framework like option-critic.
**Confidence:** high

### Other Notable Approaches

**Claim:** KAIROS proposed Preference-Based IQ-Learning (PIQL), combining IQ-Learn imitation learning with reward models trained from pairwise human preferences [^17^].
**Source:** A Retrospective of the MineRL BASALT 2022 Competition
**URL:** https://proceedings.mlr.press/v220/milani23a/milani23a.pdf
**Date:** 2023
**Excerpt:** "KAIROS proposed Preference-Based IQ-Learning (PIQL), a novel algorithm that extends IQ-Learn to additionally leverage the VPT model and online pairwise preferences over trajectories. PIQL uses pairwise preferences over videos to learn a reward function via the Bradley-Terry model through a reward head attached to the VPT network."
**Context:** PIQL combines imitation learning via IQ-Learn with RL that maximizes reward learned from human preferences, including regularization terms for BC loss and KL divergence between successive policies.
**Confidence:** high

---

## 8.4 VPT & Video PreTraining

### VPT Overview

Video PreTraining (VPT), developed by OpenAI (Baker et al., 2022), uses semi-supervised imitation learning to train Minecraft agents from internet-scale unlabeled video data.

**Claim:** VPT uses an Inverse Dynamics Model (IDM) trained on small labeled contractor data to pseudo-label 70,000 hours of unlabeled internet videos, which then trains a foundation model via behavioral cloning [^18^].
**Source:** Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos (NeurIPS 2022)
**URL:** https://proceedings.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf
**Date:** 2022
**Excerpt:** "The main contributions of this work are (1) we are the first to show promising results applying semi-supervised imitation learning to extremely large, noisy, and freely available video datasets for sequential decision domains, (2) we show that such pretraining plus fine-tuning enables agents to solve tasks that were otherwise impossible to learn."
**Context:** The IDM predicts actions from past and future frames. The VPT model (foundation policy) predicts actions from past frames alone. Architecture: ResNet visual encoder + Transformer-XL for temporal modeling.
**Confidence:** high

### VPT for Exploration

**Claim:** VPT foundation model has nontrivial zero-shot performance, accomplishing tasks impossible to learn with RL alone, such as crafting planks and crafting tables. RL from scratch fails almost completely [^19^].
**Source:** VPT paper (NeurIPS 2022)
**URL:** https://ar5iv.labs.arxiv.org/html/2206.11795
**Date:** 2022
**Excerpt:** "Training from a randomly initialized policy fails to achieve almost any reward, underscoring how hard an exploration challenge the diamond pickaxe task is for RL in the native human action space. The model never learns to reliably collect logs. RL fine-tuning from the VPT foundation model does substantially better."
**Context:** This demonstrates that exploration in Minecraft from pixels is extremely difficult for pure RL. VPT's behavioral priors from human data provide the essential exploration foundation.
**Confidence:** high

**Claim:** RL fine-tuning from VPT produces the most dramatic improvements: agents can craft diamond tools, an unprecedented result. Fine-tuning from an early-game model achieves diamond pickaxe crafting in 2.5% of 10-minute episodes [^20^].
**Source:** VPT paper (NeurIPS 2022)
**URL:** https://proceedings.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf
**Date:** 2022
**Excerpt:** "RL fine-tuning from the early-game model learns to obtain (at human-level) all items in the sequence towards a diamond pickaxe and crafts a diamond pickaxe in 2.5% of episodes."
**Context:** The progression: VPT foundation model (zero-shot) -> BC fine-tuning on smaller datasets (stone tools) -> RL fine-tuning with KL loss (diamond tools). Each stage builds on the previous.
**Confidence:** high

### STEVE-1: Text-Conditioned VPT

**Claim:** STEVE-1 extends VPT to follow text instructions by conditioning on MineCLIP latent goals, achieving 12/13 early-game tasks and training for only $60 [^21^].
**Source:** STEVE-1: A Generative Model for Text-to-Behavior in Minecraft (NeurIPS 2023)
**URL:** https://arxiv.org/abs/2306.00937
**Date:** 2023
**Excerpt:** "STEVE-1 is trained in two steps: adapting the pretrained VPT model to follow commands in MineCLIP's latent space, then training a prior to predict latent codes from text. This allows us to finetune VPT through self-supervised behavioral cloning and hindsight relabeling."
**Context:** STEVE-1 uses a CVAE prior to map text embeddings to MineCLIP visual latents, which condition the VPT policy. Classifier-free guidance (scale ~6-7) significantly boosts performance.
**Confidence:** high

---

## 8.5 Low-Level RL Policies for Minecraft

### Hierarchical Deep RL Architecture

**Claim:** A typical hierarchical architecture for Minecraft uses three levels: High-Level Planner (option-critic over subtasks, operating every 5-20 seconds), Mid-Level Skill Controllers (goal-conditioned subtask policies), and Low-Level Visuomotor Policy (pixel-to-action controller at ~20Hz, initialized from BC then fine-tuned with RL) [^22^].
**Source:** A Minecraft Agent Based on a Hierarchical Deep Reinforcement Learning Framework (IJITEE)
**URL:** https://www.ijitee.org/wp-content/uploads/papers/v14i11/K115414111025.pdf
**Date:** 2025
**Excerpt:** "Our agent uses three levels: i) High-Level Planner - a policy over options operating every 5-20 seconds. Output: subtask token; ii) Mid-Level Skill Controllers - reusable subtask policies that achieve semantic goals; iii) Low-Level Visuomotor Policy - a pixel-to-action controller that executes precise keyboard/mouse control. LLP is initialized via behavior cloning from MineRL/VPT trajectories and then fine-tuned with RL."
**Context:** This architecture is representative of the consensus approach: high-level planning (learned or scripted), mid-level skill selection, and low-level visuomotor control with LSTM/Transformer for temporal processing.
**Confidence:** high

### JueWu-MC: MineRL 2021 Champion

**Claim:** JueWu-MC (Tencent AI Lab) won the NeurIPS MineRL 2021 research competition with a hierarchical RL approach featuring action-aware representation learning, discriminator-based self-imitation learning for exploration, and ensemble behavior cloning [^23^].
**Source:** JueWu-MC: Playing Minecraft with Sample-efficient Hierarchical Reinforcement Learning (IJCAI 2022)
**URL:** https://arxiv.org/abs/2112.04907
**Date:** 2021
**Excerpt:** "We propose JueWu-MC, a sample-efficient hierarchical RL approach equipped with representation learning and imitation learning to deal with perception and exploration. Our approach includes two levels of hierarchy, where the high-level controller learns a policy to control over options and the low-level workers learn to solve each sub-task."
**Context:** JueWu-MC achieved the highest performance score ever in MineRL competition. The discriminator-based self-imitation learning explicitly addresses exploration by learning from past good behaviors.
**Confidence:** high

### Plan4MC: Skill RL and Planning

**Claim:** Plan4MC converts multi-task learning into learning basic skills and planning over them, using a novel Finding-skill with hierarchical exploration policy that maximizes area traversed [^24^].
**Source:** Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks (ICLR 2025)
**URL:** https://arxiv.org/abs/2303.16563
**Date:** 2023
**Excerpt:** "We propose to learn a Finding-skill that performs exploration to find items in the world and provides better initialization for all other skills. The Finding-skill is implemented with a hierarchical policy, maximizing the area traversed by the agent. The high-level policy observes historical locations and outputs a goal location. It drives the low-level policy to reach the goal location."
**Context:** The high-level exploration policy uses PPO with state count rewards in a grid world. The low-level policy is goal-conditioned and pre-trained with DQN on random goals. This explicitly separates exploration as a learnable skill.
**Confidence:** high

### MrSteve: Memory-Augmented Low-Level Controller

**Claim:** MrSteve introduces Place Event Memory (PEM) for low-level controllers, enabling agents to alternate between exploration and task-solving based on recalled events [^25^].
**Source:** MrSteve: Instruction-Following Agents in Minecraft with What-Where-When Memory (ICLR 2025)
**URL:** https://arxiv.org/abs/2411.06736
**Date:** 2024
**Excerpt:** "We introduce MrSteve, a novel low-level controller equipped with Place Event Memory (PEM), a form of episodic memory that captures what, where, and when information from episodes. This directly addresses the main limitation of the popular low-level controller, Steve-1."
**Context:** MrSteve has two exploration modes: task-free (randomly selects least-visited locations) and task-conditioned (selects least-visited locations relevant to the task using MineCLIP alignment scores). It uses VPT-Nav for navigation and hierarchical count-based exploration.
**Confidence:** high

### End-to-End Visuomotor Training with PPO+GRU

**Claim:** End-to-end training with PPO uses recurrent layers (GRU/LSTM) to maintain spatial memory, with visual observations encoded and concatenated with goals and previous actions for actor-critic training [^26^].
**Source:** Building Autonomous Agents with Hybrid Navigation Policies (PhD Thesis, HAL)
**URL:** https://hal.science/tel-04571955/file/these.pdf
**Date:** 2023
**Excerpt:** "During training with PPO, at each time step t, the agent encodes visual observation o_t, an ego-centric view m_t of a global map M_t, the current goal g_t and the previous action a_t and concatenate the embeddings to be passed to a recurrent layer Gated Recurrent Unit (GRU). The output recurrent state is then passed to actor-critic architecture for action prediction and value prediction."
**Context:** The recurrent layer maintains an internal spatial representation during navigation. The agent is trained with 12 parallel vectorized environments.
**Confidence:** high

---

## 8.6 Exploration Strategy Analysis

### How Different Systems Handle Exploration

#### 1. Goal-Conditioned Exploration (MineCLIP/MineDojo)
- Exploration is guided by learned vision-language reward (MineCLIP similarity)
- No explicit exploration bonus; exploration emerges from pursuing language goal
- Uses PPO + Self-Imitation Learning for training
- **Learned or scripted:** Learned reward, learned policy

#### 2. World Model Imagination (DreamerV3)
- Exploration via entropy-regularized actor in imagined rollouts
- Percentile-based return normalization enables consistent exploration across sparse/dense rewards
- Learns world model (RSSM) to predict future states and rewards
- **Learned or scripted:** Entirely learned -- world model, policy, and exploration

#### 3. Behavioral Prior + RL Fine-tuning (VPT)
- Foundation model provides human-like behavioral priors from 70K hours of video
- RL fine-tuning with KL loss prevents catastrophic forgetting of exploration skills
- Random exploration from scratch fails completely
- **Learned or scripted:** Learned priors, learned fine-tuning

#### 4. Hybrid ML + Scripting (GoUp/BASALT)
- High-level task flow scripted as finite state machine (walk -> search -> detect -> execute)
- Low-level movement learned via fine-tuned VPT
- Object detection learned via YOLOv5
- **Learned or scripted:** Hybrid -- scripts for task decomposition, ML for perception and control

#### 5. Latent Space Search (UniTeam)
- "Exploration" is retrieval of most similar expert situations from VPT latent space
- No learned exploration policy per se
- **Learned or scripted:** Scripted retrieval over learned representations

#### 6. Hierarchical Skill Learning (Plan4MC/JueWu-MC)
- Explicit Finding-skill trained with count-based intrinsic rewards
- High-level policy selects exploration goals; low-level policy executes navigation
- Discriminator-based self-imitation for efficient exploration
- **Learned or scripted:** Learned hierarchical policies with intrinsic motivation

#### 7. Episodic Memory-Guided (MrSteve)
- Count-based exploration (least-visited places first)
- Task-conditioned exploration selects relevant locations using MineCLIP alignment
- Memory of past events enables recall-based navigation
- **Learned or scripted:** Learned memory and exploration strategies

#### 8. LLM-Driven Curriculum (Voyager)
- Automatic curriculum maximizes exploration by proposing progressively harder tasks
- Skill library stores and retrieves executable code for complex behaviors
- Iterative prompting with environment feedback and self-verification
- **Learned or scripted:** LLM-generated curriculum, code-as-action space

### Key Claim: Exploration is Learned, Not Scripted, in State-of-the-Art Systems

**Claim:** In state-of-the-art Minecraft RL systems, exploration is predominantly learned rather than scripted, with the strongest results coming from systems that combine learned world models, behavioral priors from human data, or intrinsic motivation mechanisms [^27^].
**Source:** Analysis across multiple papers (DreamerV3, VPT, Plan4MC, MrSteve)
**URL:** Multiple sources
**Date:** 2022-2024
**Excerpt:** (Synthesized) "DreamerV3 learns exploration entirely through world model imagination and entropy regularization. VPT relies on learned human behavioral priors. Plan4MC trains a dedicated Finding-skill with count-based rewards. MrSteve uses learned episodic memory to guide exploration. Only hybrid competition approaches (GoUp) script the high-level task flow while learning low-level control."
**Context:** The trend is clearly toward learned exploration, with scripting limited to high-level task decomposition in competition settings where human judgment evaluates final results.
**Confidence:** high

### "Where to Go" -- Map/Memory/Planner vs Control Policy

**Claim:** Across systems, "where to go" is determined by different mechanisms: high-level planners (option-critic, LLM skill graphs, count-based exploration policies) produce subgoals, while low-level control policies execute visuomotor actions to reach them [^28^].
**Source:** Analysis across Plan4MC, MrSteve, JueWu-MC, hierarchical architecture papers
**URL:** Multiple sources
**Date:** 2021-2024
**Excerpt:** (Synthesized) "Plan4MC uses a skill graph for planning and a Finding-skill for exploration. MrSteve uses a Mode Selector to alternate between exploration and execution based on memory contents. JueWu-MC uses a high-level controller over learned options. The consensus pattern is: a high-level module decides WHAT to do and WHERE to go, while a low-level policy decides HOW to move."
**Context:** The separation of concerns is consistent: high-level reasoning (learned or scripted) handles task decomposition and target selection, while low-level visuomotor policies handle pixel-to-action control.
**Confidence:** high

---

## 8.7 Consensus Pattern for "Find Distant Resource"

Based on the research, the consensus pattern for finding distant resources in Minecraft involves:

### Pattern: Hierarchical Goal-Directed Exploration

1. **High-Level Planner/Navigator**: Determines exploration targets
   - Count-based exploration (least-visited areas) [Plan4MC, MrSteve]
   - Task-conditioned selection using MineCLIP alignment [MrSteve]
   - LLM-generated skill sequences [Plan4MC skill graph, Voyager curriculum]
   - Episodic memory recall of previously seen resources [MrSteve PEM]

2. **Mid-Level Navigation Policy**: Guides movement toward targets
   - Goal-conditioned low-level policy trained with DQN/PPO [Plan4MC]
   - VPT-Nav for visual navigation [MrSteve]
   - Fine-tuned VPT foundation model [GoUp, various]

3. **Low-Level Visuomotor Policy**: Executes precise motor control
   - PPO + LSTM/Transformer for pixel-to-action [MineDojo, JueWu-MC]
   - VPT-based behavioral cloning + RL fine-tuning [VPT, STEVE-1]
   - Discrete action space (typically 10-25 actions)

4. **Exploration Strategy**:
   - Intrinsic motivation: count-based state visitation [Plan4MC]
   - Entropy-regularized goal pursuit [DreamerV3]
   - Learned behavioral priors from human data [VPT]
   - Episodic memory for revisiting known locations [MrSteve]

### Implications for Multi-Agent Village Project

For a multi-agent Minecraft Java 1.21.1 village project using Ray RLlib PPO + LSTM:

1. **Exploration must be explicitly addressed** -- pure PPO+LSTM will struggle with sparse resources in a large world
2. **Hierarchical architecture is essential** -- separate "where to go" (high-level planner) from "how to move" (low-level policy)
3. **Count-based exploration provides a strong baseline** -- maintain visitation maps and reward visiting new areas
4. **Episodic memory significantly improves efficiency** -- agents should remember resource locations
5. **Foundation model pretraining helps enormously** -- VPT/DreamerV3 priors reduce exploration burden
6. **MineCLIP can provide goal-conditioned rewards** -- useful for specifying "find X" tasks in natural language

---

## Sources

[^1^] Fan et al., "MINEDOJO: Building Open-Ended Embodied Agents with Internet-Scale Knowledge," NeurIPS 2022 Datasets and Benchmarks Track. https://arxiv.org/abs/2206.08853

[^2^] Fan et al., MineCLIP section, NeurIPS 2022. https://arxiv.org/pdf/2206.08853

[^3^] Fan et al., MineCLIP evaluation section, NeurIPS 2022. https://arxiv.org/pdf/2206.08853

[^4^] Fan et al., MineAgent training section, NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/74a67268c5cc5910f64938cac4526a90-Paper-Datasets_and_Benchmarks.pdf

[^5^] Fan et al., Zero-shot generalization table, NeurIPS 2022. https://arxiv.org/pdf/2206.08853

[^6^] Fan et al., MineDojo motivation and reward design, NeurIPS 2022. https://arxiv.org/pdf/2206.08853

[^7^] Jiang et al., "Reinforcement Learning Friendly Vision-Language Model for Minecraft," ECCV 2024. https://arxiv.org/abs/2303.10571

[^8^] Hafner et al., "Mastering Diverse Domains through World Models," ICLR 2023. https://arxiv.org/abs/2301.04104

[^9^] Hafner et al., Actor learning section, ICLR 2023. https://ar5iv.labs.arxiv.org/html/2301.04104

[^10^] Hafner et al., DreamerV3 project page. https://danijar.com/project/dreamerv3/

[^11^] Hafner, Diamond Environment GitHub. https://github.com/danijar/diamond_env

[^12^] Emergent Mind, DreamerV3 analysis. https://www.emergentmind.com/topics/dreamerv3

[^13^] Milani et al., "A Retrospective of the MineRL BASALT 2022 Competition," PMLR 2023. https://proceedings.mlr.press/v220/milani23a/milani23a.pdf

[^14^] Milani et al., GoUp approach description, PMLR 2023. https://proceedings.mlr.press/v220/milani23a/milani23a.pdf

[^15^] Milani et al., UniTeam approach description, PMLR 2023. https://proceedings.mlr.press/v220/milani23a/milani23a.pdf

[^16^] Milani et al., voggite approach description, PMLR 2023. https://proceedings.mlr.press/v220/milani23a/milani23a.pdf

[^17^] Milani et al., KAIROS approach description, PMLR 2023. https://proceedings.mlr.press/v220/milani23a/milani23a.pdf

[^18^] Baker et al., "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos," NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf

[^19^] Baker et al., VPT exploration results, NeurIPS 2022. https://ar5iv.labs.arxiv.org/html/2206.11795

[^20^] Baker et al., RL fine-tuning results, NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf

[^21^] Lifshitz et al., "STEVE-1: A Generative Model for Text-to-Behavior in Minecraft," NeurIPS 2023. https://arxiv.org/abs/2306.00937

[^22^] "A Minecraft Agent Based on a Hierarchical Deep Reinforcement Learning Framework," IJITEE. https://www.ijitee.org/wp-content/uploads/papers/v14i11/K115414111025.pdf

[^23^] Lin et al., "JueWu-MC: Playing Minecraft with Sample-efficient Hierarchical Reinforcement Learning," IJCAI 2022. https://arxiv.org/abs/2112.04907

[^24^] Cai et al., "Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks," ICLR 2025. https://arxiv.org/abs/2303.16563

[^25^] Park et al., "MrSteve: Instruction-Following Agents in Minecraft with What-Where-When Memory," ICLR 2025. https://arxiv.org/abs/2411.06736

[^26^] "Building Autonomous Agents with Hybrid Navigation Policies," PhD Thesis. https://hal.science/tel-04571955/file/these.pdf

[^27^] Synthesis across: DreamerV3 (Hafner 2023), VPT (Baker 2022), Plan4MC (Cai 2023), MrSteve (Park 2024)

[^28^] Synthesis across: Plan4MC, MrSteve, JueWu-MC, MineDojo, hierarchical architecture papers

### Additional References

- Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," 2023. https://arxiv.org/abs/2305.16291
- MineRL BASALT 2022 Competition Retrospective: https://arxiv.org/abs/2303.13512
- MineRL Diamond 2021 Competition: https://publications.hse.ru/pubs/share/direct/971875070.pdf
- Diamond Environment (standardized benchmark): https://github.com/danijar/diamond_env
- NVIDIA MineDojo Blog: https://developer.nvidia.com/blog/building-generally-capable-ai-agents-with-minedojo/
- OpenAI VPT Blog: https://openai.com/blog/vpt

---

*Research compiled from 15+ independent web searches covering academic papers, competition retrospectives, GitHub repositories, and technical blog posts.*
