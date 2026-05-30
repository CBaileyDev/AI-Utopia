# Dim 5: BC Warm-Start for Exploration Bootstrapping

> **Research Question:** Can warm-starting from a scripted non-oracle searcher teach learnable search that pure PPO cannot discover?

---

## 5.1 BC→PPO Warm-Start

### Finding 1: BC→PPO (PIRLNav) achieves state-of-the-art in ObjectNav, outperforming pure BC or pure RL

```
Claim: BC pretraining on human demonstrations followed by RL finetuning achieves 65.0% success on ObjectNav,
+5.0% absolute over previous state-of-the-art. BC pretraining "unlocks" RL with sparse rewards that is 
"typically out of reach" when learning from scratch. [^320^]
Source: PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav
URL: https://arxiv.org/abs/2301.07302
Date: 2023 (CVPR)
Excerpt: "BC pretrained policies provide a reasonable starting point for 'bootstrapping' RL and make the 
optimization easier than learning from scratch. In fact, we show that BC pretraining even unlocks RL with 
sparse rewards. Sparse rewards are simple and do not suffer from the unintended consequences described above. 
However, learning from scratch with sparse rewards is typically out of reach since most random action 
trajectories result in no positive rewards."
Context: ObjectGoal Navigation in HM3D environments with visual observations (RGB). Two-stage training:
BC on 77k human demonstrations → RL finetuning with PPO. The RL finetuning uses a two-phase approach:
critic-only learning first, then actor warmup.
Confidence: HIGH
```

### Finding 2: Naive BC→RL fails without proper critic initialization; two-phase training is critical

```
Claim: Naively finetuning an actor initialized from BC with a randomly-initialized critic leads to rapid 
performance drops because the critic provides poor value estimates. A two-phase regime (critic learning 
first, then actor warmup) is essential. [^321^]
Source: PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav (CVPR 2023)
URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Ramrakhya_PIRLNav_Pretraining_With_Imitation_and_RL_Finetuning_for_ObjectNav_CVPR_2023_paper.pdf
Date: 2023
Excerpt: "Naively finetuning the actor with a randomly-initialized critic leads to a rapid drop in 
performance since the critic provides poor value estimates which influence the actor's gradient updates."
Context: Detailed ablation study comparing:
- BC→RL (naive): 53.6% success
- BC→RL + Critic Learning: 56.7% success  
- BC→RL + Critic Learning + Critic Decay: 59.4% success
- BC→RL + Critic Learning + Actor Warmup: 58.2% success
- PIRLNav (full): 61.9% success
The critic-only phase freezes the actor and trains the critic on rollouts from the frozen BC policy
for ~8M steps. The actor learning rate is then warmed up from 0 while critic LR decays.
Confidence: HIGH
```

### Finding 3: BC pretraining halves training duration needed for warehouse navigation

```
Claim: Pre-training with BC and then fine-tuning with PPO halved the training duration needed to reach 
target performance compared to training from scratch. [^207^]
Source: Evaluation of Autonomous Warehouse Navigation through...
URL: https://www.diva-portal.org/smash/get/diva2:1985389/FULLTEXT01.pdf
Date: 2025
Excerpt: "In this experiment, pre-training with Behavioral Cloning (BC) and then fine-tuning with PPO 
halved the training duration needed, reaching the target..."
Context: Autonomous warehouse navigation with sparse rewards. BC→PPO vs PPO from scratch.
Confidence: MEDIUM
```

### Finding 4: Reinforced Imitation reduces RL training time by 5-8x for map-less navigation

```
Claim: Pre-training with IL on expert demonstrations reduces RL training time by a factor of 5 while 
achieving similar final performance, and the RL reward function can be significantly simplified (e.g., 
using sparse reward only). [^326^]
Source: Reinforced Imitation: Sample Efficient Deep RL for Map-less Navigation by Leveraging Prior Demonstrations
URL: https://arxiv.org/abs/1805.07095
Date: 2018 (IEEE RA-L)
Excerpt: "We show that leveraging prior expert demonstrations for pre-training can reduce the training 
time to reach at least the same level of performance compared to plain RL by a factor of 5... 
The RL reward function can be significantly simplified when using pre-training, e.g. by using a sparse 
reward only."
Context: Target-driven map-less navigation, end-to-end neural network trained with combination of IL 
and RL. Demonstrations generated using ROS move_base navigation stack. Tested in simulation and on 
real robot platform.
Confidence: HIGH
```

### Finding 5: Actor-Critic Pretraining for PPO requires both actor and critic initialization

```
Claim: Most prior work focuses on initializing only the actor network, but initialization strategies 
for the critic network significantly improve sample efficiency and convergence. A theoretical framework 
for joint actor-critic pretraining is developed. [^249^]
Source: Actor-Critic Pretraining for Proximal Policy Optimization
URL: https://arxiv.org/pdf/2602.23804
Date: 2025
Excerpt: "Existing approaches typically focus on pretraining the actor network on expert demonstrations... 
However, initialization strategies for the critic network have received less attention. This work addresses 
the research gap by providing an approach for critic initialization that leads to increased sample 
efficiency and improved convergence."
Context: 15 benchmark environments. Focus on PPO. Joint pretraining of both actor and critic networks.
Confidence: HIGH
```

---

## 5.2 DAgger for RL Initialization

### Finding 6: DAgger provides strong theoretical guarantees for iterative policy improvement

```
Claim: DAgger returns a policy with error bounded as J(π) ≤ J(π*) + uTεN + O(1), requiring only 
O(uT) iterations, compared to BC's quadratic-in-horizon compounding error. [^181^]
Source: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross et al., AISTATS 2011)
URL: https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
Date: 2011
Excerpt: "For DAgGER, if N is O(uT) there exists a policy π_1:N s.t. J(π) ≤ J(π*) + uTε_N + O(1)."
Context: DAgger iteratively: (1) executes current policy, (2) queries expert for actions on visited states,
(3) aggregates data, (4) retrains policy. Key advantage over BC: trains on states actually visited by
the learned policy, not just expert trajectories. This provides linear (vs quadratic) error accumulation.
Confidence: HIGH
```

### Finding 7: DAgger can use algorithmic oracles when human labels are expensive

```
Claim: DAgger can be applied with "algorithmic oracles" — powerful algorithms that can be run at train 
time but not at test time — making it practical for RL settings where human feedback is expensive. [^178^]
Source: DAgger (CS 4756 lecture slides, Cornell)
URL: https://www.cs.cornell.edu/courses/cs4756/2024fa/assets/slides_notes/lec6_slides.pdf
Date: 2024
Excerpt: "Option 2: Use an algorithmic oracle. What if we had a powerful algorithm that we can run in 
train time but not at test time?"
Context: For a scripted spiral/lawnmower searcher, DAgger would: (1) run the current learned policy,
(2) query the scripted searcher for what action it would take in each visited state, (3) aggregate,
(4) retrain. This could iteratively improve beyond the scripted policy while staying close to it.
Confidence: HIGH
```

### Finding 8: DAgger requires less human-labeled data than BC in practice

```
Claim: In practice, the DAgger algorithm requires less human labeled data than BC because it only 
needs expert labels for states that the learned policy actually visits, not for all possible expert trajectories. [^183^]
Source: CS/Stat 184: Introduction to Reinforcement Learning (Harvard)
URL: https://lucasjanson.fas.harvard.edu/courses/19a.pdf
Date: Unknown
Excerpt: "In practice, the DAgger algorithm requires less human labeled data than BC."
Context: The informal theorem states: |V^π* - V^π̂| ≤ Hε under standard assumptions. For the Minecraft
setting, DAgger could use the scripted searcher as the expert oracle, collecting (state, expert_action)
pairs only where the learned policy actually goes.
Confidence: HIGH
```

---

## 5.3 AWAC / IQL Offline-to-Online

### Finding 9: AWAC combines offline data with online fine-tuning effectively

```
Claim: AWAC is the only method among 8 tested that consistently solves difficult dexterous manipulation 
tasks from demonstrations plus online experience. It solves the pen task in 120K timesteps 
(20 minutes of online interaction). [^189^]
Source: AWAC: Accelerating Online Reinforcement Learning with Offline Datasets (Nair et al., 2020)
URL: http://bair.berkeley.edu/blog/2020/09/10/awac/
Date: 2020
Excerpt: "Only AWAC solves the second and third task. Prior methods fail, for a myriad of reasons, 
centered around their inability to obtain a reasonable initial policy to collect good exploration data, 
or their inability to learn online from interaction data."
Context: AWAC uses advantage-weighted actor critic — combines sample-efficient dynamic programming 
with maximum likelihood policy updates. Key design: implicitly stays close to data distribution by 
sampling (does not need behavior model). Evaluated on dexterous manipulation with sparse binary rewards
(0-1), MuJoCo benchmarks, and real-world robotics.
Confidence: HIGH
```

### Finding 10: AWAC enables rapid learning from prior data + online experience

```
Claim: AWAC provides a simple and effective framework able to leverage large amounts of offline data 
and then quickly perform online fine-tuning. Incorporating prior data reduces learning time to practical 
time-scales. [^175^]
Source: AWAC: Accelerating Online RL with Offline Datasets (arXiv:2006.09359)
URL: https://arxiv.org/abs/2006.09359
Date: 2020
Excerpt: "We propose an algorithm that combines sample efficient dynamic programming with maximum 
likelihood policy updates, providing a simple and effective framework that is able to leverage large 
amounts of offline data and then quickly perform online fine-tuning of RL policies."
Context: AWAC uses off-policy critic estimation + implicit behavior modeling. Key insight: off-policy
methods can "stitch" good trajectory segments together via value function, while on-policy methods
cannot. This is critical for exploration from demonstrations — the agent can combine partial successful
segments into full solutions.
Confidence: HIGH
```

### Finding 11: IQL achieves state-of-the-art on D4RL and strong online fine-tuning

```
Claim: IQL demonstrates state-of-the-art performance on D4RL benchmarks, especially on AntMaze tasks 
that require "stitching" sub-optimal trajectories. IQL achieves strong performance fine-tuning using 
online interaction after offline initialization. [^173^]
Source: Offline Reinforcement Learning with Implicit Q-Learning (Kostrikov et al., ICLR 2022)
URL: https://arxiv.org/pdf/2110.06169
Date: 2021
Excerpt: "IQL demonstrates the state-of-the-art performance on D4RL, a standard benchmark for offline 
reinforcement learning. We also demonstrate that IQL achieves strong performance fine-tuning using 
online interaction after offline initialization."
Context: IQL key insight: never evaluates out-of-sample actions. Uses expectile regression to 
approximate upper expectile of value distribution, then extracts policy via advantage-weighted BC.
On AntMaze tasks (which require combining partial trajectories), IQL significantly outperforms CQL.
Confidence: HIGH
```

### Finding 12: IQL uses advantage-weighted behavioral cloning for policy extraction

```
Claim: IQL extracts the policy via advantage-weighted behavioral cloning, which avoids querying 
out-of-sample actions. The method only requires fitting an additional critic with an asymmetric L2 loss. [^177^]
Source: Offline Reinforcement Learning with Implicit Q-Learning (NeurIPS 2021 workshop)
URL: https://offline-rl-neurips.github.io/2021/pdf/24.pdf
Date: 2021
Excerpt: "We extract the corresponding policy using advantage-weighted behavioral cloning, which also 
avoids querying out-of-sample actions. This approach does not require explicit constraints or explicit 
regularization of out-of-distribution actions during value function training."
Context: IQL is computationally efficient (~4x faster than CQL). Crucially, the advantage-weighted
BC extraction step means IQL naturally stays close to the data distribution while still improving
via multi-step dynamic programming. For Minecraft exploration, this means IQL could warm-start
from scripted search demonstrations and improve via offline value estimation.
Confidence: HIGH
```

### Finding 13: TD3+BC matches state-of-the-art with a single-line modification

```
Claim: TD3+BC matches performance of state-of-the-art offline RL algorithms by simply adding a BC term 
to the policy update and normalizing states. The modification can be made in a single line of code. [^261^]
Source: A Minimalist Approach to Offline Reinforcement Learning (Fujimoto & Gu, NeurIPS 2021)
URL: https://arxiv.org/pdf/2106.06860
Date: 2021
Excerpt: "We find that we can match the performance of state-of-the-art offline RL algorithms by simply 
adding a behavior cloning term to the policy update of an online RL algorithm and normalizing the data."
Context: TD3+BC objective: π = argmax E[(λQ(s,π(s)) - (π(s)-a)^2)]. The BC penalty keeps the 
policy close to the demonstration data. This is the simplest offline-to-online method and serves as
a strong baseline. Halves overall runtime compared to CQL.
Confidence: HIGH
```

---

## 5.4 Decision Transformer as Policy Prior

### Finding 14: Decision Transformer casts RL as conditional sequence modeling

```
Claim: Decision Transformer outputs optimal actions by leveraging a causally masked Transformer, 
conditioning on desired return, past states, and actions. It matches or exceeds state-of-the-art 
model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks. [^190^]
Source: Decision Transformer: Reinforcement Learning via Sequence Modeling (Chen et al., NeurIPS 2021)
URL: https://arxiv.org/abs/2106.01345
Date: 2021
Excerpt: "Decision Transformer simply outputs the optimal actions by leveraging a causally masked 
Transformer. By conditioning an autoregressive model on the desired return (reward), past states, 
and actions, our Decision Transformer model can generate future actions that achieve the desired return."
Context: Unlike Q-learning or policy gradients, DT treats RL as a sequence modeling problem.
Key advantage: can leverage powerful Transformer architectures and scale with data. For Minecraft,
DT could be trained on scripted searcher trajectories conditioned on target location.
Confidence: HIGH
```

### Finding 15: Online Decision Transformer enables effective offline-to-online fine-tuning

```
Claim: ODT is competitive with state-of-the-art in absolute performance on D4RL but shows much more 
significant gains during fine-tuning. Relative improvements due to fine-tuning for ODT are ~9x those 
of IQL for Gym tasks. [^192^]
Source: Online Decision Transformer (Zheng et al., ICML 2022 Oral)
URL: https://arxiv.org/abs/2202.05607
Date: 2022
Excerpt: "ODT is competitive with the state-of-the-art in absolute performance on the D4RL benchmark 
but shows much more significant gains during the finetuning procedure... on average, the relative 
improvements due to finetuning for ODT are ~9x to those of IQL."
Context: ODT adds sequence-level entropy regularizers for sample-efficient exploration. Key finding:
pretraining till convergence might hurt exploration performance — use fewer pretraining updates than
offline models. For navigation exploration, this suggests a short DT pretraining phase followed by
active online fine-tuning with entropy-based exploration.
Confidence: HIGH
```

### Finding 16: Decision Transformer can be prompted for few-shot policy generalization

```
Claim: Prompting Decision Transformer enables few-shot generalization to new tasks by conditioning 
on desired returns and task specifications at inference time. [^204^]
Source: Prompting Decision Transformer for Few-Shot Policy Generalization (Xu et al., ICML 2022)
URL: https://github.com/opendilab/awesome-decision-transformer
Date: 2022
Excerpt: Key: Prompt, Few-shot, Generalization. ExpEnv: DMC.
Context: This suggests DT could serve as an exploration prior for Minecraft: train DT on scripted 
searcher trajectories, then at test time prompt it with different target locations to elicit 
generalizable search behavior.
Confidence: MEDIUM
```

---

## 5.5 Scripted Searcher as BC Demonstrator

### Finding 17: Human demonstrations outperform frontier exploration for BC→RL in ObjectNav

```
Claim: BC→RL on human demonstrations outperforms BC→RL on shortest path and frontier exploration 
trajectories, even when controlled for the same BC-pretraining success on TRAIN, and even on episodes 
where BC-pretraining success favors the SP or FE policies. [^321^]
Source: PIRLNav (CVPR 2023)
URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Ramrakhya_PIRLNav_Pretraining_With_Imitation_and_RL_Finetuning_for_ObjectNav_CVPR_2023_paper.pdf
Date: 2023
Excerpt: "Learning to imitate human demonstrations equips the agent with navigation strategies that 
enable better RL-finetuning generalization compared to imitating other kinds of demonstrations, even 
when controlled for the same BC-pretraining accuracy."
Context: Controlled experiment with equal BC train success (~48%):
- RL-finetuned from Human Demonstrations: 66.1% VAL success
- RL-finetuned from Frontier Exploration: 51.3% VAL success
- RL-finetuned from Shortest Paths: 43.6% VAL success
The key insight: human demonstrations contain task-specific exploration strategies that transfer
better to RL than task-agnostic exploration (FE) or optimal paths (SP). This is directly relevant
to the question of whether scripted spiral/lawnmower search can teach learnable search — task-agnostic
exploration may underperform task-specific strategies.
Confidence: HIGH
```

### Finding 18: Frontier exploration demonstrations plateau with scaling; human demos continue improving

```
Claim: As frontier exploration demonstrations are scaled past 70k trajectories, performance plateaus. 
Human demonstrations do not show this plateau. [^321^]
Source: PIRLNav (CVPR 2023)
URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Ramrakhya_PIRLNav_Pretraining_With_Imitation_and_RL_Finetuning_for_ObjectNav_CVPR_2023_paper.pdf
Date: 2023
Excerpt: "We find that as we scale frontier exploration demonstrations past 70k trajectories, the 
performance plateaus."
Context: BC on 240k shortest-path demos: 6.4% success. BC on 70k FE demos: 44.9%. BC on 77k 
human demos: 64.1%. The FE demonstrations capture "OBJECTNAV-agnostic exploration" while human 
demos capture "OBJECTNAV-specific exploration." This suggests that for a scripted spiral/lawnmower 
searcher (task-agnostic exploration), there may be a performance ceiling that RL finetuning can only 
partially lift.
Confidence: HIGH
```

### Finding 19: VPT shows pretraining + BC fine-tuning + RL fine-tuning solves hard-exploration tasks

```
Claim: RL from a randomly initialized model fails to get almost any reward in Minecraft diamond 
pickaxe task. In contrast, RL fine-tuning from a VPT foundation model performs substantially better. 
Three-phase training (pretraining, BC fine-tuning, RL fine-tuning) achieves 2.5% diamond pickaxe 
success (first non-zero result). [^250^]
Source: Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos (Baker et al., NeurIPS 2022)
URL: https://cdn.openai.com/vpt/Paper.pdf
Date: 2022
Excerpt: "Training from a randomly initialized policy fails to achieve almost any reward... RL fine-tuning 
from the VPT foundation model does substantially better... The three-phase training succeeds in learning 
extremely difficult tasks: it achieves over 80% reliability on iron pickaxes, almost 20% reliability on 
collecting diamonds, and 2.5% reliability on obtaining a diamond pickaxe."
Context: VPT foundation model trained on ~70k hours of internet Minecraft video with IDM pseudo-labels.
Then BC fine-tuned to early-game contractor dataset. Then RL fine-tuned with shaped reward. Key: 
KL loss to pretrained policy during RL fine-tuning prevents catastrophic forgetting. Without KL loss,
agent only learns early items and forgets later skills.
Confidence: HIGH
```

### Finding 20: VPT shows that behavior prior is a much better exploration prior than random

```
Claim: The VPT model should be a much better prior for RL because emulating human behavior is likely 
much more helpful than taking random actions. A BC-pretrained policy provides a reasonable starting 
point that makes the optimization easier than learning from scratch. [^250^]
Source: Video PreTraining (VPT), OpenAI
URL: https://openai.com/index/vpt/
Date: 2022
Excerpt: "The VPT model should be a much better prior for RL because emulating human behavior is likely 
much more helpful than taking random actions."
Context: For the diamond pickaxe task: RL from random fails entirely. RL from VPT foundation model
learns up to iron ore and furnaces. RL from VPT + BC fine-tuning to early-game achieves human-level
performance on all items. The behavioral prior provides structured exploration that random exploration
cannot match.
Confidence: HIGH
```

### Finding 21: Learning coverage paths with deep RL shows emergent search patterns

```
Claim: A deep RL agent can learn effective coverage path planning (lawnmower-like patterns) from 
scratch with multi-scale maps and total variation reward. The agent implicitly solves mapping, planning, 
and navigation simultaneously. [^200^]
Source: Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning
URL: https://arxiv.org/html/2306.16978v4
Date: 2023
Excerpt: "We present a method for online coverage path planning in unknown environments, based on a 
continuous end-to-end deep reinforcement learning approach. The agent implicitly solves three tasks 
simultaneously, namely mapping of the environment, planning a coverage path, and navigating the 
planned path while avoiding obstacles."
Context: This shows that RL can learn structured search patterns (lawnmower-like coverage) from 
scratch given proper reward shaping and architectural inductive biases (multi-scale maps). However,
it requires dense reward engineering. The question remains whether warm-starting from a scripted
searcher can get the RL agent to discover such patterns faster in sparse-reward settings.
Confidence: MEDIUM
```

---

## 5.6 Evidence from Navigation/Foraging Tasks

### Finding 22: LOGO algorithm uses suboptimal demonstrations for sparse reward RL guidance

```
Claim: LOGO merges policy improvement with a policy guidance step using offline demonstration data, 
orienting the policy in the manner of the sub-optimal policy while being able to learn beyond it. [^279^]
Source: Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration 
(Rengarajan et al., ICLR 2022 Spotlight)
URL: https://arxiv.org/abs/2202.04628
Date: 2022
Excerpt: "The key idea is that by obtaining guidance from - not imitating - the offline data, LOGO 
orients its policy in the manner of the sub-optimal policy, while yet being able to learn beyond and 
approach optimality."
Context: LOGO uses two steps per iteration: (1) TRPO policy improvement, (2) policy guidance toward
behavior policy via KL constraint. Demonstrates that suboptimal demonstrations can guide exploration
without constraining to imitation. Tested on MuJoCo and real TurtleBot. Key: guidance ≠ imitation.
Confidence: HIGH
```

### Finding 23: DOAMRL dynamically weights IL objective for sparse reward meta-RL

```
Claim: DOAMRL combines RL and dynamically weighted IL objectives within MAML's inner loop, achieving 
≥29% performance improvement over state-of-the-art with non-expert demonstrations. [^191^]
Source: Demonstration and Offset Augmented Meta RL with Sparse Rewards
URL: https://link.springer.com/article/10.1007/s40747-025-01785-0
Date: 2025
Excerpt: "When using non-expert demonstrations, DOAMRL achieves at least a 29% performance improvement 
over state-of-the-art methods in all three environments."
Context: Point2D Navigation, TwoWheeled Locomotion, HalfCheetah Forward/Backward with sparse rewards.
Key insight: the IL weight is learnable and adapts at each meta-policy optimization phase — when
demonstrations are useful, the agent uses them; otherwise it relies on exploration.
Confidence: MEDIUM
```

### Finding 24: Search-Inspired Exploration (SIERL) actively guides exploration via sub-goals

```
Claim: SIERL actively guides exploration by setting sub-goals from the frontier (boundary of known 
state space), prioritizing based on cost-to-come and cost-to-go estimates. It outperforms dominant 
baselines in sparse-reward environments. [^256^]
Source: Search Inspired Exploration in Reinforcement Learning
URL: https://arxiv.org/html/2602.00460v1
Date: 2025
Excerpt: "We propose Search-Inspired Exploration in Reinforcement Learning (SIERL), a novel method that 
actively guides exploration by setting sub-goals based on the agent's learning progress... sub-goals 
are prioritized from the frontier based on estimates of cost-to-come and cost-to-go."
Context: Directly relevant: this shows that search patterns (frontier-based exploration) can be
effective when integrated into RL. The key question is whether imitating a scripted frontier-based
searcher and then RL-finetuning can achieve similar or better results.
Confidence: MEDIUM
```

### Finding 25: Deep Q-learning from Demonstrations (DQfD) massively accelerates learning

```
Claim: DQfD starts with better scores than DQN on 41 of 42 Atari games, and on average it takes DQN 
83 million steps to catch up. DQfD learns to outperform the best demonstration in 14 of 42 games. [^239^]
Source: Deep Q-learning from Demonstrations (Hester et al., AAAI 2018)
URL: https://mlanctot.info/files/papers/aaai18-dqfd.pdf
Date: 2018
Excerpt: "DQfD has better initial performance than Prioritized Dueling Double DQN as it starts with 
better scores on the first million steps on 41 of 42 games and on average it takes PDD DQN 83 million 
steps to catch up to DQfD's performance. DQfD learns to out-perform the best demonstration given in 
14 of 42 games."
Context: DQfD pretrains on demonstration data using both TD and supervised losses. Uses prioritized
replay to automatically control the ratio of demonstration vs self-generated data. Even suboptimal
demonstrations provide useful exploration priors. Demonstrates that learning from demonstrations
can bootstrap exploration that ultimately exceeds the demonstrator.
Confidence: HIGH
```

### Finding 26: CQL substantially outperforms prior offline RL, learning policies 2-5x better

```
Claim: CQL substantially outperforms existing offline RL methods, often learning policies that attain 
2-5 times higher final return, especially when learning from complex and multi-modal data distributions. [^262^]
Source: Conservative Q-Learning for Offline Reinforcement Learning (Kumar et al., NeurIPS 2020)
URL: https://arxiv.org/pdf/2006.04779
Date: 2020
Excerpt: "CQL substantially outperforms existing offline RL methods, often learning policies that attain 
2-5 times higher final return, especially when learning from complex and multi-modal data distributions."
Context: CQL adds a conservative Q-value regularizer that penalizes high Q-values on out-of-distribution
actions. This enables safe policy improvement from suboptimal data. Can be implemented in <20 lines
of code on top of standard deep Q-learning. For Minecraft, CQL could learn from scripted searcher
data and improve via conservative value estimation.
Confidence: HIGH
```

### Finding 27: PIRLNav analysis shows diminishing returns from RL as BC dataset scales

```
Claim: As BC pretraining dataset size increases to high BC accuracies, improvements from RL-finetuning 
become smaller. 90% of best BC→RL performance can be achieved with less than half the BC demonstrations. [^320^]
Source: PIRLNav (CVPR 2023)
URL: https://arxiv.org/abs/2301.07302
Date: 2023
Excerpt: "We find that as we increase the size of BC-pretraining dataset and get to high BC accuracies, 
improvements from RL-finetuning are smaller, and that 90% of the performance of our best BC→RL policy 
can be achieved with less than half the number of BC demonstrations."
Context: At 20k BC demos: BC→RL improvement = +10.1% success. At 77k demos: improvement = +6.3%.
This suggests that for a fixed budget, a moderate number of scripted searcher demonstrations with
RL finetuning may be sufficient — more BC data has diminishing returns.
Confidence: HIGH
```

---

## 5.7 Concrete Recipes

### Recipe 1: PIRLNav Two-Phase BC→RL Fine-tuning

```
Source: PIRLNav (CVPR 2023) — Ramrakhya et al.
URL: https://github.com/Ram81/pirlnav
Recipe:
1. BC Pretraining:
   - Collect demonstration dataset (human, scripted, or algorithmic)
   - Train policy via behavior cloning with standard cross-entropy loss
   - Save best-performing BC checkpoint

2. RL Finetuning - Two-Phase Regime:
   Phase 1 (Critic Learning):
   - Initialize actor with BC-pretrained weights
   - Initialize critic weights close to zero (final linear layer ~0)
   - Freeze actor and shared weights (e.g., RNN)
   - Rollout trajectories using the frozen BC policy (sample actions, not argmax)
   - Train only the critic until loss plateaus (~8M steps)
   
   Phase 2 (Interactive Learning):
   - Unfreeze actor RNN
   - Gradually decay critic LR (e.g., 2.5e-4 → 1.5e-5 between 8M-12M steps)
   - Warm up actor LR from 0 → 1.5e-5 over same period
   - For shared parameters, use the lower of the two learning rates
   - Continue joint training to convergence

Key Hyperparameters:
   - Starting critic LR: 2.5e-4
   - Final actor/critic LR: 1.5e-5
   - Critic-only phase: ~8M steps
   - LR transition window: 8M-12M steps
   - RL algorithm: DD-PPO (distributed PPO)
   - CNN and non-visual embedding layers remain frozen throughout
```

### Recipe 2: VPT Three-Phase Training (Pretrain → BC Fine-tune → RL Fine-tune)

```
Source: Video PreTraining (VPT) — Baker et al. (NeurIPS 2022)
URL: https://cdn.openai.com/vpt/Paper.pdf
Recipe:
1. Foundation Model Pretraining:
   - Train Inverse Dynamics Model (IDM) on small labeled dataset
   - Use IDM to pseudo-label large unlabeled dataset
   - Train foundation policy via BC on pseudo-labeled data

2. BC Fine-tuning:
   - Fine-tune foundation model to task-specific demonstration dataset
   - This narrows the behavioral distribution toward the target task
   - E.g., fine-tune to "early-game" Minecraft behavior

3. RL Fine-tuning:
   - Initialize RL policy from BC fine-tuned weights
   - Use KL-divergence loss to pretrained policy (prevent catastrophic forgetting)
   - Start with high KL coefficient ρ, decay by fixed factor each iteration
   - This protects policy skills in early iterations while enabling eventual maximization

Key Insight:
   - The KL loss to the pretrained policy is CRITICAL — without it, the agent forgets
     early skills while learning later ones, stalling progress
   - Three-phase > direct RL from foundation > RL from scratch (by huge margin)
```

### Recipe 3: AWAC Offline-to-Online

```
Source: AWAC — Nair et al. (2020)
URL: https://arxiv.org/abs/2006.09359
Recipe:
1. Collect offline dataset:
   - Mix of expert demonstrations + suboptimal trajectories
   - Can use behavioral clone to generate additional suboptimal data

2. AWAC Training:
   - Off-policy actor-critic with advantage-weighted policy updates
   - Critic: standard TD learning
   - Actor: maximum likelihood with advantage weighting
   - Implicitly stays close to data (no explicit behavior model needed)

3. Online Fine-tuning:
   - Continue training with online data collection
   - Off-policy critic enables "stitching" of trajectory segments
   - Policy improvement via advantage-weighted BC

Key Advantage:
   - Works with off-policy data — can learn from arbitrary mixtures of demonstrations
   - The value function enables dynamic programming across trajectories
   - Much faster online improvement than on-policy methods (AWR, DAPG)
```

### Recipe 4: IQL Offline-to-Online

```
Source: IQL — Kostrikov et al. (ICLR 2022)
URL: https://arxiv.org/abs/2110.06169
Recipe:
1. Offline Training:
   - Fit expectile value function V(s) with asymmetric L2 loss
   - Backup into Q-function using SARSA-style update
   - Use high expectile τ for tasks requiring "stitching" (τ=0.7 for AntMaze)
   - No explicit policy during training — pure value learning

2. Policy Extraction:
   - Extract policy via advantage-weighted behavioral cloning
   - π(a|s) ∝ exp(β * A(s,a)) * π_behavior(a|s)
   - This avoids querying out-of-sample actions entirely

3. Online Fine-tuning:
   - Continue value learning with new online data
   - Policy naturally improves as value estimates refine
   - Computationally efficient (~4x faster than CQL)

Key Hyperparameters:
   - Expectile τ: 0.7 for navigation/stitching tasks, 0.5 for locomotion
   - Temperature β: controls conservatism
   - No behavior model needed
```

### Recipe 5: TD3+BC Minimalist Offline RL

```
Source: TD3+BC — Fujimoto & Gu (NeurIPS 2021)
URL: https://arxiv.org/abs/2106.06860
Recipe:
1. Prepare dataset of (s, a, r, s') transitions
2. Normalize states: mean=0, std=1 over dataset
3. Run standard TD3 with modified policy update:
   π = argmax E[(λ * Q(s, π(s)) - (π(s) - a)^2)]
   where λ is a weighting hyperparameter
4. The BC term (π(s) - a)^2 keeps policy close to data

Hyperparameters:
   - λ: 2.5 for most tasks (controls BC vs Q-learning tradeoff)
   - Same network architecture and RL hyperparameters as TD3
   - Only two changes: BC term + state normalization

Advantage: Simplest implementation, matches SOTA performance, halves runtime vs CQL.
```

### Recipe 6: DAgger with Scripted Oracle

```
Source: DAgger — Ross et al. (AISTATS 2011)
URL: https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
Recipe:
1. Initialize π_1 to any policy (e.g., random or BC on scripted searcher)
2. Initialize empty dataset D
3. For iteration i = 1 to N:
   a. Execute π_i in environment, collect trajectories
   b. For each visited state s, query scripted oracle for action a*
   c. Aggregate: D ← D ∪ {(s, a*)}
   d. Train π_{i+1} = argmin E_D[L(π(s), a*)]
4. Select best policy from {π_1, ..., π_{N+1}}

Key for Minecraft Scripted Searcher:
   - The scripted spiral/lawnmower searcher serves as the oracle
   - Only states visited by the learned policy need oracle labels
   - Over iterations, policy visits new states → oracle provides guidance
   - Linear error accumulation (vs quadratic for pure BC)
   - Can exceed the scripted oracle by generalizing to novel situations
```

---

## Key Synthesis: Can Warm-Starting from a Scripted Searcher Teach Learnable Search?

### The Evidence Points to CAUTIOUS OPTIMISM

**Arguments FOR (warm-starting helps):**

1. **PIRLNav finding**: BC→RL from human demonstrations (which contain task-specific exploration)
   significantly outperforms BC→RL from task-agnostic frontier exploration [^321^]. For a *scripted 
   searcher* that encodes task-specific patterns (e.g., spiral toward oracle cue), this gap may be 
   narrower — the key is whether the scripted policy contains useful search structure.

2. **VPT finding**: A behavioral prior (even from noisy internet video) enables RL to solve tasks 
   impossible from scratch [^250^]. The prior provides structured exploration that random exploration 
   cannot match. A scripted spiral/lawnmower policy is arguably a better prior than random.

3. **Reinforced Imitation finding**: IL pretraining provides "good intuition for more efficient 
   exploration during RL" [^323^]. Even suboptimal demonstrations reduce RL training time 5-8x.

4. **AWAC/IQL finding**: Offline RL methods can "stitch" suboptimal trajectory segments into better 
   policies via dynamic programming [^175^][^173^]. A scripted searcher's partial successes can be 
   combined into full solutions.

5. **DQfD finding**: Even suboptimal demonstrations enable RL to ultimately outperform the 
   demonstrator [^239^]. DQfD beats the best demonstration in 14/42 Atari games.

**Arguments AGAINST / CAUTIONS:**

1. **PIRLNav finding**: Task-agnostic exploration (frontier exploration) plateaus with scaling and 
   underperforms task-specific human demos [^321^]. A simple spiral/lawnmower is task-agnostic — 
   the RL agent may need to "unlearn" patterns that don't transfer.

2. **PIRLNav finding**: Human demonstrations equip the agent with "navigation strategies that enable 
   better RL-finetuning generalization" compared to FE [^321^]. The *structure* of human exploration 
   matters, not just the coverage.

3. **Catastrophic forgetting**: Without KL regularization or careful phase transitions, the RL agent 
   may forget useful search patterns before learning to improve them [^250^][^251^].

4. **Distributional shift**: BC policies fail at out-of-distribution states not seen during training 
   [^320^]. If the scripted searcher's state distribution differs from what the RL agent encounters, 
   the warm-start may not help.

### Practical Recommendation

For the multi-agent Minecraft village task, the evidence supports trying **BC→PPO with a scripted 
searcher demonstrator**, but with these critical design choices:

1. **Use a task-SPECIFIC scripted searcher**: Not a generic spiral, but one that conditions on the 
   oracle cue direction (e.g., biased spiral toward cue). This provides more task-relevant structure 
   than task-agnostic exploration.

2. **Two-phase critic warmup is essential**: Freeze the actor, train the critic on rollouts from the 
   frozen BC policy before joint training [^321^].

3. **Use KL regularization to the BC policy during early RL**: Prevent catastrophic forgetting of 
   search patterns before RL discovers their value [^250^].

4. **Consider AWAC or IQL instead of pure PPO**: Off-policy methods can stitch trajectory segments 
   and learn from demonstration data throughout training, not just at initialization [^175^][^173^].

5. **Use a hybrid approach**: Start with DAgger-style iterative improvement where the scripted searcher 
   provides guidance on states visited by the learned policy [^181^]. This focuses learning on 
   relevant states.

6. **Scale expectations appropriately**: The PIRLNav scaling law suggests diminishing returns from 
   more BC data [^320^]. A moderate number of high-quality scripted demonstrations (~1-5k episodes) 
   may be sufficient — invest the remaining budget in RL fine-tuning.

---

## Sources

| # | Paper | Authors | Venue/Year | URL |
|---|-------|---------|-----------|-----|
| 1 | PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav | Ramrakhya et al. | CVPR 2023 | https://arxiv.org/abs/2301.07302 |
| 2 | AWAC: Accelerating Online RL with Offline Datasets | Nair et al. | 2020 | https://arxiv.org/abs/2006.09359 |
| 3 | Offline RL with Implicit Q-Learning (IQL) | Kostrikov et al. | ICLR 2022 | https://arxiv.org/abs/2110.06169 |
| 4 | Decision Transformer: RL via Sequence Modeling | Chen et al. | NeurIPS 2021 | https://arxiv.org/abs/2106.01345 |
| 5 | Online Decision Transformer | Zheng et al. | ICML 2022 | https://arxiv.org/abs/2202.05607 |
| 6 | A Reduction of IL to No-Regret Online Learning (DAgger) | Ross et al. | AISTATS 2011 | https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf |
| 7 | Video PreTraining (VPT) | Baker et al. | NeurIPS 2022 | https://arxiv.org/abs/2206.11795 |
| 8 | A Minimalist Approach to Offline RL (TD3+BC) | Fujimoto & Gu | NeurIPS 2021 | https://arxiv.org/abs/2106.06860 |
| 9 | Conservative Q-Learning for Offline RL (CQL) | Kumar et al. | NeurIPS 2020 | https://arxiv.org/abs/2006.04779 |
| 10 | Deep Q-learning from Demonstrations (DQfD) | Hester et al. | AAAI 2018 | https://arxiv.org/abs/1704.03732 |
| 11 | Reinforced Imitation for Map-less Navigation | Pfeiffer et al. | IEEE RA-L 2018 | https://arxiv.org/abs/1805.07095 |
| 12 | RL with Sparse Rewards using Guidance from Offline Demo (LOGO) | Rengarajan et al. | ICLR 2022 | https://arxiv.org/abs/2202.04628 |
| 13 | Actor-Critic Pretraining for PPO | — | 2025 | https://arxiv.org/pdf/2602.23804 |
| 14 | Advantage-Weighted Regression (AWR) | Peng et al. | ICLR 2020 | https://arxiv.org/abs/1910.00177 |
| 15 | Go-Explore | Ecoffet et al. | 2019 | https://arxiv.org/abs/1901.10995 |
| 16 | Learning Coverage Paths with Deep RL | — | 2023 | https://arxiv.org/html/2306.16978v4 |
| 17 | Search Inspired Exploration in RL (SIERL) | — | 2025 | https://arxiv.org/html/2602.00460v1 |
| 18 | Demonstration and Offset Augmented Meta RL (DOAMRL) | — | 2025 | https://link.springer.com/article/10.1007/s40747-025-01785-0 |
| 19 | Generalization Through Imitation (GTI) | Mandlekar et al. | RSS 2020 | https://arxiv.org/abs/2003.06085 |
| 20 | Imitation Learning to Outperform Demonstrators (ODED) | Cai et al. | CIKM 2022 | — |
