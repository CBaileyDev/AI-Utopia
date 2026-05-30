## Dim 2: Episodic Memory & Count-Based Exploration

*Research on exploration methods for partially-observed RL navigation tasks*
*Date: 2025-07-01*
*Searches performed: 18 independent web searches*

---

### 2.1 Episodic Curiosity (Savinov et al. 2018)

**Claim:** Episodic Curiosity (EC) uses a learned reachability network to determine whether a current observation is "novel" based on how many environment steps it takes to reach from observations already stored in episodic memory, overcoming the "couch potato" problem of prediction-error-based methods like ICM. [^1^]
**Source:** Episodic Curiosity through Reachability (Savinov et al.)
**URL:** https://ar5iv.labs.arxiv.org/html/1810.02274
**Date:** 2018
**Excerpt:** "We propose a new curiosity method which uses episodic memory to form the novelty bonus. To determine the bonus, the current observation is compared with the observations in memory. Critically, the comparison is done based on how many environment steps it takes to reach the current observation from those in memory—which incorporates rich information about environment dynamics. This allows us to overcome the known 'couch-potato' issues of prior work."
**Context:** EC was tested on VizDoom, DMLab, and MuJoCo environments with visually rich 3D observations. The agent outperformed ICM by at least 2x in navigation tasks.
**Confidence:** High

**Claim:** The EC module consists of four components: an embedding network E, a comparator network C (together forming the reachability network R), an episodic memory buffer M, and a reward bonus estimation function B. [^2^]
**Source:** Episodic Curiosity through Reachability (ICLR 2019)
**URL:** https://openreview.net/pdf/a81ed9381b42322b6099ca895beadba6a985c8d9.pdf
**Date:** 2018
**Excerpt:** "The episodic curiosity (EC) module takes the current observation o as input and produces a reward bonus b. The module consists of both parametric and non-parametric components. There are two parametric components: an embedding network E: O → R^n and a comparator network C: R^n × R^n → [0,1]. Those parametric components are trained together to predict reachability as parts of the reachability network. There are also two non-parametric components: an episodic memory buffer M and a reward bonus estimation function B."
**Context:** The embedding network E maps observations to feature vectors. The comparator network predicts whether two observations are reachable within k steps. Memory stores embeddings of past observations from the current episode.
**Confidence:** High

**Claim:** The reachability network is trained as a binary classifier using positive examples (observations within k steps of each other) and negative examples (observations more than γk steps apart), with a logistic regression loss. [^3^]
**Source:** Episodic Curiosity through Reachability - Supplementary Material
**URL:** https://ar5iv.labs.arxiv.org/html/1810.02274
**Date:** 2018
**Excerpt:** "The pairs (o_i, o_j) where |i-j| ≤ k are taken as positive (reachable) examples while the pairs with |i-j| > γk become negative examples. The hyperparameter γ is necessary to create a gap between positive and negative examples. In the end, the network is trained with logistic regression loss to output the probability of the positive (reachable) class."
**Context:** Online training of the R-network is also possible, where the network is retrained periodically using on-policy data.
**Confidence:** High

**Claim:** PPO + EC is 1.84x slower than PPO alone and adds 13M trainable parameters (vs 1.7M for PPO and 2M for PPO+ICM). However, the authors note that "a resource-consuming Resnet-18 is not needed for the R-network—a much simpler model may work as well." [^4^]
**Source:** Episodic Curiosity through Reachability - Supplementary S7
**URL:** https://ar5iv.labs.arxiv.org/html/1810.02274
**Date:** 2018
**Excerpt:** "PPO + ICM is 1.09x slower than PPO and PPO + EC (our method) is 1.84x slower than PPO. In terms of the number of parameters, R-network brings 13M trainable variables, while PPO alone was 1.7M and PPO + ICM was 2M."
**Context:** Memory consumption for stored memories is very modest (400 KB for 200 embeddings of 512 floats). The most computationally intensive part is memory reachability queries, which are computed in parallel via mini-batching.
**Confidence:** High

**Claim:** EC has a key limitation: it encourages going to the very end of blind alleys to get low similarity scores and high intrinsic reward, even when going into empty blind alleys is not beneficial for exploration. GoBI found that EC agents "prefer going to the room corners." [^5^]
**Source:** Maximizing Episodic Reachability with World Models (Fu et al., ICML 2023)
**URL:** https://proceedings.mlr.press/v202/fu23c/fu23c.pdf
**Date:** 2023
**Excerpt:** "EC encourages going to the very end of the blind alley to reach the state with low similarity score and high intrinsic reward, even though going into an empty blind alley is not beneficial for exploration and wastes time that can be used to explore other parts of the environment."
**Context:** This was observed on MiniGrid MultiRoom environments where EC was adapted for testing.
**Confidence:** High

---

### 2.2 NGU Episodic Component (Badia et al. 2020)

**Claim:** Never Give Up (NGU) constructs an episodic memory-based intrinsic reward using k-nearest neighbors over the agent's recent experience, combined with a lifelong novelty module (RND). It was the first algorithm to achieve non-zero rewards in Pitfall! without demonstrations. [^6^]
**Source:** Never Give Up: Learning Directed Exploration Strategies (Badia et al., ICLR 2020)
**URL:** https://openreview.net/forum?id=Sye57xStvB
**Date:** 2020
**Excerpt:** "We construct an episodic memory-based intrinsic reward using k-nearest neighbors over the agent's recent experience to train the directed exploratory policies, thereby encouraging the agent to repeatedly revisit all states in its environment. A self-supervised inverse dynamics model is used to train the embeddings of the nearest neighbour lookup, biasing the novelty signal towards what the agent can control."
**Context:** NGU combines episodic novelty (within-episode) with lifelong novelty (across episodes via RND). It uses UVFA to learn a family of policies with different exploration/exploitation trade-offs.
**Confidence:** High

**Claim:** The NGU episodic intrinsic reward is computed as the inverse of the sum of kernel similarities between the current state embedding and its k-nearest neighbors in episodic memory: r^episodic_t ≈ 1 / sqrt(sum_{φ_i in N_k} K(φ(x_t), φ_i)) + c), where K uses an inverse kernel based on Euclidean distance. [^7^]
**Source:** Exploration Strategies in Deep Reinforcement Learning (Lilian Weng)
**URL:** https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
**Date:** 2020
**Excerpt:** "r^episodic_t ≈ 1 / (sqrt(sum_{φ_i in N_k} K(φ(x_t), φ_i)) + c) where K(x, y) is a kernel function for measuring the distance between two samples. N_k is a set of k nearest neighbors in M according to K(.,.). In the paper, K(x, y) is configured to be the inverse kernel: K(x, y) = ε / (d^2(x, y)/d^2_m + ε)."
**Context:** The embedding function φ is trained via an inverse dynamics model (predicting action a_t from consecutive states), making the novelty signal focus on controllable aspects of the environment.
**Confidence:** High

**Claim:** The combined NGU intrinsic reward is r^i_t = r^episodic_t × clip(α_t, 1, L), where α_t is the lifelong novelty from RND. This design enables rapid discouragement of revisiting the same state within an episode, while slowly discouraging revisiting states visited many times across episodes. [^8^]
**Source:** Never Give Up: Learning Directed Exploration Strategies (ICLR 2020)
**URL:** https://openreview.net/forum?id=Sye57xStvB
**Date:** 2020
**Excerpt:** "The design of NGU enables it to have two nice properties: 1) Rapidly discourages revisiting the same state within the same episode; 2) Slowly discourages revisiting states that have been visited many times across episodes."
**Context:** NGU was built on top of by Agent57, which added a meta-controller (bandit) to dynamically select exploration/exploitation policies, achieving above-human performance on all 57 Atari games.
**Confidence:** High

**Claim:** A simplified but faithful formulation of NGU (excluding RND and UVFA) retains the embedding network, inverse dynamics model, episodic memory, and k-NN novelty computation. This reduced formulation has been successfully extended to multi-agent settings. [^9^]
**Source:** Extending NGU to Multi-Agent RL: A Preliminary Study
**URL:** https://arxiv.org/pdf/2512.01321
**Date:** 2025
**Excerpt:** "This reduced but faithful formulation preserves NGU's core mechanism of rewarding novelty while keeping the method computationally tractable for multi-agent experiments."
**Context:** The multi-agent extension uses individual episodic memory and intrinsic reward per agent, with options for shared replay buffers and novelty sharing.
**Confidence:** High

---

### 2.3 Count-Based & Pseudo-Count Methods

**Claim:** Bellemare et al. (2016) introduced pseudo-counts derived from a density model to generalize count-based exploration to non-tabular RL. Using a CTS density model over raw pixels, they achieved state-of-the-art on Montezuma's Revenge. [^10^]
**Source:** Unifying Count-Based Exploration and Intrinsic Motivation (Bellemare et al., NeurIPS 2016)
**URL:** https://arxiv.org/abs/1606.01868
**Date:** 2016
**Excerpt:** "We use density models to measure uncertainty, and propose a novel algorithm for deriving a pseudo-count from an arbitrary density model. This technique enables us to generalize count-based exploration algorithms to the non-tabular case. We apply our ideas to Atari 2600 games, providing sensible pseudo-counts from raw pixels."
**Context:** The pseudo-count is derived from the prediction gain of the density model—how much the model's prediction changes after observing a new state. States in familiar regions get higher pseudo-counts.
**Confidence:** High

**Claim:** Tang et al. (2017) proposed using SimHash (Locality-Sensitive Hashing) to convert continuous, high-dimensional states to discrete hash codes for counting. The intrinsic reward is r^i(s) = β / sqrt(N(φ(s))), where φ(s) = sgn(A·g(s)) is the hash function. [^11^]
**Source:** #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning (Tang et al., ICLR 2017)
**URL:** https://openreview.net/pdf/ee025e0524031667162985b357a4942ab9bb62a4.pdf
**Date:** 2017
**Excerpt:** "The main idea is to use locality-sensitive hashing (LSH) to convert continuous, high-dimensional data to discrete hash codes... SimHash retrieves a binary code of state s ∈ S as φ(s) = sgn(A g(s)) ∈ {-1, 1}^k, where g : S → R^d is an optional preprocessing function and A is a k × d matrix with i.i.d. entries drawn from a standard Gaussian distribution."
**Context:** The value k controls granularity—higher values lead to fewer collisions and more state distinction. This method adds minimal computational overhead and can be combined with any RL algorithm.
**Confidence:** High

**Claim:** Ostrovski et al. (2017) extended pseudo-counts using PixelCNN, a neural density model, dramatically improving state-of-the-art on several hard Atari games. They found that the mixed Monte Carlo update is a "powerful facilitator of exploration in the sparsest of settings." [^12^]
**Source:** Count-Based Exploration with Neural Density Models (Ostrovski et al., ICML 2017)
**URL:** https://arxiv.org/abs/1703.01310
**Date:** 2017
**Excerpt:** "We demonstrate the use of PixelCNN, an advanced neural density model for images, to supply a pseudo-count... We combine PixelCNN pseudo-counts with different agent architectures to dramatically improve the state of the art on several hard Atari games. One surprising finding is that the mixed Monte Carlo update is a powerful facilitator of exploration in the sparsest of settings."
**Context:** This work showed that the quality of the density model matters significantly for exploration performance.
**Confidence:** High

**Claim:** Martin et al. (2017) introduced φ-Exploration-Bonus (φ-EB), which computes generalized visit-counts in the same feature space used for value function approximation, making it simpler and less computationally expensive than density model approaches. [^13^]
**Source:** Count-Based Exploration in Feature Space for Reinforcement Learning (Martin et al., IJCAI 2017)
**URL:** https://arxiv.org/abs/1706.08090
**Date:** 2017
**Excerpt:** "We exploit the feature map that is used for value function approximation, and construct a density model over the transformed feature space... This makes it simpler to implement and less computationally expensive than some existing proposals. Our evaluation demonstrates that this simple approach achieves near state-of-the-art performance on high-dimensional RL benchmarks."
**Context:** The method uses a product of independent factor distributions over individual features, with Krichevsky-Trofimov (KT) estimators. The bonus is R^φ_t(s,a) = β / sqrt(N^φ_t(s)).
**Confidence:** High

**Claim:** NovelD (Zhang et al., 2021) uses a first-visit episodic count combined with RND novelty difference: the intrinsic reward is max[novelty(o_{t+1}) - α·novelty(o_t), 0] × 1{N_epi(s_{t+1}) = 1}. This only gives non-zero reward for first-visit states. [^14^]
**Source:** NovelD: A Simple yet Effective Exploration Criterion (Zhang et al., NeurIPS 2021)
**URL:** https://proceedings.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf
**Date:** 2021
**Excerpt:** "NovelD only assigns non-zero rewards to a state when it is visited for the first time in the episode... NovelD leads to a more focused exploration at the boundary and broader state coverage."
**Context:** NovelD was shown to explore all rooms in a 7-room MiniGrid environment consistently, while RND got stuck in room 5 even after 10M steps. However, NovelD requires discrete state information (full grid) rather than partial observations.
**Confidence:** High

**Claim:** BeBold (Zhang et al., 2021) proposes the "regulated difference of inverse visitation counts" as an intrinsic reward criterion that pushes the agent to explore beyond the boundary of explored regions, mitigating short-sightedness and detachment issues in count-based methods. [^15^]
**Source:** BeBold: Exploration Beyond the Boundary of Explored Regions (Zhang et al., ICLR 2021)
**URL:** https://arxiv.org/abs/2012.08621
**Date:** 2021
**Excerpt:** "We propose the regulated difference of inverse visitation counts as a simple but effective criterion for IR. The criterion helps the agent explore Beyond the Boundary of explored regions and mitigates common issues in count-based methods, such as short-sightedness and detachment."
**Context:** BeBold solves 12 of the most challenging procedurally-generated MiniGrid tasks with 120M steps without curriculum learning, while previous SOTA only solved 50%.
**Confidence:** High

**Claim:** RIDE (Raileanu & Rocktaschel, 2020) uses an episodic novelty bonus that is the product of a count-based term and the difference between consecutive state embeddings: b_RIDE(s_t) = ||φ(s_{t+1}) - φ(s_t)||_2 / sqrt(N_c(s_t)). The embedding is learned via inverse and forward dynamics models. [^16^]
**Source:** Rewarding Impact-Driven Exploration (RIDE) (Raileanu & Rocktaschel, ICLR 2020)
**URL:** https://e3bagent.github.io/
**Date:** 2020
**Excerpt:** "RIDE defines a bonus based on the distance between the embeddings of two consecutive observations... The motivation for the second term in the bonus is to reward the agent for taking actions which cause significant changes in the environment."
**Context:** RIDE does not use a global novelty bonus, only an episodic one. It was designed for procedurally-generated environments where agents need to explore efficiently within each episode.
**Confidence:** High

---

### 2.4 Hash-Based Counting (SimHash, MinHash)

**Claim:** SimHash (Charikar, 2002) measures similarity by angular distance and produces binary hash codes. It is computationally efficient and the granularity is controlled by the hash dimension k. Higher k leads to fewer collisions and finer state distinction. [^17^]
**Source:** #Exploration: A Study of Count-Based Exploration for Deep RL (Tang et al., ICLR 2017)
**URL:** https://openreview.net/pdf/ee025e0524031667162985b357a4942ab9bb62a4.pdf
**Date:** 2017
**Excerpt:** "SimHash retrieves a binary code of state s ∈ S as φ(s) = sgn(A g(s)) ∈ {-1, 1}^k, where g : S → R^d is an optional preprocessing function and A is a k × d matrix... The value for k controls the granularity: higher values lead to fewer collisions."
**Context:** The method is simple: collect states, compute hash codes, update counts in a hash table, and add β/sqrt(n(φ(s))) to the reward. No neural network training is needed for the counting mechanism itself.
**Confidence:** High

**Claim:** Tang et al. also proposed a learned hashing approach using an autoencoder with a binary bottleneck layer. The autoencoder is trained to ensure distinct states map to distinct binary codes by adding uniform noise to the sigmoid output. [^18^]
**Source:** #Exploration: A Study of Count-Based Exploration for Deep RL (Tang et al., ICLR 2017)
**URL:** https://openreview.net/pdf/ee025e0524031667162985b357a4942ab9bb62a4.pdf
**Date:** 2017
**Excerpt:** "By choosing uniform noise with a sufficiently high variance, the AE is only capable of reconstructing distinct inputs s if its hidden dense layer outputs values b(s) that are sufficiently far apart from each other."
**Context:** The learned hash code is then projected to lower dimension via SimHash. This performed better than raw SimHash on some Atari games but requires additional training.
**Confidence:** High

---

### 2.5 Memory-Augmented RL for Exploration

**Claim:** MERLIN (Wayne et al., 2018) uses an external episodic memory (differentiable neural computer) to store and retrieve past experiences, enabling agents to solve navigation tasks like water mazes and T-mazes by learning to contextually load episodic memories without interference. [^19^]
**Source:** Unsupervised Predictive Memory in a Goal-Directed Agent (Wayne et al., Nature 2018)
**URL:** https://web.stanford.edu/class/cs379c/class_messages_listing/curriculum/Annotated_Readings/WayneetalCoRR-18_Annotated.pdf
**Date:** 2018
**Excerpt:** "MERLIN learned to explore the mazes and to store relevant information about them in memory and retrieve it without interference to relocate the platforms from any starting position."
**Context:** MERLIN combines a Memory-Based Predictor (MBP) with a policy. The MBP uses an external memory to store and retrieve information, while the policy uses memory readouts for decision-making. This is more complex than simple episodic curiosity modules.
**Confidence:** High

**Claim:** Memory-augmented RL for image-goal navigation (Mezghani et al., 2021) uses an attention-based end-to-end model with episodic memory and self-supervised state embeddings, achieving SOTA on Gibson dataset from RGB input alone. [^20^]
**Source:** Memory-Augmented Reinforcement Learning for Image-Goal Navigation (Mezghani et al., 2021)
**URL:** https://arxiv.org/abs/2101.05181
**Date:** 2021
**Excerpt:** "Our method is based on an attention-based end-to-end model that leverages an episodic memory to learn to navigate. First, we train a state-embedding network in a self-supervised fashion, and then use it to embed previously-visited states into the agent's memory. Our navigation policy takes advantage of this information through an attention mechanism."
**Context:** The episodic memory improved SPL by +6% over a data-augmented baseline. Data augmentation alone improved by +12% over vanilla RL baseline.
**Confidence:** High

---

### 2.6 E3B: Elliptical Episodic Bonuses

**Claim:** E3B (Henaff et al., NeurIPS 2022) extends count-based episodic bonuses to continuous state spaces using an elliptical bonus: b_E3B(s_t) = φ(s_t)^T [sum_{i=t_0}^{t-1} φ(s_i)φ(s_i)^T + λI]^{-1} φ(s_t). This is a continuous analog of inverse episodic counts. [^21^]
**Source:** Exploration via Elliptical Episodic Bonuses (Henaff et al., NeurIPS 2022)
**URL:** https://arxiv.org/abs/2210.05805
**Date:** 2022
**Excerpt:** "If φ is a one-hot encoding of the state, then C_{t-1} will be a diagonal matrix whose entries contain the counts corresponding to each state encountered in the episode. Its inverse will also be a diagonal matrix whose entries are inverse state visitation counts, and the bilinear form reads off the entry corresponding to the current state, yielding a bonus of 1/N_c(s_t)."
**Context:** E3B achieves SOTA on 16 MiniHack tasks without task-specific inductive biases. It also outperforms existing methods in reward-free exploration on Habitat.
**Confidence:** High

**Claim:** E3B uses an inverse dynamics model to learn the embedding φ, which captures controllable aspects of the environment. In high-dimensional settings, a fixed random network can also work well as the feature encoder. [^22^]
**Source:** Exploration via Elliptical Episodic Bonuses (Henaff et al., NeurIPS 2022)
**URL:** https://github.com/facebookresearch/e3b
**Date:** 2022
**Excerpt:** "The embedding is learned using an inverse dynamics model in order to capture controllable aspects of the environment... In our experiments, we compare this to other approaches such as using the policy network or random networks to produce state embeddings."
**Context:** The ridge regularization λ was tuned over {0.01, 0.1, 1.0} with 1.0 as the final value. The intrinsic reward coefficient β was tuned over {1.0, 0.1, 0.01, 0.001, 0.0001}.
**Confidence:** High

**Claim:** The effectiveness of methods like RIDE, AGAC, and NovelD critically relies on a count-based episodic term in their exploration bonus. When the count-based term is removed, all three methods fail to learn. [^23^]
**Source:** Exploration via Elliptical Episodic Bonuses (Henaff et al., NeurIPS 2022)
**URL:** https://discovery.ucl.ac.uk/id/eprint/10173689/1/2954_exploration_via_elliptical_epi.pdf
**Date:** 2022
**Excerpt:** "Figure 1 shows results for the three methods with and without their respective count-based episodic terms, on one of the MiniGrid environments used in prior work. When the count-based terms are removed, all three methods fail to learn."
**Context:** However, count-based episodic bonuses fail in environments where each state is unique (e.g., with pixel-based observations, noise, or time-varying elements like clocks or TV screens).
**Confidence:** High

---

### 2.7 GoBI: Maximizing Episodic Reachability with World Models

**Claim:** GoBI (Fu et al., ICML 2023) combines lifelong novelty with an episodic intrinsic reward designed to maximize stepwise reachability expansion. It uses a learned world model to predict future states with random actions, and assigns high intrinsic reward to states with more unique predictions not already in episodic memory. [^24^]
**Source:** Go Beyond Imagination: Maximizing Episodic Reachability with World Models (Fu et al., ICML 2023)
**URL:** https://proceedings.mlr.press/v202/fu23c.html
**Date:** 2023
**Excerpt:** "We apply learned world models to generate predicted future states with random actions. States with more unique predictions that are not in episodic memory are assigned high intrinsic rewards. Our method greatly outperforms previous state-of-the-art methods on 12 of the most challenging Minigrid navigation tasks."
**Context:** GoBI requires about 2x the wallclock time of NovelD for the same number of environment steps (10.65h vs 5.46h for 10M MiniGrid steps). It also requires pre-training a forward dynamics model (1e5 random steps).
**Confidence:** High

**Claim:** GoBI's intrinsic reward is: r_t = (m_{t+1} - m_t) / sqrt(N(o_{t+1})), where m_t is the size of the episodic buffer and N is the lifelong count. The episodic buffer stores visited states plus states reachable within k steps (predicted by the world model). [^25^]
**Source:** Maximizing Episodic Reachability with World Models (Fu et al., ICML 2023)
**URL:** https://proceedings.mlr.press/v202/fu23c/fu23c.pdf
**Date:** 2023
**Excerpt:** "GoBI: (m_{t+1} - m_t) / sqrt(N(o_{t+1})), where m_t is the size of the episodic buffer M and N(o_{t+1}) is the lifelong count of the observation."
**Context:** GoBI uses 360° panoramic views as input for world model prediction on MiniGrid, and k=1 step forward prediction with all 7 discrete actions. This makes it specifically suited for partially observable navigation.
**Confidence:** High

---

### 2.8 Comparative Assessment

#### When Methods Work vs Fail

**Claim:** Count-based episodic bonuses (NovelD, RIDE) fail completely in environments with noise or when each state is unique, because N_c(s) is always 1 and the episodic bonus loses meaning. In contrast, similarity-based methods (E3B, EC, NGU) continue to work. [^26^]
**Source:** Episodic Novelty Through Temporal Distance (ETD paper)
**URL:** https://arxiv.org/html/2501.15418v1
**Date:** 2025
**Excerpt:** "Our results indicate that NovelD, a count-based method, completely failed to effectively guide exploration [with noise], as the episodic rewards based on counts no longer provided useful information. In contrast, similarity-based methods such as E3B and DEIR continued to perform reasonably well."
**Context:** In MiniGrid with Gaussian noise (variance 0.1), NovelD failed while ETD, E3B, and DEIR maintained performance.
**Confidence:** High

**Claim:** The "noisy TV problem" (Burda et al., 2018) is a classic failure case where intrinsic motivation methods confuse aleatoric uncertainty (environment noise) with epistemic uncertainty (model ignorance). Methods relying on prediction error or novelty get stuck at sources of uncontrollable randomness. [^27^]
**Source:** Noise-Robust Exploration Via Learning Progress Monitoring
**URL:** https://arxiv.org/html/2509.25438v1
**Date:** 2025
**Excerpt:** "The noisy-TV problem is a classic failure case when the intrinsic motivation mechanism confuses aleatoric uncertainty with epistemic uncertainty... intrinsic motivation methods that rely on novelty or prediction error become increasingly driven by aleatoric uncertainty rather than epistemic uncertainty, causing the agent to focus on noise rather than meaningful transitions."
**Context:** EC (Savinov et al.) was specifically designed to avoid this by using reachability rather than prediction error, making it robust to the noisy TV problem.
**Confidence:** High

**Claim:** ICM (Intrinsic Curiosity Module) suffers from the "couch potato" problem—agents exploit actions with unpredictable but meaningless consequences (like tagging walls with a laser) instead of exploring meaningfully. EC overcomes this by rewarding only observations that take effort to reach. [^28^]
**Source:** Episodic Curiosity through Reachability (Savinov et al., 2018)
**URL:** https://techxplore.com/news/2018-10-method-instill-curiosity-agents.html
**Date:** 2018
**Excerpt:** "Surprise-based method (ICM) is persistently tagging walls with a laser-like science fiction gadget instead of exploring the maze. This behaviour is similar to the channel switching described before: even though the result of tagging is theoretically predictable, it is not easy and apparently requires a deep knowledge of physics."
**Context:** EC's reachability-based novelty avoids this by not rewarding easily-reached unpredictable observations.
**Confidence:** High

#### Compute Cost Comparison

**Claim:** Compute costs vary significantly across exploration methods:
- **RND**: Minimal overhead. Adds one forward pass through a predictor network. [^29^]
- **SimHash counting**: Minimal overhead. Just hash computation and table lookup. [^30^]
- **E3B**: Moderate overhead. Requires computing φ(s_t)^T C^{-1} φ(s_t) and updating inverse covariance, plus training inverse dynamics model. [^31^]
- **EC**: 1.84x slower than PPO. Adds 13M parameters (ResNet-18). Reachability queries are the main bottleneck. [^32^]
- **NGU (full)**: High overhead. Requires embedding network, k-NN search, RND, and UVFA family of policies. [^33^]
- **GoBI**: ~2x slower than NovelD. Requires world model pre-training and rollout predictions at each step. [^34^]
- **Pseudo-count (PixelCNN)**: High. Requires training and querying a neural density model. [^35^]

**Sources:** Multiple papers cited above
**Confidence:** High

#### Summary Table: Method Comparison

| Method | Type | Episodic? | Needs World Model? | Compute Cost | Works with Noise? | Best For |
|--------|------|-----------|-------------------|--------------|-------------------|----------|
| SimHash Count | Count-based | No | No | Minimal | No | Simple, low-dim states |
| Pseudo-count | Count-based | No | Yes (density) | High | Partially | Hard Atari games |
| RND | Prediction error | No | No (random target) | Low | Partially | Singleton MDPs |
| ICM | Prediction error | No | Yes (forward model) | Moderate | No | Dense reward tasks |
| EC | Reachability | Yes | Yes (reachability net) | Moderate-High | Yes | 3D navigation, sparse reward |
| NGU | k-NN similarity | Yes | Yes (IDF embed) | High | Yes | Very hard exploration |
| E3B | Elliptical bonus | Yes | Yes (ID model) | Moderate | Yes | Procedurally-generated CMDPs |
| NovelD | First-visit count | Yes | No (uses RND) | Moderate | No | MiniGrid, discrete states |
| RIDE | Count × embedding diff | Yes | Yes (ID+FWD) | Moderate | No | Procedurally-generated |
| GoBI | Reachability + world model | Yes | Yes (world model) | High | Yes | Partially observable nav |
| DEIR | Discriminator-based | Yes | Yes (discriminator) | Moderate | Yes | MiniGrid, ProcGen |
| BeBold | Inverse count difference | Partially | No | Low-Moderate | Partially | MiniGrid, NetHack |

---

### 2.9 Suitability for Fast-Sim PPO with Egocentric Vector Obs

**Claim:** For a fast-sim training setup with PPO, LSTM, CTDE, egocentric vector observations, and discrete actions, the most suitable exploration methods (in order of recommendation) are:

**Tier 1: Lightest Methods (Recommended for Fast-Sim)**

1. **Simple episodic count with vector state hashing**: Hash the egocentric vector observation (or a projection of it) using SimHash, maintain an episodic hash table, and reward with β/sqrt(N_episodic(hash(s))). This has near-zero compute overhead and works well for discrete-action navigation with partially observable vector states. The key challenge is choosing the right granularity for the hash function. [^36^]

2. **BeBold-style regulated count difference**: If a global novelty measure is available (e.g., simple RND with small network on vector obs), use the difference of inverse visitation counts to push exploration at boundaries. Very simple to implement with low overhead. [^37^]

3. **E3B-lite**: Use a small inverse dynamics model (or even a random network) to produce embeddings, then compute the elliptical bonus. The inverse covariance update is O(d^2) where d is embedding dimension (can be small, e.g., 32-64). No episodic memory of raw states needed—just the covariance matrix. [^38^]

**Tier 2: Moderate Overhead Methods**

4. **Simplified NGU (episodic only)**: Train a small inverse dynamics model to produce embeddings, use k-NN over an episodic buffer (size ~50-100). Skip the RND lifelong component and UVFA. This gives episodic novelty without the heavy machinery. [^39^]

5. **Simplified EC with small reachability network**: Use a small MLP (not ResNet-18) for the embedding and comparator networks. With vector observations, a 2-3 layer MLP is sufficient. Train on random policy data before main training or online. [^40^]

**Tier 3: Higher Overhead (Use with Caution)**

6. **GoBI**: Only if world model prediction is feasible for vector observations. Requires pre-training a forward dynamics model and doing k-step rollouts at each timestep. The 2x compute overhead may be acceptable if the environment is very hard to explore. [^41^]

**Claim:** For egocentric vector observations (not pixels), the following simplifications apply:
- No need for ResNet/CNN encoders; small MLPs suffice for all embedding/representation learning
- SimHash can be applied directly to the vector observation without preprocessing
- RND predictor can be very small (1-2 layer MLP) since observations are low-dimensional
- Episodic memory can store raw vector observations (no compression needed) for k-NN or reachability comparisons
- The episodic buffer can be kept small (50-200 entries) without significant performance loss [^42^]

**Source:** Synthesized from EC supplementary (S5.2) and E3B analysis
**Confidence:** Medium

**Claim:** In partially observed settings where the target is outside the observation radius ("blind" beyond ~16 blocks), episodic memory methods are particularly valuable because they allow the agent to remember what was observed at earlier positions and use that to guide directed exploration. Count-based methods that only track current observations will fail to provide useful signal when the agent is far from interesting states. [^43^]
**Source:** Synthesized from multiple papers on episodic exploration
**Context:** The episodic memory stores observations from the current episode, allowing the agent to assess novelty even when revisiting areas that are currently outside the egocentric view.
**Confidence:** Medium

**Claim:** For multi-agent settings with CTDE, each agent can maintain its own episodic memory and intrinsic reward. Sharing novelty information across agents (so a state becomes "non-novel" for everyone once visited by k different agents) can improve coordinated exploration and reduce redundancy. [^44^]
**Source:** Extending NGU to Multi-Agent RL; MACE (Jiang et al., AAAI 2024)
**URL:** https://arxiv.org/pdf/2512.01321; https://ojs.aaai.org/index.php/AAAI/article/view/29693/31185
**Date:** 2024-2025
**Excerpt:** "A state embedding becomes 'non-novel' for everyone once visited by k different agents... MACE introduces a novelty-based intrinsic reward and a hindsight-based intrinsic reward to enable coordinated exploration in decentralized cooperative tasks."
**Context:** MACE uses only local novelty communication (one float per agent) plus a hindsight-based coordination reward, making it communication-efficient.
**Confidence:** Medium

---

### 2.10 Key Implementation Considerations

**Claim:** For on-policy PPO, episodic intrinsic rewards are naturally compatible since the episode boundaries are well-defined. The intrinsic reward should be computed per-step and combined with extrinsic reward: r_total = r_extrinsic + β * r_intrinsic, where β may be annealed over training. [^45^]
**Source:** Synthesized from NovelD, E3B, and DEIR implementations
**Confidence:** High

**Claim:** When combining episodic memory with LSTM policies, the episodic memory is separate from the LSTM hidden state. The LSTM maintains temporal information for policy execution, while the episodic memory stores observations/embeddings for novelty computation. Both can coexist without conflict. [^46^]
**Source:** Synthesized from NGU, DEIR architectures
**Confidence:** High

**Claim:** The intrinsic reward coefficient β should typically be annealed from a high value (emphasizing exploration early) to near-zero (emphasizing exploitation later). GoBI uses an exponential decay: β_t = β_0 * exp(-ρ * t), where ρ is tuned per environment. [^47^]
**Source:** Maximizing Episodic Reachability with World Models (Fu et al., ICML 2023)
**URL:** https://proceedings.mlr.press/v202/fu23c/fu23c.pdf
**Date:** 2023
**Excerpt:** "The value of ρ is chosen to make the intrinsic reward large at the beginning of training and near-zero at the end of the training."
**Context:** Typical ρ values range from 5e-8 to 1.5e-6 depending on the environment and training budget.
**Confidence:** High

**Claim:** For vector observations with partial observability, a practical approach is:
1. Use the LSTM hidden state (or a projection of it) as the state representation for novelty computation
2. Apply SimHash or simple distance-based counting to these representations
3. Maintain a small episodic buffer (50-200 entries) per agent
4. Anneal the exploration bonus over training
5. Optionally add a small lifelong novelty component (simple RND on vector obs) [^48^]
**Source:** Synthesized from multiple methods
**Confidence:** Medium

---

### Sources

[^1^]: Savinov, A., Raichuk, A., Marinier, R., Vincent, D., Pollefeys, M., Lillicrap, T., & Gelly, S. (2018). Episodic Curiosity through Reachability. ICLR 2019. https://ar5iv.labs.arxiv.org/html/1810.02274

[^2^]: Savinov et al. (2018). Episodic Curiosity through Reachability. ICLR 2019. https://openreview.net/pdf/a81ed9381b42322b6099ca895beadba6a985c8d9.pdf

[^3^]: Savinov et al. (2018). EC Supplementary Material - S2. https://ar5iv.labs.arxiv.org/html/1810.02274

[^4^]: Savinov et al. (2018). EC Supplementary Material - S7. https://ar5iv.labs.arxiv.org/html/1810.02274

[^5^]: Fu, Y., Peng, R., & Lee, H. (2023). Go Beyond Imagination: Maximizing Episodic Reachability with World Models. ICML 2023. https://proceedings.mlr.press/v202/fu23c/fu23c.pdf

[^6^]: Badia, A.P., Sprechmann, P., Vitvitskyi, A., Guo, D., Piot, B., Kapturowski, S., Tieleman, O., Arjovsky, M., Pritzel, A., Bolt, A., & Blundell, C. (2020). Never Give Up: Learning Directed Exploration Strategies. ICLR 2020. https://openreview.net/forum?id=Sye57xStvB

[^7^]: Weng, L. (2020). Exploration Strategies in Deep Reinforcement Learning. https://lilianweng.github.io/posts/2020-06-07-exploration-drl/

[^8^]: Badia et al. (2020). Never Give Up. ICLR 2020. https://openreview.net/forum?id=Sye57xStvB

[^9^]: Extending NGU to Multi-Agent RL (2025). https://arxiv.org/pdf/2512.01321

[^10^]: Bellemare, M.G., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS 2016. https://arxiv.org/abs/1606.01868

[^11^]: Tang, H., Houthooft, R., Foote, D., Stooke, A., Xi Chen, Y., Duan, Y., Schulman, J., DeTurck, F., & Abbeel, P. (2017). #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning. ICLR 2017. https://openreview.net/pdf/ee025e0524031667162985b357a4942ab9bb62a4.pdf

[^12^]: Ostrovski, G., Bellemare, M.G., van den Oord, A., & Munos, R. (2017). Count-Based Exploration with Neural Density Models. ICML 2017. https://arxiv.org/abs/1703.01310

[^13^]: Martin, J., Narayanan, S., Everitt, T., & Hutter, M. (2017). Count-Based Exploration in Feature Space for Reinforcement Learning. IJCAI 2017. https://arxiv.org/abs/1706.08090

[^14^]: Zhang, T., Xu, H., Wang, X., Wu, Y., Keutzer, K., Gonzalez, J.E., & Tian, Y. (2021). NovelD: A Simple yet Effective Exploration Criterion. NeurIPS 2021. https://proceedings.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf

[^15^]: Zhang, T., Xu, H., Wang, X., Wu, Y., Keutzer, K., Gonzalez, J.E., & Tian, Y. (2021). BeBold: Exploration Beyond the Boundary of Explored Regions. ICLR 2021. https://arxiv.org/abs/2012.08621

[^16^]: Raileanu, R., & Rocktaschel, T. (2020). RIDE: Rewarding Impact-Driven Exploration. ICLR 2020. Referenced in E3B paper: https://e3bagent.github.io/

[^17^]: Tang et al. (2017). #Exploration. ICLR 2017. https://openreview.net/pdf/ee025e0524031667162985b357a4942ab9bb62a4.pdf

[^18^]: Tang et al. (2017). #Exploration (learned hashing). ICLR 2017.

[^19^]: Wayne, G. et al. (2018). Unsupervised Predictive Memory in a Goal-Directed Agent. Nature. https://web.stanford.edu/class/cs379c/class_messages_listing/curriculum/Annotated_Readings/WayneetalCoRR-18_Annotated.pdf

[^20^]: Mezghani, L. et al. (2021). Memory-Augmented Reinforcement Learning for Image-Goal Navigation. https://arxiv.org/abs/2101.05181

[^21^]: Henaff, M., Raileanu, R., Jiang, M., & Rocktaschel, T. (2022). Exploration via Elliptical Episodic Bonuses. NeurIPS 2022. https://arxiv.org/abs/2210.05805

[^22^]: Henaff et al. (2022). E3B. https://github.com/facebookresearch/e3b

[^23^]: Henaff et al. (2022). E3B. https://discovery.ucl.ac.uk/id/eprint/10173689/1/2954_exploration_via_elliptical_epi.pdf

[^24^]: Fu et al. (2023). GoBI. ICML 2023. https://proceedings.mlr.press/v202/fu23c.html

[^25^]: Fu et al. (2023). GoBI (reward formula). https://proceedings.mlr.press/v202/fu23c/fu23c.pdf

[^26^]: Episodic Novelty Through Temporal Distance (2025). https://arxiv.org/html/2501.15418v1

[^27^]: Noise-Robust Exploration Via Learning Progress Monitoring (2025). https://arxiv.org/html/2509.25438v1

[^28^]: Savinov et al. (2018). EC (couch potato problem). https://techxplore.com/news/2018-10-method-instill-curiosity-agents.html

[^29^]: Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). Exploration by Random Network Distillation. https://arxiv.org/abs/1810.12894

[^30^]: Tang et al. (2017). #Exploration. ICLR 2017.

[^31^]: Henaff et al. (2022). E3B. NeurIPS 2022.

[^32^]: Savinov et al. (2018). EC Supplementary S7.

[^33^]: Badia et al. (2020). Never Give Up. ICLR 2020.

[^34^]: Fu et al. (2023). GoBI (wallclock time). https://proceedings.mlr.press/v202/fu23c/fu23c.pdf

[^35^]: Ostrovski et al. (2017). Count-Based Exploration with Neural Density Models. ICML 2017.

[^36^]: Tang et al. (2017). #Exploration. ICLR 2017.

[^37^]: Zhang et al. (2021). BeBold. ICLR 2021.

[^38^]: Henaff et al. (2022). E3B. NeurIPS 2022.

[^39^]: Badia et al. (2020). Never Give Up. ICLR 2020.

[^40^]: Savinov et al. (2018). Episodic Curiosity through Reachability. ICLR 2019.

[^41^]: Fu et al. (2023). GoBI. ICML 2023.

[^42^]: Synthesized from EC supplementary S5.2 and E3B paper.

[^43^]: Synthesized from analysis of episodic methods for partial observability.

[^44^]: Extending NGU to Multi-Agent RL (2025); Jiang, H., Ding, Z., & Lu, Z. (2024). MACE. AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/29693/31185

[^45^]: NovelD, E3B, DEIR implementations.

[^46^]: NGU and DEIR architectures.

[^47^]: Fu et al. (2023). GoBI. ICML 2023.

[^48^]: Synthesized recommendation.
