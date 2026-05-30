## 3. Observation and Action Space Design

The design of observation and action spaces is the single most consequential architectural decision for a multi-agent village-building system. These choices determine sample efficiency, generalization, training stability, and achievable task complexity. This chapter taxonomizes the design space across 14 major Minecraft AI projects, derives quantitative tradeoffs, and produces role-specific schemas for the four village roles: gatherer, builder, farmer, and defender.

### 3.1 Observation Space Taxonomy

Observation spaces for Minecraft embodied agents fall into four categories, ordered by increasing semantic structure: raw pixels, symbolic state, hybrid multimodal, and graph-based representations. Each category carries distinct implications for training efficiency and task capability.

#### 3.1.1 Raw Pixel Observations: VPT versus MineRL

The raw pixel approach feeds the agent egocentric RGB frames from the first-person camera — the most human-like modality and the most sample-inefficient.

**VPT (Video PreTraining)**, developed by OpenAI in 2022, renders at 640×360 and downsamples to **128×128** for model input at 20 Hz [^20^]. The observation retains the full HUD — hotbar, health indicators, hand animation, and a rendered mouse cursor — enabling the model to learn GUI interaction directly. Its transformer policy encodes temporal context without explicit frame stacking. The critical weakness is sample complexity: VPT required **2,000+ hours of contractor gameplay data** for pre-training [^20^].

**MineRL** uses a more conservative **64×64×3** resolution, supplementing RGB with a `compass.angle` scalar and an `inventory` dictionary [^15^]. The lower resolution halves computational overhead but renders GUI text and distant blocks indistinguishable, forcing reliance on symbolic side channels.

The tradeoff between these resolutions is summarized in Table 1.

| Aspect | MineRL (64×64) | VPT (128×128) | Implication for Village-Building |
|---|---|---|---|
| Resolution | 64×64×3 | 128×128×3 | VPT preserves GUI readability; MineRL does not |
| Frame rate | 20 Hz | 20 Hz | Both require 20 inferences/second |
| HUD overlays | Stripped | Retained | VPT agents learn to read HUD; MineRL relies on side channels |
| FOV | 70° | 70° | Standard Minecraft default |
| Frame stacking | 1–4 frames | Single frame + transformer history | VPT's approach is more parameter-efficient |
| Pre-training data | None required | 2,000+ hours contractor video | VPT's data requirement is prohibitive for most teams |
| Sample efficiency | Very low | Low | Neither is viable for multi-agent training without augmentation |
| Block-level precision | Insufficient | Marginal | Neither resolution supports precise placement without symbolic aid |

*Table 1: Raw pixel observation characteristics. Pure pixel observations are viable only when combined with symbolic augmentation for construction tasks requiring block-level precision.*

Pure pixel observations are unsuitable as the sole input modality for village-building. Construction tasks demand sub-block placement precision, and the 128×128 grid does not reliably resolve individual block faces at distances beyond 5–6 meters. The strategic value of pixel data lies in pattern recognition — structure aesthetics, terrain classification, and mob detection — which symbolic representations cannot easily encode.

#### 3.1.2 Symbolic Observations: LiDAR, Voxels, and Inventory

Symbolic observations replace the visual rendering with programmatically accessible world state. This approach achieves 10–100× faster training by eliminating the need for convolutional feature extraction [^173^].

**GITM (Ghost in the Minecraft)** represents the pure symbolic paradigm. Its observation space contains seven components and **no RGB pixels whatsoever**: LiDAR rays emitted at 5-degree horizontal and vertical intervals for object localization; a 3×3×3 voxel grid of surrounding blocks (expanded to a 10-unit radius for navigation tasks); full inventory with item types and counts; life statistics (health, hunger, oxygen); GPS coordinates (x, y, z); current biome type; and ground status (on or under ground) [^173^]. The GITM authors explicitly note that "RGB is not used in our implementation, although it provides more information than LiDAR rays" — a deliberate efficiency tradeoff that enabled them to claim "10,000× more efficiency than prior RL methods" [^173^].

**Voyager** renders observations as structured text prompts to its LLM: biome, time, nearby blocks and entities within 32 blocks, health, hunger, position, equipment, inventory, and chest contents [^31^]. This representation is semantically dense but variable-length, requiring the LLM to parse each field from raw text.

**Craftax** provides the most compact formal symbolic space: a 9×11 grid with one-hot encodings for 37 block types, 5 item types, and 36 creature types, plus a normalized inventory vector. The total flat observation is **8,268 dimensions** [^113^]. Its JAX backend enables billion-step training runs on consumer hardware — throughput impossible with pixel-based environments.

Symbolic observations excel at training speed but sacrifice visual generalization: agents cannot leverage pre-trained visual representations from ImageNet or CLIP, nor interpret decorative patterns critical for aesthetically coherent village construction.

#### 3.1.3 Hybrid Observations: Combining Pixels with Structured Metadata

Hybrid observation spaces combine RGB frames with symbolic metadata, achieving the best of both worlds: pre-trained visual representations for pattern recognition and precise symbolic data for state-aware decision-making.

**JARVIS-1** was the first major system to demonstrate this approach at scale. It feeds first-person RGB frames into a MineCLIP visual encoder alongside textual task instructions, storing past observations in a multimodal memory module [^63^]. Goal-conditioned controllers dispatch plans to low-level action heads, enabling the system to execute over 200 tasks ranging from short-horizon tree-chopping to long-horizon diamond-pickaxe acquisition. The critical architectural insight is separation of concerns: the visual encoder handles perception, the LLM handles planning, and the memory module bridges the two with retrieval-augmented context [^63^].

**Optimus-3** (2025) extends this design using **Qwen2.5-VL-7B** with a Mixture-of-Experts (MoE) architecture: a shared knowledge expert plus five task-specific experts (planning, perception, action, grounding, reflection) [^5^]. The model consumes RGB at 128×128 plus natural language task descriptions and outputs low-level controls at 20 Hz. Training required 230,000 SFT samples, 58,000 multimodal reasoning samples, and 5,000 RL samples — a data scale reflecting the cost of hybrid approaches [^5^].

**MineDojo** provides the most comprehensive formal definition of a hybrid observation space [^134^]. Its unified interface exposes ten modalities: RGB (3, H, W), equipment state (6,), inventory (36,), surrounding voxel grid (3, 3, 3), life statistics, GPS coordinates (3,), compass heading (2,), crafting proximity sensors (2,), damage source information, and configurable LiDAR raycasts. MineDojo's design philosophy is "provide everything and let the agent learn what to attend to" — a generous approach that enables flexible research but increases the burden on the learning algorithm to perform feature selection.

For village-building, hybrid observation is strongly recommended. Construction requires spatial precision (symbolic voxel data) and pattern recognition for aesthetics and terrain (pixel data). The key design question is how to weight and combine modalities per role.

#### 3.1.4 World State and Graph-Based Observations

Beyond pixels and symbols, the most semantically rich observation modality models the world as a graph of entities and relationships. This representation is essential for multi-agent coordination because it explicitly encodes task dependencies, spatial partitions, and inter-agent relationships.

**VillagerAgent** introduced a Directed Acyclic Graph (DAG) representation for multi-agent construction, where each node is a subtask with assigned agents and completion status, and edges encode dependency constraints [^100^]. For example, `build_house` depends on both `craft_planks` and `clear_site` being completed first, enabling automatic scheduling and deadlock detection.

**Scene graphs** extend this to the physical environment: entities (agents, mobs, crops, chests) are nodes, and edges encode relationships (`adjacent_to`, `threatens`, `in_farm_plot`). Spatial partitions define functional regions — farm plots, build sites, defensive perimeters — as labeled bounding boxes. Scene graphs directly encode structural constraints that pixel or symbolic representations can only recover through inference.

**Block adjacency matrices** encode which blocks touch which faces within a construction zone, enabling structural integrity checks and scaffolding generation. Server-side world state, accessible through Mineflayer or Fabric server hooks [^93^], should feed the critic during centralized training but remain unavailable during decentralized execution to maintain realistic partial observability.

### 3.2 Action Space Taxonomy

Action spaces in Minecraft span four levels of abstraction, from individual keypresses to full programs. The choice of action granularity directly determines sample efficiency, temporal horizon capability, and the engineering complexity of the execution layer.

#### 3.2.1 Low-Level Keypresses: MineRL Discrete and Continuous Space

The lowest-level action space maps directly to keyboard and mouse inputs. MineRL defines a 10-dimensional action: binary flags for `attack`, `back`, `forward`, `jump`, `left`, `right`, `sneak`, and `sprint`; a discrete `place` action selecting an item type; and a continuous 2D `camera` action controlling pitch and yaw in degrees [^15^]. VPT uses an equivalent space at 128×128 resolution, outputting discretized actions via an autoregressive transformer policy at 20 Hz [^20^].

Low-level action spaces maximize human-likeness but suffer from extreme sample complexity. Placing a single block requires dozens of coordinated actions: camera movement, positioning, hotbar selection, aiming, and right-clicking. A 50-block wall requires thousands of timesteps. For multi-agent construction with coordinated placement across large structures, training from low-level actions is infeasible without massive pre-training data.

#### 3.2.2 High-Level Skills: Voyager's Code-as-Policy

Voyager's foundational innovation treats **generated code as the action space** [^14^]. Instead of outputting keypresses, the LLM writes JavaScript functions that call Mineflayer APIs. A single action — a complete JavaScript function — can execute for hundreds of timesteps.

The Voyager skill library exposes primitives such as `bot.pathfinder.goto(goal)` for navigation, `bot.dig(block)` for block breaking, `bot.placeBlock(referenceBlock, faceVector)` for placement, and `bot.craft(recipe, count, craftingTable)` for crafting [^22^]. Compositional skills build upon these primitives: `mineWoodLog` calls `findBlock`, `pathfinder.goto`, and `bot.dig` in sequence. Voyager's automatic curriculum generates increasingly complex tasks, and successful skills are stored in a vector-indexed library for future retrieval [^14^].

**ODYSSEY** extended this paradigm to **40 primitive skills** and **183 compositional skills** with recursive prerequisite resolution [^13^]. Code-as-policy offers unmatched expressivity, but it is fundamentally LLM-driven — it cannot be directly optimized through gradient-based RL without significant architectural adaptation.

#### 3.2.3 Structured Primitives: GITM's Nine Parameterized Actions

GITM defines a compact vocabulary of **nine structured actions**, each parameterized with typed arguments [^173^]:

- `equip(item_name, slot)` — Equip an item from inventory
- `explore(direction)` — Discover resources in the environment
- `approach(target_type, target_id)` — Navigate to a target entity or block
- `mine(target, tool)` — Break a block or attack an entity
- `dig_down(layers)` — Excavate downward
- `go_up(height)` — Ascend by placing a pillar
- `build(block_type, position_list)` — Place blocks following a pattern
- `craft(recipe, count)` — Manufacture items from raw materials
- `apply(item, target)` — Use an item on a target (e.g., bonemeal on crops)

Each action is backed by a hand-scripted execution module that handles the low-level motor control: pathfinding, camera alignment, and keypress sequences. This separation — high-level symbolic commands with deterministic low-level execution — achieves reliable task completion without requiring the planner to reason about pixel coordinates or key timings. The tradeoff is reduced flexibility: if a scripted module fails on an edge case (e.g., placing a block on a diagonal surface), the planner cannot adapt the execution strategy.

#### 3.2.4 Hierarchical Actions: Plan4MC's Skill Tiers

**Plan4MC** organizes actions into three skill tiers that mirror the structure of Minecraft gameplay itself [^72^]. *Finding skills* (`explore`, `find_tree`, `find_cave`, `find_animal`) handle spatial search. *Manipulation skills* (`chop_tree`, `mine_stone`, `attack_mob`, `place_block`) handle physical interaction. *Crafting skills* (`craft_planks`, `craft_sticks`, `craft_pickaxe`, `smelt_ore`) handle resource transformation. An LLM constructs a skill dependency graph (skill graph), and a search algorithm walks the graph to find executable sequences. Plan4MC's hierarchy enables systematic decomposition: building a house requires finding wood (finding), chopping it (manipulation), crafting planks (crafting), and placing blocks (manipulation), with the LLM planner determining the correct ordering.

**DEPS (Describe, Explain, Plan, Select)** adds interactive replanning: on subtask failure, a Descriptor summarizes state, an Explainer LLM diagnoses the cause, a Planner revises the plan, and a Selector prioritizes subtasks by accessibility. DEPS improved Minecraft success rates by **52.74%** over baseline planning [^74^].

This hierarchy reveals a fundamental principle: **temporal abstraction and sample efficiency are inversely correlated with action space cardinality**. A single Voyager skill function replaces thousands of low-level timesteps, compressing effective episode length by orders of magnitude.

### 3.3 Comparison Matrix

Table 2 provides a side-by-side comparison of observation and action space choices across ten representative projects, spanning the full spectrum from pure pixel-based RL to pure LLM-driven planning.

| Project | Year | Obs. Type | Resolution / Size | Action Type | # Actions | Sample Efficiency | Generalization |
|---|---|---|---|---|---|---|---|
| MineRL | 2019 | Pixel | 64×64×3 | Low-level keypress | ~15 | Very low | Poor |
| VPT | 2022 | Pixel | 128×128×3 | Low-level keypress | ~15 | Low | Poor |
| MineDojo | 2022 | Hybrid | 128×128 + 8268-dim | Discrete | 24 | High (with symbolic) | Moderate |
| Craftax | 2024 | Symbolic | 8268-dim flat | Discrete | 20+ | Very high | Good |
| GITM | 2023 | Symbolic | ~200 dims (LiDAR+voxel) | Structured primitives | 9 | High | Good |
| Voyager | 2023 | Symbolic (text) | Variable | Code-as-policy | Infinite | Very high (LLM) | Excellent |
| Plan4MC | 2023 | Hybrid | 128×128 + symbolic | Hierarchical skills | ~30 | Medium | Good |
| JARVIS-1 | 2023 | Hybrid | 128×128 + text | Code-like functions | Variable | Medium-high | Very good |
| TeamCraft | 2024 | Hybrid | 640×480 + inventory | Parameterized skills | 8 | Medium | Good |
| Optimus-3 | 2025 | Hybrid (VLM) | 128×128 | Low-level + MoE | Variable | Medium | Very good |

*Table 2: Cross-project comparison of observation and action space designs. Projects are ordered by year of publication. Sample efficiency ratings reflect the empirical training speed reported in each project's evaluation.*

#### 3.3.1 Performance Implications

The empirical performance data in Table 3, drawn from the OpenHA benchmark suite [^111^], quantifies the consequences of these design choices. OpenHA provides a standardized evaluation across embodied action success rate (ASR), GUI interaction ASR, and combat ASR — three capabilities essential for village-building.

| Method | Embodied ASR | GUI ASR | Combat ASR | Inference FPS | Source |
|---|---|---|---|---|---|
| VPT | 6.0% | 0.8% | 3.6% | N/A | [^111^] |
| STEVE-1 | 8.0% | 3.2% | 3.9% | N/A | [^111^] |
| JARVIS-VLA | 30.0% | 25.1% | 18.5% | N/A | [^111^] |
| GroundingHA | 37.1% | 6.7% | 26.5% | 5.61 | [^111^] |
| OpenHA (universal) | 30.1% | 32.5% | 31.9% | 1.36 | [^111^] |

*Table 3: Standardized performance comparison from the OpenHA benchmark (2025). ASR = Action Success Rate. Higher values indicate better performance on the benchmark task suite.*

Three conclusions emerge. First, **hybrid approaches outperform pure methods**: JARVIS-VLA at 30.0% embodied ASR is 5× better than VPT's 6.0%, attributable to its multimodal memory and goal-conditioned controller [^63^]. Second, **universal training across action spaces produces positive transfer**: OpenHA's universal model achieves the highest GUI ASR (32.5%) despite diverse training, showing that exposure to multiple action representations improves policy learning [^111^]. Third, **inference speed is a binding constraint**: OpenHA universal runs at 1.36 FPS due to VLM processing overhead [^111^], implying that VLM perception must be cached, distilled, or reserved for planning rather than per-tick processing.

The action space comparison reveals a clear efficiency ordering. Low-level keypresses (MineRL, VPT) require prohibitive samples — VPT's 2,000 hours of contractor data is beyond most teams [^20^]. Code-as-policy (Voyager) and structured primitives (GITM) both achieve high efficiency through different mechanisms: function-call composability [^14^] versus deterministic execution [^173^]. Hierarchical skills (Plan4MC) offer systematic decomposition with moderate requirements [^72^]. For village-building, Plan4MC's finding-manipulation-crafting tiers provide the most transferable template.

### 3.4 Recommended Design for 4-Role Village

Drawing on the taxonomy and comparison data above, this section defines role-specific observation and action schemas for the four village roles. The design follows three principles: (1) each role receives only observation modalities relevant to its function, reducing cognitive load and improving sample efficiency; (2) all roles share a common core observation plus a role-specific overlay, enabling parameter sharing during joint training; and (3) actions are parameterized skill primitives that execute for multiple timesteps, not low-level keypresses.

#### 3.4.1 Role-Specific Observation Schemas

Table 4 specifies the observation components for each role, identifying which modalities are shared across all roles and which are role-specific.

| Component | Gatherer | Builder | Farmer | Defender | Rationale |
|---|---|---|---|---|---|
| RGB patch (128×128) | Yes, FOV 90° | Yes, FOV 70° | Yes, FOV 80° | Yes, FOV 100° | FOV tailored to task: wider for scouting, narrower for precision |
| Agent state (pos, health, hunger) | Yes | Yes | Yes | Yes + armor rating | Shared core; defender needs armor data |
| Inventory | Yes | Yes (building mats) | Yes (seeds/produce) | Yes (weapons/ammo) | Content organization differs by role |
| Communication buffer | Yes | Yes | Yes | Yes | All roles share team messages |
| **Role-specific overlay** | | | | | |
| Resource proximity map | Yes | No | No | No | Ores, trees, surface resources within 64 blocks |
| Voxel grid (11³) | No | Yes | No | No | Block-level precision for placement |
| Construction plan / blueprint | No | Yes | No | No | Target blocks with placed/pending status |
| Crop and livestock state | No | No | Yes | No | Growth stage, hydration, breeding readiness |
| Threat assessment | No | No | No | Yes | Hostile entity tracking with time-to-village estimates |
| Defensive structure state | No | No | No | Yes | Wall integrity, trap status, watchtower occupancy |
| Friendly positions | No | No | No | Yes | Coordinates and health of all teammate agents |
| Farm plot state | No | No | Yes | No | Tillage, hydration, crop type per plot |

*Table 4: Role-specific observation components. "Yes" indicates the modality is present in that role's observation space. Shared components enable parameter sharing during joint training; role-specific overlays provide task-relevant precision.*

The **gatherer** receives a resource proximity map encoding nearby ores, trees, and surface resources with type, position, distance, and quantity estimates — analogous to GITM's LiDAR but with resource-type semantic filtering [^173^]. A resource richness score within a 64-block radius supports exploration-versus-exploitation decisions.

The **builder** receives an 11×11×11 voxel grid (each cell contains a block type ID) for precise placement decisions, plus a construction plan overlay specifying target blocks with placed/pending status and progress fraction. This combination enables real-time structural deviation detection.

The **farmer** receives crop plot state (growth stage, hydration, pending actions) and livestock state (animal type, pen location, breeding readiness). A temporal context module provides day number and estimated growth ticks remaining for harvest scheduling.

The **defender** receives a threat assessment module tracking hostiles within 64 blocks with type, position, time-to-village estimates, and priority scores (1–5). A defensive perimeter tracker reports wall section status, and friendly position tracking enables intercept path calculation.

#### 3.4.2 Role-Specific Action Schemas

The action space follows a three-tier hierarchy: an LLM-based task decomposer manages the construction DAG [^100^]; each role's policy selects parameterized skill primitives; and hand-scripted motor modules translate parameters into low-level API calls. This mirrors GITM's structured primitives [^173^] and Plan4MC's skill tiers [^72^], adapted for multi-agent coordination.

```python
# ============================================================
# UNIFIED OBSERVATION SCHEMA — shared core + role overlays
# ============================================================
{
    "agent_id": str,
    "role": "gatherer" | "builder" | "farmer" | "defender",
    "tick": int,

    "visual": {
        "rgb": ndarray[128, 128, 3],
        "fov": int,                           # 70-100, role-dependent
        "yaw_pitch": [float, float],
    },

    "agent_state": {
        "position": [float, float, float],
        "health": int,                        # 0-20
        "hunger": int,
        "experience_level": int,
    },

    "inventory": {
        "slots": list[{"item": str, "count": int, "durability": int|null}],
        "slots_used": int,                    # 0-36
        "slots_free": int,
    },

    "communication_buffer": list[{
        "sender_id": str,
        "sender_role": str,
        "timestamp": int,
        "message_type": "state_update" | "request_help" | "task_offer" | "threat_alert" | "coordination",
        "payload": dict,
    }],

    "assigned_task": {
        "task_id": str,
        "task_description": str,
        "priority": int,                      # 1 (critical) to 5 (low)
        "deadline_tick": int|null,
    },

    # Role-specific overlay selected by "role" field
    "role_overlay": {
        "gatherer": {
            "resource_map": list[{
                "resource_type": str,
                "position": [float, float, float],
                "distance": float,
                "quantity_estimate": int,
            }],
            "resource_richness_score": float,   # Aggregate in 64-block radius
            "nearby_hostiles": list[{"type": str, "distance": float}],
        },

        "builder": {
            "voxel_grid": {
                "shape": [11, 11, 11],
                "data": ndarray[11, 11, 11],      # Block type IDs, -1 = unknown
                "origin_offset": [int, int, int],
            },
            "construction_plan": {
                "blueprint_id": str,
                "target_blocks": list[{
                    "position": [int, int, int],
                    "block_type": str,
                    "status": "pending" | "placed" | "incorrect",
                }],
                "progress_fraction": float,
            },
            "materials_needed": dict[str, int],
        },

        "farmer": {
            "crop_plots": list[{
                "position": [int, int, int],
                "crop_type": str,
                "growth_stage": int,              # 0-7 for wheat
                "hydration": float,               # 0.0-1.0
                "needs_action": str|null,         # "water", "harvest", "plant"
            }],
            "livestock": list[{
                "animal_type": str,
                "position": [float, float, float],
                "count": int,
                "can_breed": bool,
            }],
            "day_number": int,
        },

        "defender": {
            "threat_assessment": {
                "hostile_entities": list[{
                    "type": str,
                    "position": [float, float, float],
                    "distance": float,
                    "heading_toward_village": bool,
                    "time_to_village": float,
                    "priority": int,              # 1 (immediate) to 5 (monitor)
                }],
                "threat_level": float,            # 0.0-1.0 aggregate
            },
            "defensive_perimeter": list[{
                "section_id": str,
                "status": "secure" | "breached" | "at_risk",
            }],
            "friendly_positions": list[{
                "agent_id": str,
                "role": str,
                "position": [float, float, float],
                "health": int,
            }],
            "armor_rating": int,
        },
    }
}
```

The action schema below defines the parameterized skill primitives available to each role. Each action executes for an extended temporal duration (10–500+ environment steps) through a scripted motor module, abstracting away low-level motor control.

```python
# ============================================================
# ROLE-SPECIFIC ACTION SCHEMA (parameterized skill primitives)
# ============================================================

# --- GATHERER ACTIONS ---
{
    "action_type": "gatherer_navigate",
    "parameters": {
        "target": {"type": "coordinates", "position": [x, y, z]}
                  | {"type": "resource", "resource_type": str, "max_distance": int},
        "pathfinding_mode": "direct" | "safe" | "resource_optimized",
    },
    "expected_duration_steps": int,           # Motor module estimate
}

{
    "action_type": "gatherer_harvest",
    "parameters": {
        "resource_type": str,                 # "oak_log", "iron_ore", "wheat", etc.
        "quantity": int,
        "collection_mode": "mine_nearest" | "strip_mine" | "selective",
        "tool": str|null,                     # Preferred tool; null = auto-select
    },
    "expected_duration_steps": int,
}

{
    "action_type": "gatherer_deposit",
    "parameters": {
        "chest_position": [x, y, z],
        "items": {str: int},                  # item_name: quantity
    },
    "expected_duration_steps": int,
}

# --- BUILDER ACTIONS ---
{
    "action_type": "builder_place_block",
    "parameters": {
        "block_type": str,
        "position": [x, y, z],
        "orientation": "normal" | "upside_down" | "east" | "west" | null,
    },
    "expected_duration_steps": int,
}

{
    "action_type": "builder_place_sequence",
    "parameters": {
        "block_type": str,
        "positions": list[[x, y, z]],         # Batch placement (walls, floors)
        "material_substitutions": {str: str}, # e.g., {"oak_planks": "spruce_planks"}
    },
    "expected_duration_steps": int,
}

{
    "action_type": "builder_follow_blueprint",
    "parameters": {
        "blueprint_id": str,
        "anchor_position": [x, y, z],
        "layers": "all" | [int, int],         # Vertical slice to build
    },
    "expected_duration_steps": int,
}

{
    "action_type": "builder_fetch_materials",
    "parameters": {
        "from_chest": [x, y, z],
        "materials": {str: int},
    },
    "expected_duration_steps": int,
}

# --- FARMER ACTIONS ---
{
    "action_type": "farmer_till",
    "parameters": {
        "positions": list[[x, y, z]],
    },
    "expected_duration_steps": int,
}

{
    "action_type": "farmer_plant",
    "parameters": {
        "crop_type": str,
        "positions": list[[x, y, z]],         # Must be tilled farmland
    },
    "expected_duration_steps": int,
}

{
    "action_type": "farmer_harvest",
    "parameters": {
        "positions": list[[x, y, z]],
        "replant": bool,
    },
    "expected_duration_steps": int,
}

{
    "action_type": "farmer_breed",
    "parameters": {
        "animal_type": str,
        "pen_location": [x, y, z],
        "feed_item": str,
    },
    "expected_duration_steps": int,
}

# --- DEFENDER ACTIONS ---
{
    "action_type": "defender_patrol",
    "parameters": {
        "waypoints": list[[x, y, z]],
        "pattern": "circular" | "back_and_forth",
    },
    "expected_duration_steps": int,
}

{
    "action_type": "defender_intercept",
    "parameters": {
        "threat_id": str,                     # Target entity from threat assessment
        "interception_point": [x, y, z],
        "tactic": "melee" | "ranged" | "hit_and_run",
    },
    "expected_duration_steps": int,
}

{
    "action_type": "defender_defend_agent",
    "parameters": {
        "protect_target": str,                # Agent ID to protect
        "formation": "shield" | "escort" | "perimeter",
    },
    "expected_duration_steps": int,
}

{
    "action_type": "defender_build_defense",
    "parameters": {
        "structure_type": "wall" | "watchtower" | "trap" | "moat",
        "position": [x, y, z],
        "material": str,
        "specifications": dict,               # Height, length, etc.
    },
    "expected_duration_steps": int,
}

# --- SHARED COMMUNICATION ACTIONS (all roles) ---
{
    "action_type": "broadcast",
    "parameters": {
        "message_type": "state_update" | "request_help" | "task_offer" | "threat_alert",
        "payload": dict,                      # Role-specific content
        "target_roles": list[str]|null,       # Null = broadcast to all
        "priority": "normal" | "urgent",
    },
    "expected_duration_steps": 1,             # Instant (single tick)
}

{
    "action_type": "wait",
    "parameters": {
        "duration_ticks": int,                # Idle / hold position
    },
    "expected_duration_steps": int,
}
```

Each action is backed by a scripted motor module translating parameters into low-level API calls. For example, `builder_place_sequence` handles pathfinding, hotbar selection, camera orientation, and placement for each position. On sub-step failure (e.g., obstructed block), the module reports a structured error code, enabling the policy to select an alternative action or request assistance.

#### 3.4.3 Communication Channel Design

Multi-agent coordination requires a communication protocol that is expressive enough to convey task-relevant state yet compact enough to process efficiently during RL training. Research from NeurIPS 2024 demonstrates that agents learn optimal communication protocols shaped jointly by RL objectives (what information improves team reward) and supervised regularization (encouraging human-interpretable message patterns) [^36^].

The recommended channel uses a structured schema with **128-dimension message limits** for efficient MARL throughput [^172^]. Each message contains a header (sender ID, role, timestamp, priority), a typed payload, and routing metadata. Four message types cover most coordination: `state_update` shares local observations; `request_help` signals distress; `task_offer` proposes handoffs; and `threat_alert` is a priority-elevated warning from the defender. Range is limited to 32 blocks unless both agents have line of sight to a village-center repeater block, forcing localized coordination and preventing full state sharing.

#### 3.4.4 Dec-POMDP Formalization

The four-role village system is formally a **Decentralized Partially Observable Markov Decision Process** (Dec-POMDP). Each agent observes only a local subset of the global state, takes actions based on its private observation and communication history, and receives a shared team reward. The formal definition follows the CTDE (Centralized Training with Decentralized Execution) paradigm, which is the prevailing architecture for cooperative multi-agent RL [^165^].

During **centralized training**, a global critic receives privileged state: all agent positions, inventories, the construction zone block grid, task DAG progress, and communication logs. It computes $Q_{\text{total}}(s, a_1, \ldots, a_n)$ for credit assignment, mitigating the credit assignment problem in sparse team rewards [^165^]. During **decentralized execution**, each actor $\pi_i(a_i \mid o_i, z_i)$ conditions only on local observation $o_i$ and a learned role embedding $z_i$ — a vector enabling behavioral specialization without separate network parameters per role.

The transition from training to execution requires that role embeddings and communication protocols substitute for the lost privileged state. This risk is mitigated by **privileged feature dropout**: with probability 0.1, the critic receives masked global state, forcing actors to rely more heavily on local observations and messages.

The per-agent loop proceeds as: (1) environment delivers observation $o_i^{(t)}$ and communication buffer; (2) actor selects parameterized skill $a_i^{(t)}$; (3) scripted motor module executes for $\tau$ timesteps; (4) environment returns reward $r_i^{(t)}$ and next observation; (5) agent may emit message $m_i^{(t)}$. Team reward combines village completion, resource efficiency, and survival, with individual shaping rewards (blocks placed, crops harvested, mobs defeated) to improve early-phase credit assignment.
