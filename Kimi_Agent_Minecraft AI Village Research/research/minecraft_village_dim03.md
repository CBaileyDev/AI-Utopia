# Research Dimension 3: Observation and Action Space Design for Minecraft RL

## Executive Summary

- **No single observation or action space is universally optimal** for all Minecraft tasks. The choice is highly contingent on the task domain, agent architecture, and whether the agent uses LLM-based or RL-based decision-making [^111^](https://arxiv.org/html/2509.13347v1).
- **For a multi-agent village-building scenario with 4 specialized roles (gatherer, builder, farmer, defender), a hybrid observation space combining symbolic state + local pixel patches + role-specific overlays is strongly recommended**, based on the success of hybrid designs in JARVIS-1 [^63^](https://arxiv.org/abs/2311.05997), Optimus-3 [^5^](https://arxiv.org/html/2506.10357v1), and TeamCraft [^98^](https://arxiv.org/html/2412.05255v1).
- **For action spaces, structured high-level skill primitives significantly outperform low-level keypresses** in sample efficiency. Voyager's code-as-policy approach [^14^](https://voyager.minedojo.org/) and GITM's structured actions [^173^](https://hal.science/hal-04107105v2/file/GITM_hal.pdf) demonstrate that hierarchical action spaces are essential for long-horizon tasks like village construction.
- **Role-specific observation/action customization dramatically improves coordination**: Builders need voxel grid/block placement data, farmers need crop growth state, gatherers need resource proximity maps, and defenders need entity tracking and threat assessment [^100^](https://arxiv.org/html/2406.05720v1).
- **A graph-based task dependency structure (DAG) combined with communication channels encoding task-relevant state** is the optimal coordination architecture, following VillagerAgent's demonstrated effectiveness for multi-agent construction in Minecraft [^100^](https://arxiv.org/html/2406.05720v1).

---

## 1. Observation Space Designs

### 1.1 Raw Pixel Observations — VPT, MineRL Approach

The raw pixel observation approach uses egocentric RGB frames from the agent's first-person perspective. This is the most human-like observation modality.

**VPT (Video PreTraining) Observation Space** [^20^](https://cdn.openai.com/vpt/Paper.pdf):
- **Rendering**: 640x360 rendered at 20Hz, downsampled to **128x128** for model input
- **Field of View**: 70 degrees (Minecraft default)
- **GUI scale**: Set to 2; brightness set to 2
- **Overlays included**: Hotbar, health indicators, hand animation — all retained (unlike MineRL which strips some)
- **Cursor**: A rendered mouse cursor image overlaid at the correct position when GUI is open
- **Frame stacking**: Single frames fed into the model (the VPT policy uses frame history via its transformer architecture)

**MineRL Observation Space** [^15^](https://minerl.readthedocs.io/en/v0.4.4/environments/):
- **Resolution**: **64x64x3** (much smaller than VPT)
- **Additional modalities per task**:
  - `compass.angle`: scalar heading toward goal (Navigate tasks)
  - `inventory`: dict of item counts (e.g., `{"dirt": 5}`)
  - `vector`: 64-dim obfuscated feature vector (competition track)

**Key characteristics of pixel observations**:
| Aspect | Value | Implications |
|--------|-------|-------------|
| Resolution | 64x64 (MineRL) to 128x128 (VPT) | Lower = faster training; higher = more detail |
| FOV | 70 degrees standard | Affects spatial awareness |
| Frame rate | 20 Hz | High compute cost for RL training |
| Stacking | Typically 1-4 frames | Temporal context for motion/action |
| Privileged info | Usually none in pure pixel setup | Agent must learn to read GUI/HUD |

**Assessment for village-building**: Pure pixel observations are extremely sample-inefficient for structured multi-agent construction. The 128x128 resolution from VPT is the minimum needed to discern GUI elements, but block-level precision for placement tasks requires symbolic augmentation.

### 1.2 Symbolic Observations — GITM Approach

Symbolic observations extract structured, programmatically accessible world state rather than raw pixels. This approach achieves dramatically higher sample efficiency.

**GITM (Ghost in the Minecraft) Observation Space** [^173^](https://hal.science/hal-04107105v2/file/GITM_hal.pdf):
- **LiDAR rays**: Rays at 5-degree intervals in horizontal and vertical directions for locating objects
- **Voxel grid**: 3x3x3 surrounding blocks (from MineDojo, expanded to 10-unit radius for navigation)
- **Inventory**: Item types and counts
- **Life statistics**: Health, hunger, oxygen
- **Agent location**: GPS coordinates (x, y, z)
- **Biome**: Current biome type
- **Ground status**: Whether agent is on/under ground
- **NO RGB pixels**: "RGB is not used in our implementation, although it provides more information than LiDAR rays"

**Voyager's Symbolic Observation** [^31^](https://arxiv.org/pdf/2305.16291):
```python
# Voyager observation structure (textual prompt to LLM)
observation = {
    "biome": "plains",                    # e.g., 'plains', 'forest', 'desert'
    "time": "day",                        # 'sunrise', 'day', 'noon', 'sunset', 'night', 'midnight'
    "nearby_blocks": ["dirt", "stone", "grass_block", "oak_log"],  # within 32-block radius
    "other_blocks_recently_seen": ["water", "sand"],
    "nearby_entities": ["pig", "sheep", "zombie"],  # nearest to farthest within 32 blocks
    "health": 20,                         # max 20
    "hunger": 15,                         # max 20
    "position": (100, 64, -200),          # (x, y, z)
    "equipment": ["iron_pickaxe"],        # currently equipped
    "inventory": {                        # item dict with counts (xx/36 slots)
        "oak_log": 12,
        "cobblestone": 32,
        "iron_ingot": 5,
    },
    "chests": {
        "chest_at_(105,64,-195)": "Unknown",  # content unknown until opened
        "chest_at_(98,64,-210)": {"wheat": 8, "bread": 3}
    }
}
```

**Craftax/Crafter Symbolic Observation** [^113^](https://arxiv.org/html/2402.16801v1):
- **Grid view**: 9x11 window of the world around the agent
- **Per-cell encoding**: One-hot for block type (37 types), item type (5 types), creature type (36 types), plus light level
- **Inventory/stats vector**: Normalized counts of all items, health, hunger, etc.
- **Total flat observation size**: 8268 dimensions for Craftax, 1345 for Craftax-Classic

**Key characteristics of symbolic observations**:
| Aspect | Value | Implications |
|--------|-------|-------------|
| Information density | Very high | Every element is semantically meaningful |
| Training speed | 10-100x faster than pixels | No convnet needed; direct MLP processing |
| Human-likeness | Low | Humans use vision, not structured data |
| Generalization | Good within symbolic space | Cannot leverage visual priors from pre-training |
| Partial observability | Explicit and controllable | Can tune what information is included |

### 1.3 Hybrid Observations — JARVIS-1, Optimus-3

Hybrid observations combine pixel inputs with symbolic metadata, achieving the best of both worlds: leveraging pre-trained visual representations while maintaining precise state awareness.

**JARVIS-1 Observation Space** [^63^](https://arxiv.org/abs/2311.05997):
- **Multimodal input**: Visual observations (RGB frames) + textual instructions
- **Memory-augmented**: Multimodal memory stores past observations and experiences
- **Goal-conditioning**: Plans are dispatched to goal-conditioned controllers
- **Over 200 tasks**: From "chopping trees" (short-horizon) to "obtaining diamond pickaxe" (long-horizon)

**Optimus-3 Observation Space** [^5^](https://arxiv.org/html/2506.10357v1):
- Built on **Qwen2.5-VL-7B** multimodal foundation model
- **Multimodal reasoning**: Combines vision with language planning
- **Input**: First-person RGB + task description in natural language
- **Output**: Mouse and keyboard control actions at 20Hz
- **MoE architecture**: Shared knowledge expert + 5 task-specific experts (planning, perception, action, grounding, reflection)
- **Dataset**: 230k SFT samples, 58k multimodal reasoning samples, 5k RL samples

**MineDojo Unified Observation Space** [^134^](https://ar5iv.labs.arxiv.org/html/2206.08853):
| Modality | Shape | Description |
|----------|-------|-------------|
| RGB | (3, H, W) | Egocentric RGB frames |
| Equipment | (6,) | Names, quantities, variants, durabilities |
| Inventory | (36,) | Names, quantities, variants, durabilities |
| Voxel | (3, 3, 3) | Surrounding block types, variants, properties |
| Life stats | (1,) | Health, oxygen, food saturation |
| GPS | (3,) | Agent position |
| Compass | (2,) | Yaw and pitch |
| Nearby tools | (2,) | Crafting table / furnace proximity |
| Damage source | (1,) | Information about incoming damage |
| LIDAR | (N,) | Ground-truth block type along raycasts |

**Assessment for village-building**: Hybrid observation is the **strongly recommended approach**. Village construction requires both spatial precision (which symbolic voxel data provides) and pattern recognition for structure aesthetics (which pixel data enables).

### 1.4 VLM-Processed Observations

When a vision-language model processes the agent's view, it creates a semantically rich representation that bridges pixel and symbolic spaces.

**VLM-as-Observation-Encoder Approaches**:

1. **Optimus-2 (GOAP)** [^66^](https://arxiv.org/html/2502.19902v1):
   - Uses **DeepSeek-VL-1.3B** as the base MLLM
   - **Two-phase training**: (1) Behavior pre-training aligns language with observation-action sequences; (2) Action fine-tuning transforms language space into action space
   - The VLM processes RGB frames into goal-observation-action conditioned policy
   - Planner uses **GPT-4V** for planning and reflection
   - Achieves 0.99 success rate on wood tasks, 0.53 on iron tasks

2. **VLM Promptable Representations for RL** [^73^](https://openreview.net/forum?id=vQDKYYuqWA):
   - Uses VLM embeddings as **promptable representations** for embodied RL
   - Policies trained on VLM embeddings outperform policies on generic image embeddings
   - **Chain-of-thought prompting** produces representations of common-sense semantic reasoning
   - Performance in novel scenes improves by **1.5x** vs. non-promptable embeddings

3. **RL-GPT Integration** [^168^](https://proceedings.neurips.cc/paper_files/paper/2024/file/31f119089f702e48ecfd138c1bc82c4a-Paper-Conference.pdf):
   - LLM generates high-level coded actions for RL training
   - VLM processes observations to generate reward signals (CLIP-based)
   - Action space contains RL-learned skills + LLM-generated code actions

**Schema example for VLM-processed observation**:
```python
{
    "raw_rgb": "<base64-encoded 128x128 image>",
    "vlm_caption": "Agent is facing a flat grassy area with oak trees nearby. "
                   "A partially built wooden house is visible to the left. "
                   "The sky is clear and it is daytime.",
    "vlm_detected_objects": [
        {"class": "oak_log", "bbox": [45, 80, 60, 95], "distance_estimate": "5m"},
        {"class": "grass_block", "bbox": [0, 90, 128, 128]},
        {"class": "wooden_planks", "bbox": [20, 70, 40, 85], "label": "partial_structure"}
    ],
    "vlm_scene_assessment": "suitable_for_construction",
    "embedding": <768-dim vector from VLM hidden layer>,
}
```

### 1.5 World State Observations — Server-Side World Data

Server-side world observations provide privileged access to the true state of the Minecraft world, bypassing the agent's limited first-person view.

**Mineflayer API-Based World State** [^93^](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md):
```python
# Available programmatic state via Mineflayer
world_state = {
    # Agent state
    "position": bot.entity.position,           # Vec3(x, y, z)
    "velocity": bot.entity.velocity,
    "health": bot.health,
    "food": bot.food,
    "inventory": bot.inventory.items(),        # Full inventory with metadata
    "equipment": bot.entity.equipment,         # Currently equipped items

    # World state queries
    "block_at": bot.blockAt(position),         # Block type at any loaded coordinate
    "nearby_entities": bot.nearbyEntities(),   # Entities within loaded chunks
    "players": bot.players,                     # All online players/bots

    # Event-based updates
    "block_updates": [                          # Fires on "blockUpdate" event
        {"pos": (x, y, z), "old": old_block, "new": new_block}
    ],
    "entity_updates": [                         # Fires on "entityUpdate" event
        {"id": entity.id, "position": pos, "type": entity.type}
    ],
}
```

**MineDojo Privileged Observations** [^134^](https://ar5iv.labs.arxiv.org/html/2206.08853):
- **LIDAR sensor**: Returns ground-truth block types along configurable raycast directions
- Considered privileged — not used in benchmarking but useful for reward shaping
- Ray count and directions are configurable

**Key considerations for world state observations**:
| Aspect | Value |
|--------|-------|
| Privileged access | Yes — bypasses partial observability |
| Training efficiency | Very high — no perception learning needed |
| Realism | Low — humans can't query block types at arbitrary positions |
| Multi-agent benefit | Essential — enables shared world model |
| Computation cost | Low for loaded chunks; high for full world |

**Assessment for village-building**: Server-side world state should be used **during centralized training** (as part of the CTDE critic) but **not during decentralized execution** to maintain realistic partial observability. The builder role may need enhanced block grid data, while other roles operate on local observations.

### 1.6 Graph-Based Observations — Scene Graphs, Block Adjacency

Graph-based observations explicitly model relationships between entities and blocks in the environment, which is critical for multi-agent coordination.

**VillagerAgent DAG-Based Task Decomposition** [^100^](https://arxiv.org/html/2406.05720v1):
```python
# VillagerAgent represents tasks as a Directed Acyclic Graph
task_dag = {
    "nodes": [
        {
            "id": "collect_wood",
            "description": "Collect 64 oak_log",
            "agents_assigned": ["gatherer_1"],
            "dependencies": [],                          # No prerequisites
            "status": "completed"
        },
        {
            "id": "craft_planks",
            "description": "Craft 256 oak_planks",
            "agents_assigned": ["builder_1"],
            "dependencies": ["collect_wood"],            # Needs wood first
            "status": "in_progress"
        },
        {
            "id": "build_house",
            "description": "Build 7x7 house at (100, 64, 200)",
            "agents_assigned": ["builder_1", "builder_2"],
            "dependencies": ["craft_planks", "clear_site"],
            "status": "pending"
        }
    ],
    "edges": [
        {"from": "collect_wood", "to": "craft_planks"},
        {"from": "craft_planks", "to": "build_house"},
        {"from": "clear_site", "to": "build_house"}
    ]
}
```

**Scene Graph for Spatial Relationships**:
```python
scene_graph = {
    "entities": [
        {"id": "agent_1", "type": "player", "position": (100, 64, 200), "role": "builder"},
        {"id": "zombie_1", "type": "hostile_mob", "position": (105, 64, 210), "health": 20},
        {"id": "wheat_crop_1", "type": "crop", "position": (110, 64, 190), "growth_stage": 7},
        {"id": "chest_1", "type": "container", "position": (98, 64, 202), "contents": {"oak_log": 32}},
    ],
    "relationships": [
        {"subject": "agent_1", "relation": "adjacent_to", "object": "chest_1", "distance": 2.8},
        {"subject": "zombie_1", "relation": "threatens", "object": "agent_1", "distance": 12.5},
        {"subject": "wheat_crop_1", "relation": "in_farm_plot", "object": "farm_area_1"},
    ],
    "spatial_partitions": {
        "farm_area_1": {"bounds": [(108, 64, 188), (115, 64, 195)], "type": "farmland"},
        "build_site_1": {"bounds": [(95, 64, 195), (105, 66, 205)], "type": "construction_zone"},
    }
}
```

**Why graph observations matter for village-building**:
- **Block adjacency**: Directly encodes which blocks touch which (critical for structural integrity)
- **Entity relationships**: Encodes "defender protects farmer", "gatherer supplies builder"
- **Task dependencies**: DAG structure encodes that walls must precede roof, foundation precedes walls
- **Spatial partitioning**: Farm plots, build sites, and defensive perimeters are explicit regions

---

## 2. Action Space Designs

### 2.1 Low-Level Keypresses — MineRL/VPT

The most fundamental action space maps directly to keyboard and mouse controls.

**MineRL Action Space (Human-Compatible)** [^15^](https://minerl.readthedocs.io/en/v0.4.4/environments/):
```python
{
    "attack": 0 or 1,                    # Left mouse button
    "back": 0 or 1,                      # S key
    "camera": [pitch_delta, yaw_delta],  # Mouse movement in degrees
    "forward": 0 or 1,                   # W key
    "jump": 0 or 1,                      # Space key
    "left": 0 or 1,                      # A key
    "place": "none" or "dirt",           # Right-click placement
    "right": 0 or 1,                     # D key
    "sneak": 0 or 1,                     # Shift key
    "sprint": 0 or 1,                    # Double-tap W / Ctrl+W
}
```

**VPT Action Space** [^20^](https://cdn.openai.com/vpt/Paper.pdf):
- Same low-level controls as MineRL but at 128x128 resolution
- **Action frequency**: 20Hz (matching human gameplay)
- Model outputs discretized actions via a transformer policy
- Action history encoded via autoregressive generation

**Characteristics**:
| Aspect | Value |
|--------|-------|
| Action dimensionality | ~11 discrete + 2 continuous (camera) |
| Execution speed | 20 FPS |
| Sample complexity | Extremely high |
| Human-likeness | Maximum |
| Suitable for village-building | No — too low-level for construction coordination |

### 2.2 High-Level Skills — Voyager

Voyager introduced using **code as the action space**, where each action is a JavaScript program executed via Mineflayer APIs.

**Voyager Skill Library Structure** [^14^](https://voyager.minedojo.org/):
```javascript
// Example Voyager skill: mine wood logs
async function mineWoodLog(bot, numLogs = 1) {
    const woodBlocks = ['oak_log', 'birch_log', 'spruce_log'];
    for (let i = 0; i < numLogs; i++) {
        const woodBlock = bot.findBlock({
            matching: block => woodBlocks.includes(block.name),
            maxDistance: 32
        });
        if (!woodBlock) {
            bot.chat("No wood found nearby!");
            return false;
        }
        await bot.pathfinder.goto(new GoalGetToBlock(woodBlock.position.x, 
                                                       woodBlock.position.y, 
                                                       woodBlock.position.z));
        await bot.dig(woodBlock);
    }
    return true;
}
```

**Voyager Primitive APIs** [^22^](https://arxiv.org/html/2305.16291):
- `bot.pathfinder.goto(goal)` — Navigation
- `bot.equip(item, destination)` — Item equipping
- `bot.dig(block)` — Block breaking
- `bot.placeBlock(referenceBlock, faceVector)` — Block placement
- `bot.craft(recipe, count, craftingTable)` — Crafting
- `bot.attack(entity)` — Combat
- `bot.activateBlock(block)` — Right-click interactions

**ODYSSEY's Extended Skill Library** [^13^](https://www.ijcai.org/proceedings/2025/0022.pdf):
- **40 primitive skills** (vs. Voyager's 18): 32 operational + 8 spatial skills
- **183 compositional skills**: mineX, craftX, plantX, breedX, cookX, etc.
- Recursive skill execution: `mineDiamond` automatically calls `craftIronPickaxe` if needed

### 2.3 Hierarchical Actions — Plan4MC, DEPS

**Plan4MC** [^72^](https://arxiv.org/html/2303.16563v2) defines three tiers of skills:

```python
# Plan4MC skill hierarchy
finding_skills = {
    "explore": "Maximize area traversed to find resources",
    "find_tree": "Navigate to nearest tree",
    "find_cave": "Navigate to nearest cave entrance",
    "find_animal": "Navigate to nearest animal",
}

manipulation_skills = {
    "chop_tree": "Break logs until tree is felled",
    "mine_stone": "Break stone blocks",
    "attack_mob": "Defeat a hostile mob",
    "place_block": "Place a block at target position",
}

crafting_skills = {
    "craft_planks": "Convert logs to planks",
    "craft_sticks": "Convert planks to sticks",
    "craft_pickaxe": "Combine sticks and material for pickaxe",
    "smelt_ore": "Use furnace to smelt ore",
}

# Planning: LLM constructs a skill dependency graph (skill graph)
# Then skill search algorithm walks the graph to find executable sequences
```

**DEPS (Describe, Explain, Plan, Select)** [^74^](https://chemicbook.com/files/deps-summary.pdf):
- **Descriptor**: Summarizes current situation when failure occurs
- **Explainer**: LLM locates errors in the previous plan
- **Planner**: Re-plans the task based on feedback
- **Selector**: Prioritizes sub-tasks by proximity/accessibility
- Increases Minecraft task success rate by **52.74%** over baseline planning

### 2.4 Structured Action Primitives — GITM

GITM defines a compact set of **9 structured actions** [^173^](https://hal.science/hal-04107105v2/file/GITM_hal.pdf):

| Action | Description | Parameters |
|--------|-------------|------------|
| `equip` | Equip item from inventory | item_name, slot |
| `explore` | Explore to discover resources | direction (optional) |
| `approach` | Navigate to target | target_type, target_id |
| `mine` | Break block / Attack entity | target, tool |
| `dig_down` | Dig straight down | layers |
| `go_up` | Ascend (build pillar) | height |
| `build` | Place blocks to construct | block_type, position_list |
| `craft` | Craft items | recipe, count |
| `apply` | Use item on target | item, target |

```python
# GITM action example
gitm_action = {
    "action": "build",
    "parameters": {
        "block_type": "oak_planks",
        "positions": [(100, 64, 200), (101, 64, 200), (100, 65, 200)],
        "pattern": "wall_segment"
    }
}
```

### 2.5 Code-as-Policy — Voyager's Action Generation

Voyager's most significant innovation is treating **generated code as the action space**:

```python
# Voyager action generation pipeline
voyager_action = {
    "type": "generated_javascript",
    "code": """
        async function buildHouseWall(bot) {
            // Get resources from shared chest
            await withdrawItem(bot, new Vec3(98, 64, 202), {"oak_planks": 32});
            
            // Build a 5x3 wall segment
            const wall_positions = [];
            for (let x = 0; x < 5; x++) {
                for (let y = 0; y < 3; y++) {
                    wall_positions.push(new Vec3(100+x, 64+y, 200));
                }
            }
            
            for (const pos of wall_positions) {
                const refBlock = bot.blockAt(pos.offset(0, -1, 0));
                await bot.placeBlock(refBlock, new Vec3(0, 1, 0));
            }
            
            // Return unused materials
            await depositItem(bot, new Vec3(98, 64, 202), {"oak_planks": "all"});
        }
    """,
    "execution_timeout": 300,  # seconds
    "skill_embedding": <vector>,  # For retrieval from skill library
}
```

**Advantages of code-as-policy**:
- **Temporally extended**: A single code action can run for hundreds of timesteps
- **Composable**: Skills build upon other skills via function calls
- **Interpretable**: Code can be read, debugged, and reused
- **Generalizable**: Same skill works in different locations/contexts
- **Persistent**: Stored in skill library for future use

---

## 3. Comparison Matrix

### 3.1 Observation Space Comparison

| Project | Year | Type | Resolution/Size | Key Modalities | Privileged Info | Training Efficiency |
|---------|------|------|-----------------|----------------|-----------------|---------------------|
| **MineRL** | 2019 | Pixel | 64x64x3 | RGB only | None (obfuscated vector in competition) | Very low |
| **VPT** | 2022 | Pixel | 128x128x3 | RGB + HUD overlays | None | Low (requires video pre-training) |
| **GITM** | 2023 | Symbolic | ~200 dims | LiDAR + Voxel + Inventory + GPS | LiDAR rays, full inventory | High |
| **Voyager** | 2023 | Symbolic (text) | Variable | Inventory + Nearby blocks/entities + Biome + Position | Full inventory, nearby entities | Very high (LLM-based, no RL training) |
| **JARVIS-1** | 2023 | Hybrid | 128x128 + text | RGB + Multimodal memory + Text instructions | Memory of past experiences | Medium-high |
| **MineDojo** | 2022 | Hybrid | 128x128 + 8268-dim symbolic | RGB + Voxel + Inventory + Life stats + GPS + Compass | LIDAR (optional) | High with symbolic; low with pure pixels |
| **Optimus-3** | 2025 | Hybrid (VLM) | 128x128 | RGB processed by Qwen2.5-VL + language | Pre-trained VLM knowledge | Medium (requires MLLM fine-tuning) |
| **TeamCraft** | 2024 | Hybrid | 640x480 + inventory | First-person RGB + inventory dict | None (multi-agent collaborative) | Medium |
| **Craftax** | 2024 | Symbolic | 8268-dim flat | One-hot grid + inventory vector | Full grid state | Very high ( JAX-based, 1B steps feasible) |
| **OpenHA** | 2025 | VLM-based | Variable | VLM-processed RGB + action abstractions | Pre-trained VLA knowledge | Medium |

### 3.2 Action Space Comparison

| Project | Year | Type | # Actions | Temporality | Human-Likeness | Sample Complexity | Generalization |
|---------|------|------|-----------|-------------|----------------|-------------------|----------------|
| **MineRL/VPT** | 2019/2022 | Low-level keypress | ~15 | 1 timestep | Max | Extremely high | Poor |
| **MineDojo** | 2022 | Discrete | 24 | 1 timestep | Low | High | Moderate |
| **Plan4MC** | 2023 | Hierarchical skills | ~30 | Extended (skill) | Medium | Medium | Good |
| **DEPS** | 2023 | Hierarchical (LLM-planned) | Variable | Extended (plan) | Medium | N/A (LLM-based) | Very good |
| **GITM** | 2023 | Structured primitives | 9 | Extended (scripted) | Low | High | Good |
| **Voyager** | 2023 | Code-as-policy | Infinite | Extended (program) | Medium | Very high (LLM-based) | Excellent |
| **ODYSSEY** | 2025 | Code-as-policy | 40 primitive + 183 compositional | Extended | Medium | Very high (LLM-based) | Excellent |
| **OpenHA (Grounding)** | 2025 | Grounding actions | 8 | 1-3 timesteps | Medium | Medium | Very good |
| **OpenHA (Motion)** | 2025 | Motion primitives | Variable | 1-10 timesteps | Medium | Medium | Moderate |
| **TeamCraft** | 2024 | Parameterized skills | 8 skill types | Extended | Medium | Medium | Good |

### 3.3 Key Performance Results

| Method | Embodied ASR | GUI ASR | Combat ASR | Inference FPS | Source |
|--------|-------------|---------|------------|---------------|--------|
| VPT | 6.0% | 0.8% | 3.6% | N/A | [^111^](https://arxiv.org/html/2509.13347v1) |
| STEVE-1 | 8.0% | 3.2% | 3.9% | N/A | [^111^](https://arxiv.org/html/2509.13347v1) |
| JARVIS-VLA | 30.0% | 25.1% | 18.5% | N/A | [^111^](https://arxiv.org/html/2509.13347v1) |
| GroundingHA | 37.1% | 6.7% | 26.5% | 5.61 | [^111^](https://arxiv.org/html/2509.13347v1) |
| OpenHA (universal) | 30.1% | 32.5% | 31.9% | 1.36 | [^111^](https://arxiv.org/html/2509.13347v1) |

**Critical finding**: OpenHA's universal training across diverse action spaces consistently outperforms specialist models, demonstrating positive knowledge transfer [^111^](https://arxiv.org/html/2509.13347v1).

---

## 4. Multi-Agent Specific Considerations

### 4.1 Partial Observability in Multi-Agent Settings

Each agent in a multi-agent village-building scenario has limited information about the full world state.

**Dec-POMDP Formalization** [^165^](https://www.emergentmind.com/topics/centralized-training-decentralized-execution-ctde-5f144c51-b600-4937-a6b0-bf976fcfd47b):
```python
# Multi-agent partially observable MDP
multi_agent_state = {
    # Global state (centralized training ONLY)
    "global": {
        "all_positions": {"agent_1": (x1,y1,z1), "agent_2": (x2,y2,z2), ...},
        "shared_inventory": {"oak_log": 128, "wheat": 64, ...},  # pooled resources
        "world_time": 12000,
        "task_dag_state": {"completed": [...], "in_progress": [...]},
    },
    
    # Per-agent observation (decentralized execution)
    "agent_1": {
        "local_rgb": <128x128x3 patch>,
        "inventory": {"oak_log": 16, "stone_axe": 1},
        "nearby_entities": ["agent_2", "cow", "zombie"],  # within 32-block radius
        "nearby_blocks": ["grass_block", "oak_log"],
        "health": 18,
        "position": (100, 64, 200),
        "communication_buffer": [  # Messages from other agents
            {"from": "agent_2", "type": "request_help", "details": "under_attack", "position": (105, 64, 210)},
            {"from": "agent_3", "type": "resource_update", "item": "wheat", "count": 32},
        ],
        "assigned_task": "build_wall_segment_3",
        "visible_agents": ["agent_2"],  # Line-of-sight only
    }
}
```

### 4.2 Communication Channel Design

**Language-Based Multi-Agent Communication** [^36^](https://proceedings.neurips.cc/paper_files/paper/2024/file/a06e129e01e0d2ef853e9ff67b911360-Paper-Conference.pdf):

Key insight from NeurIPS 2024 work on language-grounded MARL: agents learn communication protocols shaped by both RL objectives (what is useful to share) and supervised learning objectives (imitating human-like communication patterns).

**Recommended Communication Schema**:
```python
communication_message = {
    "header": {
        "sender_id": "agent_1",
        "sender_role": "gatherer",
        "timestamp": 12450,
        "priority": "normal",  # "urgent" for threats
    },
    "content": {
        "type": "state_update",  # Types: state_update, request_help, task_offer, coordination
        "payload": {
            # For state_update: share local observation relevant to team
            "position": (100, 64, 200),
            "nearby_resources": [{"type": "oak_log", "count": 8, "location": (102, 64, 198)}],
            "nearby_threats": [{"type": "zombie", "distance": 15, "direction": "south"}],
            "inventory_status": "full",  # or {"oak_log": 36/36}
            "current_task": "collecting_wood_near_river",
            "task_progress": 0.7,
        }
    },
    "routing": {
        "target_roles": ["builder", "defender"],  # Who should receive this
        "proximity_based": True,  # Only agents within range
    }
}
```

**Bandwidth-Constrained Communication** [^172^](https://ojs.aaai.org/index.php/AAAI/article/view/39219):
- Real-world communication has bandwidth limits and packet loss
- **Recommendation**: Compress messages using learned encoders
- Each message should be **at most 128 dimensions** for efficient MARL training

### 4.3 CTDE (Centralized Training with Decentralized Execution)

**CTDE is the prevailing paradigm for multi-agent RL** [^165^](https://www.emergentmind.com/topics/centralized-training-decentralized-execution-ctde-5f144c51-b600-4937-a6b0-bf976fcfd47b):

```python
# CTDE Architecture for Village Building

# --- CENTRALIZED TRAINING PHASE ---
centralized_critic_input = {
    "global_state": {
        "all_agent_positions": [...],       # All positions (privileged)
        "all_inventories": [...],            # All inventories (privileged)
        "full_block_grid": [...],            # World state (privileged)
        "task_dag_progress": [...],          # Global task progress
    },
    "joint_actions": [a1, a2, a3, a4],     # All agents' actions
    "team_reward": r,                       # Shared reward for village completion
}
# Critic computes: Q_total(s, a_1, ..., a_n) -> value

# --- DECENTRALIZED EXECUTION PHASE ---
decentralized_actor_input = {
    "local_observation": {                  # Each agent sees only its own view
        "rgb_patch": <128x128x3>,
        "inventory": {...},
        "nearby_entities": [...],
        "comm_buffer": [msg1, msg2, ...],   # Messages from teammates
    },
    "role_embedding": "builder",           # Role-specific policy
}
# Actor computes: π_i(a_i | local_obs_i, role_i) -> action
```

---

## 5. Concrete Recommendations for Multi-Agent Village Building

### 5.1 Recommended Observation Space by Role

```python
# ============================================================
# GATHERER OBSERVATION SPACE
# ============================================================
gatherer_observation = {
    # Visual modality (for pattern recognition and obstacle detection)
    "rgb_patch": {
        "shape": (128, 128, 3),
        "fov": 90,  # Wider FOV for resource scouting
        "description": "First-person view with extended FOV",
    },
    
    # Symbolic core (high-precision state)
    "agent_state": {
        "position": (float, float, float),      # x, y, z
        "yaw_pitch": (float, float),            # Camera orientation
        "health": int,                          # 0-20
        "hunger": int,                          # 0-20
        "experience_level": int,
    },
    
    # Role-specific: Resource awareness
    "resource_map": {
        "nearby_ores": [
            {"type": "iron_ore", "position": (x,y,z), "distance": float, "quantity_estimate": int}
        ],
        "nearby_trees": [
            {"type": "oak_log", "position": (x,y,z), "trunk_height": int}
        ],
        "nearby_surface_resources": [
            {"type": "sugar_cane", "position": (x,y,z)}
        ],
        "resource_richness_score": float,       # Estimated resources in 64-block radius
    },
    
    # Inventory
    "inventory": {
        "slots_used": int,                      # 0-36
        "items": {"item_name": count, ...},
        "tools": [{"type": "iron_pickaxe", "durability": int}, ...],
    },
    
    # Communication
    "incoming_messages": [communication_message, ...],
    
    # Task context
    "assigned_task": {
        "task_id": str,
        "task_type": "gather_wood" | "gather_stone" | "gather_food" | "explore",
        "target_resource": str,
        "target_quantity": int,
        "dropoff_location": (float, float, float),
    },
    
    # Environmental awareness
    "environment": {
        "biome": str,
        "time_of_day": str,
        "weather": str,
        "nearby_hostiles": [{"type": str, "distance": float, "direction": str}],
    },
}

# ============================================================
# BUILDER OBSERVATION SPACE
# ============================================================
builder_observation = {
    "rgb_patch": {
        "shape": (128, 128, 3),
        "fov": 70,  # Standard FOV for precise placement
    },
    
    "agent_state": {
        "position": (float, float, float),
        "yaw_pitch": (float, float),
        "health": int,
        "hunger": int,
    },
    
    # Role-specific: Construction-critical voxel grid
    "voxel_grid": {
        "shape": (11, 11, 11),                  # 11x11x11 centered on agent
        "encoding": "block_type_id",            # Integer block type per cell
        "relative_positions": True,             # Coordinates relative to agent
    },
    
    # Role-specific: Building blueprint overlay
    "construction_plan": {
        "blueprint_id": str,
        "target_blocks": [
            {"position": (x,y,z), "block_type": "oak_planks", "status": "placed" | "pending"}
        ],
        "progress_fraction": float,              # 0.0 to 1.0
        "next_block_to_place": {"position": (x,y,z), "type": str},
    },
    
    # Inventory with building-relevant organization
    "inventory": {
        "slots_used": int,
        "building_materials": {"oak_planks": 24, "cobblestone": 16},
        "tools": [{"type": "wooden_axe", "durability": 45}],
        "total_material_count": int,             # Quick reference for batch decisions
    },
    
    "incoming_messages": [communication_message, ...],
    
    "assigned_task": {
        "task_id": str,
        "task_type": "build_wall" | "build_floor" | "build_roof" | "place_furniture" | "repair",
        "structure_id": str,
        "dependencies_satisfied": bool,          # Whether prerequisite tasks are done
    },
    
    # Structural awareness
    "adjacent_agents": [
        {"agent_id": str, "role": str, "position": (x,y,z), "distance": float}
    ],
    "shared_chests": [
        {"position": (x,y,z), "contents": {"oak_planks": 64}}
    ],
}

# ============================================================
# FARMER OBSERVATION SPACE
# ============================================================
farmer_observation = {
    "rgb_patch": {
        "shape": (128, 128, 3),
        "fov": 80,
    },
    
    "agent_state": {
        "position": (float, float, float),
        "yaw_pitch": (float, float),
        "health": int,
        "hunger": int,
    },
    
    # Role-specific: Crop and livestock state
    "farm_state": {
        "crop_plots": [
            {
                "position": (x,y,z),
                "crop_type": "wheat" | "carrot" | "potato" | "beetroot",
                "growth_stage": int,                 # 0-7 for wheat
                "hydration_level": float,            # 0.0-1.0
                "needs_action": "water" | "harvest" | "plant" | "bone_meal" | None,
            }
        ],
        "livestock": [
            {
                "type": "cow" | "sheep" | "chicken" | "pig",
                "position": (x,y,z),
                "count_in_pen": int,
                "can_breed": bool,
                "food_to_breed": str,
            }
        ],
        "composters": [
            {"position": (x,y,z), "fill_level": int}  # 0-7
        ],
    },
    
    "inventory": {
        "slots_used": int,
        "seeds": {"wheat_seeds": 16, "carrot": 8},
        "produce": {"wheat": 12, "raw_beef": 4},
        "tools": [{"type": "wooden_hoe", "durability": 30}],
    },
    
    "incoming_messages": [communication_message, ...],
    
    "assigned_task": {
        "task_type": "plant_crops" | "harvest_crops" | "breed_animals" | "collect_eggs" | "craft_food",
        "target_farm_plot": str,
    },
    
    # Seasonal/time awareness (critical for farming)
    "temporal_context": {
        "time_of_day": str,
        "day_number": int,
        "growth_ticks_remaining_estimate": int,  # For planning harvest timing
    },
}

# ============================================================
# DEFENDER OBSERVATION SPACE
# ============================================================
defender_observation = {
    "rgb_patch": {
        "shape": (128, 128, 3),
        "fov": 100,                              # Maximum FOV for threat detection
    },
    
    "agent_state": {
        "position": (float, float, float),
        "yaw_pitch": (float, float),
        "health": int,
        "hunger": int,
        "armor_rating": int,
    },
    
    # Role-specific: Threat assessment (highest priority)
    "threat_assessment": {
        "hostile_entities": [
            {
                "type": "zombie" | "skeleton" | "creeper" | "spider" | "pillager",
                "position": (x,y,z),
                "distance": float,
                "direction": str,                # "north", "northeast", etc.
                "heading_toward_village": bool,
                "estimated_time_to_village": float,
                "priority": int,                 # 1 (immediate) to 5 (monitor)
            }
        ],
        "threat_level": float,                   # 0.0-1.0 aggregate score
        "defensive_perimeter_breaches": [        # Which perimeter sections are compromised
            {"section_id": str, "status": "secure" | "breached" | "at_risk"}
        ],
    },
    
    # Village defense layout
    "defensive_structures": [
        {"type": "wall", "position_range": [(x1,y1,z1), (x2,y2,z2)], "integrity": float},
        {"type": "watchtower", "position": (x,y,z), "occupied": bool},
        {"type": "trap", "position": (x,y,z), "armed": bool},
    ],
    
    # Friendly positions for coordination
    "friendly_positions": [
        {"agent_id": str, "role": str, "position": (x,y,z), "health": int}
    ],
    
    "inventory": {
        "slots_used": int,
        "weapons": [{"type": "iron_sword", "durability": 200, "damage": 6}],
        "armor": [{"slot": "chest", "type": "iron_chestplate", "durability": 180}],
        "ammo": {"arrow": 32},
        "food": {"cooked_beef": 8},              # Critical for sustained combat
    },
    
    "incoming_messages": [communication_message, ...],
    
    "assigned_task": {
        "task_type": "patrol" | "intercept" | "guard_structure" | "repair_defenses" | "escort",
        "patrol_route": [(x,y,z), ...],
        "guard_target": str,                     # Structure or agent to protect
    },
}
```

### 5.2 Recommended Action Space by Role

```python
# ============================================================
# GATHERER ACTION SPACE
# ============================================================
gatherer_actions = {
    # Navigation primitives
    "navigate_to": {
        "target": {"type": "coordinates", "position": (x,y,z)} | {"type": "entity", "name": str},
        "pathfinding": "direct" | "safe" | "resource_optimized",
    },
    "explore_region": {
        "center": (x,y,z),
        "radius": int,                           # Blocks to explore
        "objective": "find_resources" | "scout" | "map_terrain",
    },
    
    # Resource gathering
    "harvest_resource": {
        "resource_type": str,                    # "oak_log", "iron_ore", "wheat", etc.
        "tool": str,                             # "iron_axe", "iron_pickaxe", etc.
        "quantity": int,
        "collection_mode": "mine_nearest" | "strip_mine" | "selective",
    },
    "collect_drops": {
        "item_types": [str],                     # Filter for specific drops
        "radius": int,
    },
    
    # Inventory management
    "deposit_to_chest": {
        "chest_position": (x,y,z),
        "items": {"item_name": quantity, ...},
    },
    "craft_tool": {
        "tool_type": str,
        "quantity": int,
    },
    
    # Communication
    "broadcast_resource_found": {
        "resource_type": str,
        "location": (x,y,z),
        "quantity_estimate": int,
    },
    "return_to_base": {},
}

# ============================================================
# BUILDER ACTION SPACE
# ============================================================
builder_actions = {
    # Construction primitives
    "place_block": {
        "block_type": str,
        "position": (x,y,z),
        "orientation": "normal" | "upside_down" | "east" | "west",  # For directional blocks
    },
    "place_block_sequence": {
        "block_type": str,
        "positions": [(x,y,z), ...],             # Batch placement for walls/floors
    },
    "remove_block": {
        "position": (x,y,z),
        "collect_drops": bool,
    },
    
    # Blueprint-based construction
    "follow_blueprint": {
        "blueprint_id": str,
        "start_position": (x,y,z),               # Anchor point
        "layers": "all" | [int, int],            # Which vertical layers to build
        "material_substitutions": {"oak_planks": "spruce_planks"},
    },
    "check_blueprint_progress": {
        "blueprint_id": str,
    },
    
    # Material management
    "fetch_materials": {
        "from_chest": (x,y,z),
        "materials": {"item_name": quantity, ...},
    },
    "request_materials": {
        "materials_needed": {"oak_planks": 64, "cobblestone": 32},
        "urgency": "normal" | "blocking",
    },
    
    # Quality assurance
    "inspect_structure": {
        "area": [(x1,y1,z1), (x2,y2,z2)],
        "check": "integrity" | "aesthetics" | "completeness",
    },
    "repair_structure": {
        "area": [(x1,y1,z1), (x2,y2,z2)],
    },
}

# ============================================================
# FARMER ACTION SPACE
# ============================================================
farmer_actions = {
    # Crop management
    "till_soil": {
        "positions": [(x,y,z), ...],             # Dirt blocks to convert to farmland
    },
    "plant_crop": {
        "crop_type": str,
        "positions": [(x,y,z), ...],             # Farmland positions
    },
    "apply_bone_meal": {
        "positions": [(x,y,z), ...],             # Accelerate growth
    },
    "harvest_crop": {
        "positions": [(x,y,z), ...],             # Harvest mature crops
        "replant": bool,                         # Whether to replant immediately
    },
    
    # Water management
    "place_water_source": {
        "position": (x,y,z),                     # For irrigation
    },
    "check_hydration": {
        "farm_plot_id": str,
    },
    
    # Livestock management
    "breed_animals": {
        "animal_type": str,
        "feed_item": str,
        "pen_location": (x,y,z),
    },
    "shear_sheep": {
        "target_positions": [(x,y,z), ...],
    },
    "collect_eggs": {
        "chicken_pen": (x,y,z),
    },
    "lead_animal": {
        "animal_type": str,
        "from": (x,y,z),
        "to": (x,y,z),
    },
    
    # Food processing
    "craft_food_item": {
        "recipe": str,                           # "bread", "cake", "pumpkin_pie"
        "quantity": int,
    },
    "deposit_produce": {
        "chest_location": (x,y,z),
        "items": {"item_name": quantity, ...},
    },
}

# ============================================================
# DEFENDER ACTION SPACE
# ============================================================
defender_actions = {
    # Movement and positioning
    "patrol_route": {
        "waypoints": [(x,y,z), ...],
        "pattern": "circular" | "back_and_forth",
    },
    "move_to_defensive_position": {
        "position": (x,y,z),
        "vantage_type": "elevated" | "chokepoint" | "open",
    },
    "intercept_threat": {
        "threat_id": str,
        "interception_point": (x,y,z),
    },
    
    # Combat
    "attack_entity": {
        "target_id": str,
        "weapon": str,                           # "iron_sword", "bow"
        "tactic": "melee" | "ranged" | "hit_and_run",
    },
    "defend_agent": {
        "protect_target": str,                   # Agent ID to protect
        "formation": "shield" | "escort" | "perimeter",
    },
    "retreat_to_safe_zone": {
        "if_health_below": int,                  # e.g., 8 hearts
    },
    
    # Defensive construction
    "build_defensive_structure": {
        "type": "wall" | "watchtower" | "trap" | "moat",
        "position": (x,y,z),
        "material": str,
        "specifications": dict,                  # Size, height, etc.
    },
    "repair_defenses": {
        "area": [(x1,y1,z1), (x2,y2,z2)],
    },
    "arm_trap": {
        "trap_position": (x,y,z),
    },
    
    # Alert and coordination
    "broadcast_threat_warning": {
        "threat_type": str,
        "threat_position": (x,y,z),
        "severity": "low" | "medium" | "high" | "critical",
        "recommended_action": str,
    },
    "request_reinforcement": {
        "location": (x,y,z),
        "threat_description": str,
    },
}
```

### 5.3 Global Task Graph for Village Construction

```python
village_construction_dag = {
    "metadata": {
        "village_id": "village_alpha",
        "total_tasks": 47,
        "estimated_completion_time": "2_hours",
    },
    
    "phases": [
        {
            "phase_id": "site_preparation",
            "description": "Clear and prepare the village site",
            "tasks": [
                {"id": "clear_ground", "action": "remove_blocks", "area": [...], 
                 "agents": ["builder_1", "builder_2"], "dependencies": []},
                {"id": "level_terrain", "action": "fill_and_remove", "target_y": 64,
                 "agents": ["builder_1"], "dependencies": ["clear_ground"]},
                {"id": "place_foundation", "action": "place_blocks", "pattern": "outline",
                 "agents": ["builder_1", "builder_2"], "dependencies": ["level_terrain"]},
            ]
        },
        {
            "phase_id": "resource_gathering",
            "description": "Collect all required materials",
            "tasks": [
                {"id": "gather_wood", "action": "harvest", "resource": "oak_log", "quantity": 256,
                 "agents": ["gatherer_1"], "dependencies": [], "parallel_with": ["clear_ground"]},
                {"id": "gather_stone", "action": "harvest", "resource": "cobblestone", "quantity": 128,
                 "agents": ["gatherer_2"], "dependencies": [], "parallel_with": ["clear_ground"]},
                {"id": "gather_food_seeds", "action": "harvest", "resource": "wheat_seeds", "quantity": 32,
                 "agents": ["farmer_1"], "dependencies": [], "parallel_with": ["clear_ground"]},
                {"id": "process_wood", "action": "craft", "input": "oak_log", "output": "oak_planks", "quantity": 1024,
                 "agents": ["builder_1"], "dependencies": ["gather_wood"]},
            ]
        },
        {
            "phase_id": "construction",
            "description": "Build village structures",
            "tasks": [
                {"id": "build_house_1", "action": "follow_blueprint", "blueprint": "small_house",
                 "agents": ["builder_1"], "dependencies": ["place_foundation", "process_wood"]},
                {"id": "build_house_2", "action": "follow_blueprint", "blueprint": "small_house",
                 "agents": ["builder_2"], "dependencies": ["place_foundation", "process_wood"]},
                {"id": "build_wall", "action": "follow_blueprint", "blueprint": "perimeter_wall",
                 "agents": ["builder_1", "builder_2"], "dependencies": ["place_foundation", "gather_stone"]},
            ]
        },
        {
            "phase_id": "farming",
            "description": "Establish food production",
            "tasks": [
                {"id": "prepare_farm", "action": "till_soil", "area": [...],
                 "agents": ["farmer_1"], "dependencies": ["clear_ground"]},
                {"id": "plant_crops", "action": "plant", "crop_type": "wheat",
                 "agents": ["farmer_1"], "dependencies": ["prepare_farm", "gather_food_seeds"]},
                {"id": "build_pen", "action": "follow_blueprint", "blueprint": "animal_pen",
                 "agents": ["farmer_1", "builder_2"], "dependencies": ["place_foundation"]},
            ]
        },
        {
            "phase_id": "defense",
            "description": "Secure the village perimeter",
            "tasks": [
                {"id": "patrol_perimeter", "action": "patrol", "route": [...],
                 "agents": ["defender_1"], "dependencies": ["build_wall"]},
                {"id": "build_watchtower", "action": "follow_blueprint", "blueprint": "watchtower",
                 "agents": ["builder_1"], "dependencies": ["build_wall"]},
            ]
        }
    ],
    
    # Dynamic task allocation
    "allocation_policy": {
        "mode": "role_primary_with_cross_training",
        "rules": [
            "Gatherers may assist builders when inventory is full and no gathering tasks pending",
            "Builders may assist farmers with pen construction (building skill overlap)",
            "Defenders may assist with urgent gathering if no threats detected for > 5 min",
            "Farmers focus exclusively on food production (critical path for survival)",
        ]
    }
}
```

---

## 6. Implementation Recommendations

### 6.1 Architecture Summary

Based on our comprehensive analysis, we recommend a **hybrid hierarchical architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    VILLAGE CONSTRUCTION SYSTEM               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   TASK DECOMPOSER   │    │   AGENT CONTROLLER  │        │
│  │  (DAG-based, LLM)   │◄──►│  (Task allocation)  │        │
│  └─────────────────────┘    └─────────────────────┘        │
│            │                         │                       │
│            ▼                         ▼                       │
│  ┌─────────────────────────────────────────────────┐        │
│  │           STATE MANAGER (Centralized)            │        │
│  │  • Full world state (training only)              │        │
│  │  • Task DAG progress                             │        │
│  │  • Shared resource pool                          │        │
│  │  • Communication log                             │        │
│  └─────────────────────────────────────────────────┘        │
│            │                                                 │
│            ▼                                                 │
│  ┌─────────────────────────────────────────────────┐        │
│  │            AGENT POOL (4 agents)                 │        │
│  │                                                  │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │        │
│  │  │ GATHERER │ │ BUILDER  │ │  FARMER  │ ...    │        │
│  │  │          │ │          │ │          │        │        │
│  │  │• 128x128 │ │• 128x128│ │• 128x128│        │        │
│  │  │  RGB     │ │  RGB     │ │  RGB     │        │        │
│  │  │• Resource│ │• Voxel   │ │• Crop    │        │        │
│  │  │  map     │ │  11³     │ │  state   │        │        │
│  │  │• Inventory│ │• Blueprint│ │• Livestock│      │        │
│  │  │• Comm    │ │• Comm    │ │• Comm    │        │        │
│  │  └──────────┘ └──────────┘ └──────────┘        │        │
│  │                                                  │        │
│  │  ┌──────────────────────────────────────┐       │        │
│  │  │            DEFENDER                   │       │        │
│  │  │  • 128x128 RGB (FOV 100°)            │       │        │
│  │  │  • Threat assessment map              │       │        │
│  │  │  • Defensive structure state          │       │        │
│  │  │  • Friendly positions                 │       │        │
│  │  └──────────────────────────────────────┘       │        │
│  └─────────────────────────────────────────────────┘        │
│                                                             │
│  CTDE TRAINING: Centralized critic sees global state        │
│  EXECUTION: Each agent acts on local obs + comm messages    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Critical Design Decisions

| Decision | Recommendation | Justification |
|----------|---------------|---------------|
| **Observation type** | Hybrid: symbolic + local RGB patches | Symbolic for precision; RGB for pattern recognition and leveraging pre-trained VLMs |
| **Action type** | Structured high-level skills (parameterized) | 10-100x more sample efficient than low-level keypresses; composable for complex construction |
| **Coordination** | DAG-based task decomposition + comm channels | VillagerAgent demonstrated strong results for multi-agent construction in Minecraft |
| **Training paradigm** | CTDE (MAPPO or QMIX) | Industry standard for cooperative MARL; enables global coordination during training with decentralized execution |
| **Role specialization** | Yes — different obs/action per role | Each role has fundamentally different information needs; attempting a unified space wastes capacity |
| **Communication** | Task-relevant state sharing + threat alerts | Bandwidth-constrained; should be learned (not hand-designed) with content-aware gating |

### 6.3 Training Pipeline

```python
training_pipeline = {
    "phase_1_supervised_pretraining": {
        "description": "Pre-train perception modules on Minecraft data",
        "data": "VPT contractor dataset or MineDojo demonstrations",
        "components": {
            "visual_encoder": "Pre-trained VLM (Qwen2.5-VL or similar)",
            "symbolic_encoder": "MLP inventory/block state encoder",
            "action_decoder": "Skill policy head",
        },
        "duration": "100k-500k steps",
    },
    
    "phase_2_multi_agent_rl": {
        "description": "Train agents together with CTDE",
        "algorithm": "MAPPO (Multi-Agent PPO) or QMIX",
        "critic_input": "Global state (all positions, inventories, task progress)",
        "actor_input": "Local observation + communication buffer",
        "reward_shaping": {
            "team_reward": "village_completion_score",
            "individual_reward": "role_contribution",
            "coordination_bonus": "efficient_resource_sharing",
        },
        "duration": "10M-50M steps per agent",
    },
    
    "phase_3_finetuning": {
        "description": "Fine-tune with LLM-generated skill sequences",
        "method": "RL-GPT style: LLM generates high-level plans, RL executes",
        "benefit": "LLM provides curriculum; RL optimizes execution",
        "duration": "1M-5M steps",
    }
}
```

---

## 7. Open Questions

1. **Visual fidelity vs. training speed trade-off**: What is the optimal RGB resolution for each role? Builders need pixel-perfect placement (higher res), while gatherers need wide FOV for resource detection (lower res, wider view).

2. **Communication bandwidth optimization**: How much information must agents share? The theoretical minimum for coordinated village-building is unknown. Learned communication protocols [^115^](https://arxiv.org/html/2311.14770v1) may discover efficient encodings.

3. **Emergent vs. scripted coordination**: Should the task DAG be pre-defined (as in VillagerAgent) or emergent from agent interaction? Pre-defined DAGs are more reliable; emergent coordination is more flexible but harder to train.

4. **Cross-role knowledge transfer**: How much should role-specific policies share parameters? Full parameter sharing simplifies training but may limit specialization; full separation limits transfer.

5. **Long-horizon credit assignment**: Village construction spans thousands of timesteps. How should rewards be decomposed across agents and time? Success is only observable at completion, making sparse reward propagation challenging.

6. **VLM integration cost**: Running a VLM for each agent at each timestep is computationally prohibitive. Can VLMs be distilled into lightweight perception networks after initial training?

7. **Handling agent failure**: What happens when a gatherer dies or disconnects? How should the remaining agents dynamically reallocate tasks? The system needs robust failure recovery mechanisms not well-studied in current benchmarks.

8. **Scaling beyond 4 agents**: VillagerAgent found that performance peaks at moderate agent counts (2-4) and declines at 8+ due to resource contention and context length issues [^100^](https://arxiv.org/html/2406.05720v1). What architectural changes enable scaling to larger villages?

---

## References

[^5^](https://arxiv.org/html/2506.10357v1) Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts, arXiv 2025.

[^13^](https://www.ijcai.org/proceedings/2025/0022.pdf) ODYSSEY: Empowering Minecraft Agents with Open-World Skills, IJCAI 2025.

[^14^](https://voyager.minedojo.org/) Voyager: An Open-Ended Embodied Agent with Large Language Models, 2023.

[^15^](https://minerl.readthedocs.io/en/v0.4.4/environments/) MineRL Environment Documentation.

[^20^](https://cdn.openai.com/vpt/Paper.pdf) VPT: Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos, OpenAI 2022.

[^31^](https://arxiv.org/pdf/2305.16291) Voyager: An Open-Ended Embodied Agent with Large Language Models, arXiv 2023.

[^36^](https://proceedings.neurips.cc/paper_files/paper/2024/file/a06e129e01e0d2ef853e9ff67b911360-Paper-Conference.pdf) Language Grounded Multi-agent Reinforcement Learning, NeurIPS 2024.

[^63^](https://arxiv.org/abs/2311.05997) JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models, arXiv 2023.

[^66^](https://arxiv.org/html/2502.19902v1) Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy, arXiv 2025.

[^72^](https://arxiv.org/html/2303.16563v2) Plan4MC: Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks, arXiv 2023.

[^73^](https://openreview.net/forum?id=vQDKYYuqWA) Vision-Language Models Provide Promptable Representations for Embodied RL, OpenReview 2024.

[^93^](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md) Mineflayer API Documentation, PrismarineJS.

[^98^](https://arxiv.org/html/2412.05255v1) TeamCraft: A Benchmark for Multi-Modal Multi-Agent Systems in Minecraft, arXiv 2024.

[^100^](https://arxiv.org/html/2406.05720v1) VillagerAgent: A Graph-Based Multi-Agent Framework for Coordinating Complex Task Dependencies in Minecraft, arXiv 2024.

[^111^](https://arxiv.org/html/2509.13347v1) OpenHA: A Series of Open-Source Hierarchical Agentic Models in Minecraft, arXiv 2025.

[^113^](https://arxiv.org/html/2402.16801v1) Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning, arXiv 2024.

[^115^](https://arxiv.org/html/2311.14770v1) Learning to Cooperate and Communicate Over Imperfect Channels, arXiv 2023.

[^134^](https://ar5iv.labs.arxiv.org/html/2206.08853) MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge, arXiv 2022.

[^164^](https://arxiv.org/html/2511.11992v1) Goal-Oriented Multi-Agent Reinforcement Learning for Decentralized Agent Teams, arXiv 2025.

[^165^](https://www.emergentmind.com/topics/centralized-training-decentralized-execution-ctde-5f144c51-b600-4937-a6b0-bf976fcfd47b) CTDE in Multi-Agent Reinforcement Learning, 2025.

[^168^](https://proceedings.neurips.cc/paper_files/paper/2024/file/31f119089f702e48ecfd138c1bc82c4a-Paper-Conference.pdf) RL-GPT: Integrating Reinforcement Learning and Code-as-Policy, NeurIPS 2024.

[^172^](https://ojs.aaai.org/index.php/AAAI/article/view/39219) Communication-efficient Multi-Agent Reinforcement Learning with Spatiotemporal Information Hub, AAAI 2026.

[^173^](https://hal.science/hal-04107105v2/file/GITM_hal.pdf) Ghost in the Minecraft: Generally Capable Agents for Open-World Tasks, 2023.

[^190^](https://arxiv.org/html/2601.05215v1) Task Suite for Memory-Aware Minecraft Agents, arXiv 2026.

[^193^](https://hal.science/hal-04107105v2/file/GITM_hal.pdf) GITM: Ghost in the Minecraft — Structured Actions and Feedback, 2023.
