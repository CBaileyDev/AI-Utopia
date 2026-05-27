"""Section 7.2 RLlib RLModule layer for the AI Utopia village (M1).

Module composition (all live as instance attributes on AiUtopiaRoleRLModule):
  - CoreEncoderModule      universal core obs -> 256-d
  - SharedBackboneModule   Linear(448->384) + LSTM(256)
  - CTDECriticModule       two-stage encoder + MLP -> V(s)
  - GathererRoleEncoder    role-specific obs -> 128-d
  - GathererActorHead      action distribution producer

M2 may refactor to share Core/Backbone/Critic across multiple role policies;
the M1 plan deliberately ships one policy with bare nn.Module attributes.
"""
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule, core_obs_flat_dim, flatten_core_obs_batched,
)

__all__ = ["CoreEncoderModule", "core_obs_flat_dim", "flatten_core_obs_batched"]
