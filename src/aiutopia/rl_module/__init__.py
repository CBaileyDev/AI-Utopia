"""Section 7.2 RLlib RLModule layer for the AI Utopia village (M1).

Module composition (all live as instance attributes on AiUtopiaRoleRLModule):
  - CoreEncoderModule      universal core obs -> 256-d
  - SharedBackboneModule   Linear(448->384) + LSTM(256)
  - CTDECriticModule       two-stage encoder + MLP -> V(s)
  - GathererRoleEncoder    role-specific obs -> 128-d
  - GathererActorHead      action distribution producer

M2 may refactor to share Core/Backbone/Critic across multiple role policies;
the M1 plan deliberately ships one policy with bare nn.Module attributes.

## Ray ConnectorV2 obs-flatten note (T21 fix #9, generalised)

Ray 2.55's new-API-stack ConnectorV2 pipeline flattens Dict obs to 2-D
(B, prod(shape)) before passing them to the RLModule. Unit tests construct
4-D obs directly so they don't hit this; only real EnvRunner rollouts do.

When a role encoder takes a multi-D obs (image, grid, patch), it must use
`unflatten_role_obs` to restore the spatial shape. Pattern, copy-paste-ready
for M2's BuilderRoleEncoder pixel patch:

    grid = unflatten_role_obs(obs["g_resource_grid"], (32, 32, 6))
    grid = grid.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W) for conv2d
"""
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule, core_obs_flat_dim, flatten_core_obs_batched,
    unflatten_role_obs,
)

__all__ = [
    "CoreEncoderModule",
    "core_obs_flat_dim",
    "flatten_core_obs_batched",
    "unflatten_role_obs",
]
