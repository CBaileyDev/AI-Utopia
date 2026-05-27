"""Section 4.3 CTDE critic with two-stage encoder."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aiutopia.rl_module.core_encoder import core_obs_flat_dim


PER_AGENT_COMPRESSED_DIM = 128
NUM_AGENT_SLOTS = 4
VILLAGE_INV_DIM = 64


class CTDECriticModule(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        head_hidden = config.get("critic_hidden", 256)
        in_dim = core_obs_flat_dim()
        self.per_agent_encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, PER_AGENT_COMPRESSED_DIM),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(NUM_AGENT_SLOTS * PER_AGENT_COMPRESSED_DIM + VILLAGE_INV_DIM,
                       head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, all_agents_obs, village_inv):
        batch, n_slots, _ = all_agents_obs.shape
        flat = all_agents_obs.reshape(batch * n_slots, -1)
        compressed = self.per_agent_encoder(flat).reshape(batch, -1)
        x = torch.cat([compressed, village_inv], dim=-1)
        return self.head(x).squeeze(-1)
