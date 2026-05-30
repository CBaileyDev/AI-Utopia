"""Farmer RoleRLModule — crop planting/waiting/harvesting policy.

§M2 — Farmer learns temporal credit assignment (delayed rewards, 64 ticks plant→harvest).
Reuses CoreEncoder + SharedBackbone + CTDE critic; role-specific encoder handles
f_crop_grid (32×32 ConvNet) + f_ripeness + f_planted_count (scalars).
Actor head outputs 7 skills (PLOW, PLANT, HARVEST, NAVIGATE, WAIT, etc.) with
spatial/scalar params.
"""

from __future__ import annotations

import torch
from ray.rllib.core import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchMultiDistribution,
)
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from aiutopia.rl_module.actor_head import (
    build_actor_head,
    farmer_action_dist_config,
)
from aiutopia.rl_module.core_encoder import (
    CoreEncoderModule,
    flatten_core_obs_batched,
)
from aiutopia.rl_module.ctde_critic import VILLAGE_INV_DIM, CTDECriticModule
from aiutopia.rl_module.role_encoder import build_role_encoder
from aiutopia.rl_module.shared_backbone import SharedBackboneModule


class FarmerRoleRLModule(TorchRLModule, ValueFunctionAPI):
    """Per-role policy module for Farmer (crop cultivation with delayed rewards).

    Reuses the same core architecture as GathererRoleRLModule:
    CoreEncoder + SharedBackbone (LSTM) + role-specific encoder + ActorHead.
    Stateful: LSTM hidden carried across ticks via `Columns.STATE_IN`/`STATE_OUT`.
    """

    def setup(self) -> None:
        super().setup()
        cfg = self.model_config
        self.role = cfg["role"]
        assert self.role == "farmer", (
            f"FarmerRoleRLModule requires role='farmer', got {self.role!r}"
        )

        self.core_encoder = CoreEncoderModule(cfg.get("core_encoder", {"core_hidden": [512, 256]}))
        self.shared_backbone = SharedBackboneModule(
            cfg.get("shared_backbone", {"lstm_hidden": 256})
        )
        self.ctde_critic = CTDECriticModule(cfg.get("ctde_critic", {}))
        self.role_encoder = build_role_encoder(self.role, cfg)
        self.actor_head = build_actor_head(self.role, cfg)
        self.pixel_encoder = None  # M2+ for builder
        self._pixel_zero_dim = 64

        # Set the action distribution class. Farmer outputs 7 skills + spatial/scalar params.

        if self.role == "farmer":
            child_struct, input_lens = farmer_action_dist_config()
            self.action_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
                child_distribution_cls_struct=child_struct,
                input_lens=input_lens,
            )
        else:
            raise NotImplementedError(f"action_dist_cls for role {self.role!r} not configured")

    # ─────────────────────────────────────────────────────────────
    # RLlib new-API-stack stateful contract
    def get_initial_state(self) -> dict[str, torch.Tensor]:
        """Per-AGENT initial state (unbatched, rank-1)."""
        H = self.shared_backbone.lstm_hidden
        device = next(self.parameters()).device
        return {"h": torch.zeros(H, device=device), "c": torch.zeros(H, device=device)}

    # ─────────────────────────────────────────────────────────────
    # Forward — handles inference (B, ...) and training (B, T, ...)
    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, with_value=True)

    # ─────────────────────────────────────────────────────────────
    # ValueFunctionAPI — required by Ray 2.55's GAE connector.
    def compute_values(self, batch, embeddings=None):
        out = self._forward(batch, with_value=True)
        return out[Columns.VF_PREDS]

    def _forward(self, batch: dict, *, with_value: bool) -> dict:
        obs = batch[Columns.OBS]

        # Detect leading shape via a known 1-D-per-sample key.
        ref = obs["goal_embedding"]
        time_dimension = ref.ndim == 3

        def _flatten_time(v):
            """Fold T into B for any tensor; recurse into nested dicts."""
            if isinstance(v, dict):
                return {k: _flatten_time(sub) for k, sub in v.items()}
            if torch.is_tensor(v) and v.ndim >= 2:
                return v.reshape(-1, *v.shape[2:])
            return v

        if time_dimension:
            batch_size, seq_len = ref.shape[:2]
            obs_flat_t = {k: _flatten_time(v) for k, v in obs.items()}
        else:
            batch_size = ref.shape[0]
            seq_len = 1
            obs_flat_t = obs

        core_input = flatten_core_obs_batched(obs_flat_t)  # (B*T, D)
        core_feat = self.core_encoder(core_input)  # (B*T, 256)
        role_feat = self.role_encoder(obs_flat_t)  # (B*T, 128)
        pixel_feat = torch.zeros(
            core_feat.size(0), self._pixel_zero_dim, device=core_feat.device
        )  # (B*T, 64)
        fused = torch.cat([core_feat, role_feat, pixel_feat], dim=-1)  # (B*T, 448)

        # Reshape to (B, T, 448) for LSTM
        fused_seq = fused.view(batch_size, seq_len, -1)

        # State: RLlib gives (B, H); LSTM wants (1, B, H).
        state_in = batch.get(Columns.STATE_IN)
        if state_in is None:
            h0, c0 = self.shared_backbone.initial_state(
                batch_size=batch_size, device=fused_seq.device
            )
        else:
            h0 = state_in["h"].unsqueeze(0).to(fused_seq.device)
            c0 = state_in["c"].unsqueeze(0).to(fused_seq.device)

        backbone_out, (h1, c1) = self.shared_backbone(fused_seq, (h0, c0))
        # backbone_out: (B, T, 256)

        # Flatten time back into batch for the actor head
        hidden_flat = backbone_out.reshape(-1, backbone_out.size(-1))  # (B*T, 256)
        goal_flat = obs_flat_t["goal_embedding"]  # (B*T, 512)
        action_dist_inputs_flat = self.actor_head(hidden_flat, goal_flat)  # (B*T, ~344)

        # Restore time dim for RLlib if we had one
        if time_dimension:
            action_dist_inputs = action_dist_inputs_flat.view(batch_size, seq_len, -1)
        else:
            action_dist_inputs = action_dist_inputs_flat

        result = {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
            Columns.STATE_OUT: {"h": h1.squeeze(0), "c": c1.squeeze(0)},
        }

        if with_value:
            # CTDE critic: M2 pilot, dummy agents at indices 0, 2, 3.
            n = core_input.size(0)  # B*T
            all_agents = torch.zeros(n, 4, core_input.size(-1), device=core_feat.device)
            all_agents[:, 2, :] = core_input  # farmer is agent index 2
            village_inv = torch.zeros(n, VILLAGE_INV_DIM, device=core_feat.device)
            v_flat = self.ctde_critic(all_agents, village_inv)  # (B*T,)
            if time_dimension:
                result[Columns.VF_PREDS] = v_flat.view(batch_size, seq_len)
            else:
                result[Columns.VF_PREDS] = v_flat

        return result
