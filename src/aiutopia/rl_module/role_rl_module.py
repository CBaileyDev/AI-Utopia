"""Section 7.2 AiUtopiaRoleRLModule — per-role TorchRLModule.

For M1 there is exactly one policy (gatherer). All sub-encoders live as
instance attributes on this RLModule. Cross-policy weight sharing for M2+
is a separate refactor.

The forward path handles both (B, obs_dim) inference obs and (B, T, obs_dim)
training obs that RLlib produces under the LSTM connector with max_seq_len>1.
The original v1 plan collapsed the time dim with unsqueeze(1) which silently
disabled the LSTM during training.
"""
from __future__ import annotations

from typing import Any

import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

from aiutopia.rl_module.actor_head     import build_actor_head
from aiutopia.rl_module.core_encoder   import flatten_core_obs_batched
from aiutopia.rl_module.ctde_critic    import CTDECriticModule, VILLAGE_INV_DIM
from aiutopia.rl_module.role_encoder   import build_role_encoder
from aiutopia.rl_module.shared_backbone import SharedBackboneModule
from aiutopia.rl_module.core_encoder   import CoreEncoderModule


class AiUtopiaRoleRLModule(TorchRLModule):
    """Per-role policy module with embedded shared submodules.

    Stateful: LSTM hidden carried across ticks via `Columns.STATE_IN`/`STATE_OUT`.
    """

    def setup(self) -> None:
        super().setup()
        cfg = self.model_config
        self.role = cfg["role"]
        self.core_encoder    = CoreEncoderModule(cfg.get("core_encoder",
                                                          {"core_hidden": [512, 256]}))
        self.shared_backbone = SharedBackboneModule(cfg.get("shared_backbone",
                                                              {"lstm_hidden": 256}))
        self.ctde_critic     = CTDECriticModule(cfg.get("ctde_critic", {}))
        self.role_encoder    = build_role_encoder(self.role, cfg)
        self.actor_head      = build_actor_head(self.role, cfg)
        self.pixel_encoder   = None         # M2+ for builder
        self._pixel_zero_dim = 64

    # ─────────────────────────────────────────────────────────────
    # RLlib new-API-stack stateful contract
    def get_initial_state(self) -> dict[str, torch.Tensor]:
        """Per-AGENT initial state (unbatched, rank-1)."""
        H = self.shared_backbone.lstm_hidden
        device = next(self.parameters()).device
        return {"h": torch.zeros(H, device=device),
                "c": torch.zeros(H, device=device)}

    # ─────────────────────────────────────────────────────────────
    # Forward — handles inference (B, ...) and training (B, T, ...)
    def _forward_inference(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_exploration(self, batch, **kwargs):
        return self._forward(batch, with_value=False)

    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, with_value=True)

    def _forward(self, batch: dict, *, with_value: bool) -> dict:
        obs = batch[Columns.OBS]

        # Detect leading shape. Compare against a known 1-D-per-sample key
        # (goal_embedding is (B, GOAL_EMBED_DIM) at inference and
        # (B, T, GOAL_EMBED_DIM) under recurrent training).
        ref = obs["goal_embedding"]
        time_dimension = (ref.ndim == 3)
        if time_dimension:
            batch_size, seq_len = ref.shape[:2]
            # Flatten T into B for per-tick encoders, restore for LSTM.
            obs_flat_t = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        else:
            batch_size = ref.shape[0]
            seq_len = 1
            obs_flat_t = obs

        core_input = flatten_core_obs_batched(obs_flat_t)        # (B*T, D)
        core_feat  = self.core_encoder(core_input)                # (B*T, 256)
        role_feat  = self.role_encoder(obs_flat_t)                # (B*T, 128)
        pixel_feat = torch.zeros(core_feat.size(0), self._pixel_zero_dim,
                                   device=core_feat.device)       # (B*T, 64)
        fused = torch.cat([core_feat, role_feat, pixel_feat], dim=-1)  # (B*T, 448)

        # Reshape to (B, T, 448) for LSTM
        fused_seq = fused.view(batch_size, seq_len, -1)

        # State: RLlib gives (B, H); LSTM wants (1, B, H).
        state_in = batch.get(Columns.STATE_IN, None)
        if state_in is None:
            h0, c0 = self.shared_backbone.initial_state(
                batch_size=batch_size, device=fused_seq.device)
        else:
            h0 = state_in["h"].unsqueeze(0).to(fused_seq.device)
            c0 = state_in["c"].unsqueeze(0).to(fused_seq.device)

        backbone_out, (h1, c1) = self.shared_backbone(fused_seq, (h0, c0))
        # backbone_out: (B, T, 256)

        # Flatten time back into batch for the actor head
        hidden_flat = backbone_out.reshape(-1, backbone_out.size(-1))  # (B*T, 256)
        goal_flat   = obs_flat_t["goal_embedding"]                       # (B*T, 512)
        action_dist_inputs_flat = self.actor_head(hidden_flat, goal_flat)  # (B*T, 340)

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
            # CTDE critic: M1 single-agent, other 3 slots zero-padded.
            n = core_input.size(0)                                     # B*T
            all_agents = torch.zeros(n, 4, core_input.size(-1),
                                      device=core_feat.device)
            all_agents[:, 0, :] = core_input
            village_inv = torch.zeros(n, VILLAGE_INV_DIM,
                                        device=core_feat.device)
            v_flat = self.ctde_critic(all_agents, village_inv)         # (B*T,)
            if time_dimension:
                result[Columns.VF_PREDS] = v_flat.view(batch_size, seq_len)
            else:
                result[Columns.VF_PREDS] = v_flat

        return result
