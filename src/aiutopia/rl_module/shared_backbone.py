"""Section 4.3 SharedBackbone — Linear projection + LSTM(256)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

FUSED_INPUT_DIM = 448   # core(256) + role(128) + pixel(64), per section 4.3


class SharedBackboneModule(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.lstm_hidden = config.get("lstm_hidden", 256)
        self.proj = nn.Sequential(
            nn.Linear(FUSED_INPUT_DIM, 384),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, fused, state):
        # fused: (B, T, 448) ; state: ((1,B,H), (1,B,H))
        projected = self.proj(fused)
        out, new_state = self.lstm(projected, state)
        return out, new_state

    def initial_state(self, batch_size: int, *, device: torch.device | str = "cpu"):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return h, c
