import torch
import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, edge_out_dim: int):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_layer):
        
        out = self.decoder(hidden_layer)

        return out