import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int):
        super().__init__()

        self.node_attr_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU()
        )
        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_attr):

        x_encoded = self.node_attr_encoder(x)
        edge_attr_encoded = self.edge_attr_encoder(edge_attr)

        return x_encoded, edge_attr_encoded