import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_dim, h_dim=None, out_dim=None, activation=nn.GELU, drop=0.):
        super().__init__()

        out_dim = out_dim or in_dim
        h_dim = h_dim or in_dim

        self.mlp = nn.Sequential([
            nn.Linear(in_dim, h_dim),
            activation(),
            nn.Linear(h_dim, out_dim),
            nn.Dropout(drop),
        ])

    def forward(self, x):
        return self.mlp(x)

