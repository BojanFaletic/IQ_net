import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, n_input: int, n_hypotheses: int):
        super().__init__()
        net_shape = n_input * n_hypotheses

        self.hypotheses_net = nn.Sequential(
            nn.Linear(n_input, net_shape),
            nn.ReLU(),
            nn.Linear(net_shape, n_input)
        )
        self.ln = nn.LayerNorm(n_input)

    def forward(self, x):
        a = self.hypotheses_net(x)
        out = self.ln(a + x)
        return out


class Transformer(nn.Module):
    def __init__(self, n_input: int, depth: int, width: int):
        super().__init__()
        self.trans_layer = nn.Sequential(
            *[TransformerBlock(n_input, width)] * depth
            )

    def forward(self, x):
        out = self.trans_layer(x)
        return torch.sigmoid(out)
