import math
import numpy as np

import torch
from torch import nn
from torch.nn import Conv1d, AvgPool1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class DiscriminatorS(nn.Module):
    def __init__(
            self,
            scale,
            channels=32,
            num_layers=4
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        c = channels
        g = 1
        self.pool = AvgPool1d(scale)
        self.convs.append(weight_norm(Conv1d(1, c, 41, 1, 20)))
        for _ in range(num_layers):
            self.convs.append(weight_norm(Conv1d(c, c*2, 21, 3, 10, groups=g)))
            g = g*2
            c = c*2
        self.post = weight_norm(Conv1d(c, 1, 21, 1, 10))

    def forward(self, x):
        x = self.pool(x)
        fmap = []
        for l in self.convs:
            x = l(x)
            F.leaky_relu(x, 0.1)
            fmap.append(x)
        logit = self.post(x)
        return logit, fmap
    

class Discriminator(nn.Module):
    def __init__(self, scales=[1, 2, 4], channels=32, num_layers=4):
        super().__init__()
        self.sub_discs = nn.ModuleList()
        for s in scales:
            self.sub_discs.append(DiscriminatorS(s, channels, num_layers))
    
    def forward(self, wf):
        wf = wf.unsqueeze(1)
        logits, fmap = [], []
        for d in self.sub_discs:
            l, f = d(wf)
            logits.append(l)
            fmap += f
        return logits, fmap