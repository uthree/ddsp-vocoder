import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddsp import subtractive_synthesizer


# Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    # x: [BatchSize, cnannels, *]
    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x.mT, (self.channels,), self.gamma, self.beta, self.eps)
        return x.mT


# Global Resnponse Normalization for 1d Sequence (shape=[BatchSize, Channels, Length])
class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    # x: [batchsize, channels, length]
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ConvNeXt v2
class ConvNeXtLayer(nn.Module):
    def __init__(self, channels=512, kernel_size=7, mlp_mul=2):
        super().__init__()
        padding = kernel_size // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, groups=channels)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.grn = GRN(channels * mlp_mul)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)

    # x: [batchsize, channels, length]
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.c3(x)
        x = x + res
        return x


class Generator(nn.Module):
    def __init__(
            self,
            n_mels=80,
            internal_channels=256,
            num_layers=4,
            n_fft=1920,
            frame_size=480,
            sample_rate=48000,
        ):
        super().__init__()
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        fft_bin = n_fft // 2 + 1
        self.input_layer = nn.Conv1d(n_mels, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(internal_channels) for _ in range(num_layers)])
        self.post_norm = LayerNorm(internal_channels)
        self.to_aperiodic = nn.Conv1d(internal_channels, fft_bin, 1)
        self.to_periodic = nn.Conv1d(internal_channels, fft_bin, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.post_norm(x)
        periodic = F.softplus(self.to_periodic(x))
        aperiodic = F.softplus(self.to_aperiodic(x))
        return periodic, aperiodic
    
    def synthesize(self, x, f0):
        periodic, aperiodic = self.forward(x)
        output = subtractive_synthesizer(f0, periodic, aperiodic, self.n_fft, self.frame_size, self.sample_rate)
        return output