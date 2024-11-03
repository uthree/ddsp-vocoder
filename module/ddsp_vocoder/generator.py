import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddsp import impulse_train, spectral_envelope_filter, framewise_fir_filter


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
            num_filters=4,
        ):
        super().__init__()
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        fft_bin = n_fft // 2 + 1
        self.fft_bin = fft_bin
        self.num_filters = num_filters
        self.input_layer = nn.Conv1d(n_mels, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(internal_channels) for _ in range(num_layers)])
        self.post_norm = LayerNorm(internal_channels)
        self.to_aperiodic = nn.Conv1d(internal_channels, fft_bin, 1)
        self.to_periodic = nn.Conv1d(internal_channels, fft_bin, 1)
        self.to_filters = nn.Conv1d(internal_channels, n_fft * num_filters, 1)
    
    def net(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.post_norm(x)
        aperiodic = F.softplus(self.to_aperiodic(x))
        periodic = F.softplus(self.to_periodic(x))
        filters = self.to_filters(x)
        return aperiodic, periodic, filters
    
    def forward(self, x, f0):
        aperiodic, periodic, filters = self.net(x)
        dtype = x.dtype

        impulse = impulse_train(f0.squeeze(1), self.frame_size, self.sample_rate).to(torch.float)
        noise = torch.randn_like(impulse).to(torch.float)
        impulse = spectral_envelope_filter(impulse, periodic, self.n_fft, self.frame_size)
        noise = spectral_envelope_filter(noise, aperiodic, self.n_fft, self.frame_size)
        signal = noise + impulse
        filters = torch.chunk(filters, self.num_filters, dim=1)
        aux_outputs = []
        for f in filters:
            aux_outputs.append(signal.to(dtype))
            signal = framewise_fir_filter(signal, f.to(torch.float), self.n_fft, self.frame_size) + signal
        return signal.to(dtype), aux_outputs
    
    def infer(self, x, f0):
        out, aux = self.forward(x, f0)
        return out