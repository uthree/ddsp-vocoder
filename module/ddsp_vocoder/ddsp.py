import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectrogram(waveform, n_fft, frame_size):
    device = waveform.device
    w = torch.hann_window(n_fft, device=device)
    spec = torch.stft(waveform, n_fft, frame_size, window=w, return_complex=True).abs()[:, :, 1:]
    return spec


def oscillate_impluse(f0: torch.Tensor, frame_size: int, sample_rate: float):
    '''
    f0: [N, 1, L]
    frame_size: int
    sample_rate: float

    Output: [N, 1, L * frame_size]
    '''
    f0 = F.interpolate(f0, scale_factor=frame_size, mode='linear')
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impluse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    return impluse


def filter(wf: torch.Tensor, kernel: torch.Tensor, n_fft: int, frame_size: int):
    '''
    wf: [N, L * frame_size]
    kernel: [N, fft_bin, L], where fft_bin = n_fft // 2 + 1
    frame_size: int

    Output: [N, L * frame_size]
    '''
    C = kernel.shape[1]
    device = wf.device
    window = torch.hann_window(n_fft, device=device)
    wf_stft = torch.stft(wf, n_fft, frame_size, window=window, return_complex=True)[:, :, 1:]
    out_stft = F.pad(wf_stft * kernel, [0, 1])
    out = torch.istft(out_stft, n_fft, frame_size, window=window)
    return out


def vocoder(f0: torch.Tensor, periodicity: torch.Tensor, kernel: torch.Tensor, frame_size: int, n_fft: int, sample_rate: float):
    '''
    f0: [N, 1, L], fundamental frequency
    periodicity: [N, 1, L], periodicty
    kernel: [N, fft_bin, L], where fft_bin = n_fft // 2 + 1, spectral envelope
    frame_size: int
    n_fft: int

    Output: [N, L * frame_size]
    '''
    impluse = oscillate_impluse(f0, frame_size, sample_rate)
    noise = torch.rand_like(impluse)
    periodicity = F.interpolate(periodicity, scale_factor=frame_size, mode='linear')
    source = (periodicity * impluse + (1-periodicity) * noise).squeeze(1)
    output = filter(source, kernel, n_fft, frame_size)
    return output