import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def oscillate_impulse(f0: torch.Tensor, frame_size: int, sample_rate: float):
    '''
    f0: [N, 1, L]
    frame_size: int
    sample_rate: float

    Output: [N, 1, L * frame_size]
    '''
    f0 = F.interpolate(f0, scale_factor=frame_size, mode='linear')
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impulse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    return impulse


def filter(wf: torch.Tensor, kernel: torch.Tensor, n_fft: int, frame_size: int):
    '''
    stft-based FIR Filter

    wf: [N, L * frame_size]
    kernel: [N, fft_bin, L], where fft_bin = n_fft // 2 + 1
    frame_size: int

    Output: [N, L * frame_size]
    '''
    device = wf.device
    window = torch.hann_window(n_fft, device=device)
    wf_stft = torch.stft(wf, n_fft, frame_size, window=window, return_complex=True)[:, :, 1:]
    out_stft = F.pad(wf_stft * kernel, [0, 1])
    out = torch.istft(out_stft, n_fft, frame_size, window=window)
    return out


def vocoder(f0: torch.Tensor, noise_spec: torch.Tensor, impulse_response: torch.Tensor, frame_size: int, n_fft: int, sample_rate: float):
    '''
    f0: [N, 1, L], fundamental frequency
    noise_spec: [N, fft_bin, L], where fft_bin = n_fft // 2 + 1, noise spectrogram
    impulse_response: [N, n_fft, L] framewise impulse response
    frame_size: int
    n_fft: int

    Output: [N, L * frame_size]
    '''

    dtype = f0.dtype
    f0 = f0.to(torch.float)
    noise_spec = noise_spec.to(torch.float)
    impulse_response = impulse_response.to(torch.float)
    
    pulse = oscillate_impulse(f0, frame_size, sample_rate).squeeze(1)
    noise = torch.randn_like(pulse)
    noise = filter(noise, noise_spec, n_fft, frame_size)
    ir = torch.fft.rfft(impulse_response, dim=1)
    pulse = filter(pulse, ir, n_fft, frame_size)
    output = noise + pulse

    output = output.to(dtype)
    return output