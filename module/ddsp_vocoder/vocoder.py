import torch
import torch.nn as nn
import torch.nn.functional as F


def oscillate_cyclic_noise(
        f0: torch.Tensor,
        frame_size: int,
        sample_rate: float,
        base_frequency: int = 100.0,
        beta: float = 0.78,
        kernel_size: int = 1024
    ):
    with torch.no_grad():
        f0 = F.interpolate(f0, scale_factor=frame_size, mode='linear')
        rad = torch.cumsum(-f0 / sample_rate, dim=2)
        sawtooth = rad % 1.0
        impluse = sawtooth - sawtooth.roll(1, dims=(2))
        impluse = F.pad(impluse, (kernel_size - 1, 0))
        t = torch.arange(0, kernel_size, device=f0.device)[None, None, :]
        decay = torch.exp(-t * base_frequency / beta / sample_rate)
        noise = torch.randn_like(decay)
        kernel = noise * decay
        cyclic_noise = F.conv1d(impluse, kernel)
    return cyclic_noise


def filter(wf: torch.Tensor, kernel: torch.Tensor, n_fft: int, frame_size: int):
    '''
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


def ddsp(f0: torch.Tensor, ap: torch.Tensor, se: torch.Tensor, frame_size: int, n_fft: int, sample_rate: float, max_aperiodicty: float = 0.12):
    '''
    f0: [N, 1, L], fundamental frequency
    ap: [N, 1, L], aperiodicty
    se: [N, fft_bin, L], where fft_bin = n_fft // 2 + 1, spectral envelope
    frame_size: int
    n_fft: int

    Output: [N, 1, L * frame_size]
    '''
    cyclic_noise = oscillate_cyclic_noise(f0, frame_size, sample_rate)
    noise = torch.randn_like(cyclic_noise)
    ap = F.interpolate(ap, scale_factor=frame_size, mode='linear') * max_aperiodicty
    source = (cyclic_noise + noise * ap).squeeze(1)
    output = filter(source, se, n_fft, frame_size)
    return torch.tanh(output.unsqueeze(1))
