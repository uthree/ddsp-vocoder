import torch
import torch.fft as fft
import torch.nn.functional as F

from typing import Optional


def log_spectrogram(waveform, n_fft, frame_size):
    '''
    calculate log linear spectrogram

    waveform: [N, L * frame_size]
    Output: [N, fft_bin, L]

    fft_bin = n_fft // 2 + 1
    '''
    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(waveform, n_fft, frame_size, window=window, return_complex=True)
    spec = torch.log(spec.abs() + 1e-6)
    spec = spec[:, :, 1:]
    return spec


def framewise_fir_filter(
    signal: torch.Tensor,
    filter: torch.Tensor,
    n_fft: int,
    hop_length: int,
    center: bool = True,
) -> torch.Tensor:
    """
    args:
        signal: [batch_size, length * hop_length]
        filter: [batch_size, n_fft, length]
        n_fft: int
        hop_length: int
        center: bool
    outputs:
        signal: [batch_size, length * hop_length]
    """

    dtype = signal.dtype

    x = signal.to(torch.float)
    window = torch.hann_window(n_fft, device=x.device)
    x_stft = torch.stft(
        x, n_fft, hop_length, n_fft, window, center, return_complex=True
    )
    filter = F.pad(filter, (0, 1), mode="replicate")
    filter_stft = torch.fft.rfft(filter, dim=1)
    x_stft = x_stft * filter_stft
    x = torch.istft(x_stft, n_fft, hop_length, n_fft, window, center)
    signal = x.to(dtype)
    return signal


def spectral_envelope_filter(
    signal: torch.Tensor, envelope: torch.Tensor, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    args:
        signal: [batch_size, length * hop_length]
        envelope: [batch_size, fft_bin, length], where fft_bin = n_fft // 2 + 1
    outputs:
        signal: [batch_size, length * hop_length]
    """
    window = torch.hann_window(n_fft, device=signal.device)
    signal_stft = (
        torch.stft(signal, n_fft, hop_length, window=window, return_complex=True)[
            :, :, 1:
        ]
        * envelope
    )
    signal_stft = F.pad(signal_stft, (0, 1))
    signal = torch.istft(signal_stft, n_fft, hop_length, window=window)
    return signal


def impulse_train(
    f0: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    uv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    args:
        f0: [batch_size, length]
        sample_rate: int
        hop_length: int
        uv: [batch_size, length]
    outputs:
        signal: [batch_size, length * hop_length]
    """
    f0 = f0.unsqueeze(1)
    f0 = F.interpolate(f0, scale_factor=hop_length, mode="linear")
    if uv is not None:
        uv = uv.to(f0.dtype)
        uv = uv.unsqueeze(1)
        uv = F.interpolate(uv, scale_factor=hop_length)
        f0 = f0 * uv
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impulse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)
    impulse = impulse.squeeze(1)
    return impulse
