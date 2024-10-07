import torch
import torch.fft as fft
import torch.nn.functional as F


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


def fft_convolve(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    '''
    depthwise causal convolution using fft for performance

    signal: [N, C, L]
    kernel: [N, C, kernel_size]
    Output: [N, C, L]
    '''
    kernel = F.pad(kernel, (0, signal.shape[-1] - kernel.shape[-1]))

    signal = F.pad(signal, (0, signal.shape[-1]))
    kernel = F.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def generate_harmonics(f0: torch.Tensor, amps: torch.Tensor, sr: float, frame_size: int, min_frequency: float = 20.0) -> torch.Tensor:
    '''
    generate harmonic signal

    f0: [N, 1, L]
    amps: [N, C, L]
    Output: [N, 1, L]
    '''
    device = f0.device
    mul = (torch.arange(amps.shape[1], device=device) + 1).unsqueeze(0).unsqueeze(2)
    fs = F.interpolate(f0, scale_factor=frame_size, mode='linear') * mul
    amps = F.interpolate(amps, scale_factor=frame_size, mode='linear')
    uv = (f0 > min_frequency).to(torch.float)
    uv = F.interpolate(uv, scale_factor=frame_size, mode='linear')
    I = torch.cumsum(fs / sr, dim=2) # integration
    theta = 2 * torch.pi * (I % 1) # phase
    harmonics = (torch.sin(theta) * uv * amps).sum(dim=1, keepdim=True)
    return harmonics


def generate_noise(amps: torch.Tensor, n_fft: int, frame_size: int) -> torch.Tensor:
    '''
    generate filtered noise

    fft_bin = n_fft // 2 + 1
    amps: [N, fft_bin, L]
    '''
    angle = torch.rand_like(amps) * 2 * torch.pi
    noise_stft = torch.exp(1j * angle) # euler's equation: exp(1j * theta) = cos(theta) + 1j * sin(theta)
    noise_stft *= amps # amplitude modulation
    noise_stft = F.pad(noise_stft, [1, 0])
    window = torch.hann_window(n_fft, device=amps.device)
    noise = torch.istft(noise_stft, n_fft, frame_size, window=window)
    noise = noise.unsqueeze(1)
    return noise


def generate_filterd_impulse(f0: torch.Tensor, spectral_envelope: torch.Tensor, n_fft: int, frame_size: int, sample_rate: float):
    '''
    f0: [N, 1, L]
    spectral_envelope: [N, fft_bin, L]
    frame_size: int
    sample_rate: float

    Output: [N, 1, L * frame_size]
    '''
    f0 = F.interpolate(f0, scale_factor=frame_size, mode='linear')
    I = torch.cumsum(f0, dim=2)
    sawtooth = (I / sample_rate) % 1.0
    impulse = sawtooth - sawtooth.roll(-1, dims=(2)) + (f0 / sample_rate)

    impulse = impulse.squeeze(1)
    window = torch.hann_window(n_fft, device=f0.device)
    impulse_stft = torch.stft(impulse, n_fft, frame_size, window=window, return_complex=True)[:, :, 1:] * spectral_envelope
    impulse_stft = F.pad(impulse_stft, (1, 0))
    impulse = torch.istft(impulse_stft, n_fft, frame_size, window=window).unsqueeze(1)
    return impulse


def additive_synthesizer(
        f0: torch.Tensor,
        harmonics_amplitude: torch.Tensor,
        noise_spectral_envelope: torch.Tensor,
        ir: torch.Tensor,
        n_fft: int,
        frame_size: int,
        sr: float
    ):
    '''
    f0: [N, C, L]
    harmonics_amplitude: [N, n_harmonics, L]
    noise_spectral_envelope: [N, fft_bin, L]
    ir: [N, 1, ir_length]
    output: [N, L * frame_size]
    '''
    dtype = harmonics_amplitude.dtype

    f0 = f0.to(torch.float)
    harmonics_amplitude = harmonics_amplitude.to(torch.float)
    noise_spectral_envelope = noise_spectral_envelope.to(torch.float)
    ir = ir.to(torch.float)
    
    harmonics = generate_harmonics(f0, harmonics_amplitude, sr, frame_size)
    noise = generate_noise(noise_spectral_envelope, n_fft, frame_size)
    o = fft_convolve(harmonics + noise, ir)
    o = o.squeeze(1)

    o = o.to(dtype)
    return o


def subtractive_synthesizer(
        f0: torch.Tensor,
        periodic_spectral_envelope: torch.Tensor,
        noise_spectral_envelope: torch.Tensor,
        n_fft: int,
        frame_size: int,
        sr: float
    ):
    '''
    f0: [N, C, L]
    periodic_spectral_envelope: [N, fft_bin, L]
    noise_spectral_envelope: [N, fft_bin, L]
    ir: [N, 1, ir_length]
    output: [N, L * frame_size]
    '''
    dtype = periodic_spectral_envelope.dtype

    f0 = f0.to(torch.float)
    periodic_spectral_envelope = periodic_spectral_envelope.to(torch.float)
    noise_spectral_envelope = noise_spectral_envelope.to(torch.float)

    periodic = generate_filterd_impulse(f0, periodic_spectral_envelope, n_fft, frame_size, sr)
    noise = generate_noise(noise_spectral_envelope, n_fft, frame_size)
    o = periodic + noise
    o = o.squeeze(1)

    o = o.to(dtype)
    return o