import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=24000, n_fft=1920, hop_length=480, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft, hop_length=hop_length, n_mels=n_mels)
    
    def forward(self, x):
        return safe_log(self.mel(x)[:, :, 1:])