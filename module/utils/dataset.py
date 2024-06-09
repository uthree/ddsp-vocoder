import torch
import torchaudio
import json
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
import os


class VocoderDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir='dataset_cache'):
        super().__init__()
        self.root = Path(cache_dir)
        self.audio_file_paths = []
        self.feature_paths = []

        for path in self.root.glob("*.wav"):
            self.audio_file_paths.append(path)
            self.feature_paths.append(path.with_suffix('.pt'))

    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.audio_file_paths[idx])
        features = torch.load(self.feature_paths[idx])
        mel = features['mel'].squeeze(0)
        f0 = features['f0'].squeeze(0)
        return wf.mean(dim=0), mel, f0
        
    def __len__(self):
        return len(self.audio_file_paths)


class VocoderDataModule(L.LightningDataModule):
    def __init__(
            self,
            cache_dir='dataset_cache',
            batch_size=1,
            num_workers=1,
            ):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = VocoderDataset(
                self.cache_dir)
        dataloader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=(os.name=='nt'))
        return dataloader