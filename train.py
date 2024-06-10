import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from module.ddsp_vocoder import DDSPVocoder
from module.utils.dataset import VocoderDataModule
from module.utils.config import load_json_file
from module.utils.safetensors import save_tensors

class SaveCheckpoint(L.Callback):
    def __init__(self, models_dir, interval=200):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.interval = interval
        if not self.models_dir.exists():
            self.models_dir.mkdir()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.interval == 0:
            ckpt_path = self.models_dir / "model.ckpt"
            trainer.save_checkpoint(ckpt_path)
            save_tensors(pl_module.generator.state_dict(), self.models_dir / "generator.safetensors")
            save_tensors(pl_module.discriminator.state_dict(), self.models_dir / "discriminator.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/default.json")
    parser.add_argument("-nod", "--no-discriminator", type=bool, default=False)
    parser.add_argument("-b", "--batch-size", default=0, type=int)
    args = parser.parse_args()

    config = load_json_file(args.config)
    model_path = Path(config['save']['models_dir']) / "model.ckpt"
    cb_save_checkpoint = SaveCheckpoint(config['save']['models_dir'], interval=config['save']['interval'])
    trainer = L.Trainer(**config["trainer"], callbacks=[cb_save_checkpoint])

    if model_path.exists():
        print(f"loading model from {model_path}")
        model = DDSPVocoder.load_from_checkpoint(model_path)
    else:
        model = DDSPVocoder(config["model"])
    
    dm_config = config['data_module']

    if args.batch_size != 0:
        dm_config['batch_size'] = args.batch_size

    dm = VocoderDataModule(**dm_config)
    if args.no_discriminator:
        model.discriminator_active = False
    else:
        model.discriminator_active = True
    trainer.fit(model, dm)
