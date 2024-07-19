import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator

from module.ddsp_vocoder.loss import multiscale_stft_loss


class DDSPVocoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(**config["generator"])
        self.save_hyperparameters()

    def training_step(self, batch):
        wf, mel, f0 = batch
        
        ap_hat, se_hat = self.generator(mel)
        fake = self.generator.ddsp(ap_hat, se_hat, f0).squeeze(1)

        loss_stft = multiscale_stft_loss(wf, fake)

        self.log("MS-STFT", loss_stft.item())

        return loss_stft
    
    def configure_optimizers(self):
        opt_g = optim.AdamW(self.parameters(), lr=1e-4, betas=(0.8, 0.99))
        return opt_g