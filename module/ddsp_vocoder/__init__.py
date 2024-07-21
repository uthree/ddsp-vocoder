import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator
from .discriminator import Discriminator

from module.ddsp_vocoder.loss import multiscale_stft_loss, feature_loss, generator_loss, discriminator_loss


class DDSPVocoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator(**config["generator"])
        self.discriminator = Discriminator(**config["discriminator"])

    def training_step(self, batch):
        wf, mel, f0 = batch
        opt_G, opt_D = self.optimizers()
        G, D = self.generator, self.discriminator
        
        # Train Generator
        opt_G.zero_grad()
        self.toggle_optimizer(opt_G)
        fake = G.synthesize(mel, f0)
        logits_fake, fmap_fake = D(fake)
        _, fmap_real = D(wf)
        loss_stft = multiscale_stft_loss(wf, fake)
        loss_feat = feature_loss(fmap_real, fmap_fake)
        loss_adv = generator_loss(logits_fake)
        loss_G = loss_stft * 45.0 + loss_feat + loss_adv
        self.manual_backward(loss_G)
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_G.step()
        self.untoggle_optimizer(opt_G)

        # Train Discriminator
        opt_D.zero_grad()
        fake = fake.detach()
        self.toggle_optimizer(opt_D)
        logits_fake, _= D(fake)
        logits_real, _= D(wf)
        loss_D = discriminator_loss(logits_real, logits_fake)
        self.manual_backward(loss_D)
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_D.step()
        self.untoggle_optimizer(opt_D)

        self.log("MS-STFT", loss_stft.item())
        self.log("Feature", loss_feat.item())
        self.log("G. Adv.", loss_adv.item())
        self.log("D. Adv.", loss_D.item())

        return loss_stft
    
    def configure_optimizers(self):
        opt_G = optim.AdamW(self.generator.parameters(), lr=1e-4, betas=(0.8, 0.99))
        opt_D = optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=(0.8, 0.99))
        return opt_G, opt_D