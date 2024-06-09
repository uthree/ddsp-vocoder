import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator
from .discriminator import Discriminator

from module.utils.loss import generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss, multiscale_stft_loss


class Cordvox(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(**config["generator"])
        self.discriminator = Discriminator(**config["discriminator"])
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.discriminator_active = True

    def training_step(self, batch):
        wf, mel, f0 = batch

        # get optimizer
        opt_g, opt_d = self.optimizers()

        # aliases
        G, D = self.generator, self.discriminator
        
        fake = G.synthesize(mel, f0).squeeze(1)
        loss_stft = multiscale_stft_loss(wf, fake)

        if self.discriminator_active:
            logits_fake, feats_fake = D(fake)
            logits_real, feats_real = D(wf)
            loss_adv = generator_adversarial_loss(logits_fake)
            loss_feat = feature_matching_loss(feats_real, feats_fake)
            loss_G = loss_stft + loss_feat + loss_adv
        else:
            loss_G = loss_stft

        # backward G.
        self.manual_backward(loss_G)
        nn.utils.clip_grad_norm_(G.parameters(), 1.0, 2.0)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        if self.discriminator_active:
            # train discriminator
            self.toggle_optimizer(opt_d)
            fake = fake.detach()
            logits_fake, _ = D(fake)
            logits_real, _ = D(wf)
            loss_D = discriminator_adversarial_loss(logits_real, logits_fake)

            # backward D.
            self.manual_backward(loss_D)
            nn.utils.clip_grad_norm_(D.parameters(), 1.0, 2.0)
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

            loss_dict = {
                "MS-STFT": loss_stft.item(),
                "Generator Adversarial": loss_adv.item(),
                "Feature Matching": loss_feat.item(),
                "Discriminator Adversarial": loss_D.item(),
            }
        else:
            loss_dict = {
                "MS-STFT": loss_stft.item(),
            }

        for k, v in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"loss/{k}", v)

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=1e-4, betas=(0.8, 0.99))
        opt_d = optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=(0.8, 0.99))
        return opt_g, opt_d