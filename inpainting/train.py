from collections import OrderedDict

import torch.nn.functional as F

import pytorch_lightning as pl
from inpainting.model import *

from inpainting.data import random_bbox_fixed
from inpainting.utils import spatial_discount

from torchvision.utils import make_grid
from imageio import imwrite
import numpy as np


class GAN(pl.LightningModule):

    def __init__(self, config, opt_params, weight_cliping_limit=0.01, bbox_size=64):
        super().__init__()
        self.save_hyperparameters()

        self.coarse_network = CoarseNetwork(config['CoarseNetwork'])

        self.global_critic = GlobalCritic(config['GlobalCritic'])
        self.bbox_size = bbox_size

    def forward(self, x):
        coarse_output = self.coarse_network(x)

        return coarse_output

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, imgs, batch_idx, optimizer_idx):
        for p in self.global_critic.parameters():
            p.data.clamp_(-self.hparams.weight_cliping_limit, self.hparams.weight_cliping_limit)


        bbox = random_bbox_fixed(64, 64, input_shape=(256, 256))

        x = imgs.float()
        x[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top] = 0.0

        spatial_dis = spatial_discount(0.999, (64, 64), True).cuda()

        cn_output = self.coarse_network(x)

        x_cn = cn_output

        if batch_idx % 100 == 0:
            grid = make_grid(x)
            self.logger.experiment.add_image('source', grid, batch_idx)

            grid = make_grid(x_cn)
            self.logger.experiment.add_image('output', grid, batch_idx)

            grid = make_grid(imgs)
            self.logger.experiment.add_image('gt', grid, batch_idx)

        if optimizer_idx == 0:
            gc_preds_fake = self.global_critic(x_cn)
            gc_preds_real = self.global_critic(x)

            loss = gc_preds_fake.mean() - gc_preds_real.mean()

            self.logger.log_metrics({'D_loss': loss.item()}, batch_idx)
            return loss

        if optimizer_idx == 1:
            l1_losses = spatial_dis * (
                (torch.abs(x_cn - x))[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top]
            )

            gc_preds_fake = self.global_critic(x_cn)

            adversarial_losses = - gc_preds_fake.mean()

            loss = l1_losses.mean() + adversarial_losses

            self.logger.log_metrics({'G_loss': loss.item()}, batch_idx)
            return loss

    def configure_optimizers(self):
        opt_D = torch.optim.RMSprop(
            list(self.global_critic.parameters()),
            lr=self.hparams.opt_params['D']['lr'])

        opt_G = torch.optim.RMSprop(
            list(self.coarse_network.parameters()),
            lr=self.hparams.opt_params['G']['lr'],
        )
        return [opt_D, opt_G], []
