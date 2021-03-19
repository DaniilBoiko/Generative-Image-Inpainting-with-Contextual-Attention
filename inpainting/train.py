import torch.nn.functional as F

import pytorch_lightning as pl
from inpainting.model import *

from inpainting.data import random_bbox_fixed
from inpainting.utils import spatial_discount


class GAN(pl.LightningModule):

    def __init__(self, config, opt_params, bbox_size=64):
        super().__init__()
        self.save_hyperparameters()

        self.coarse_network = CoarseNetwork(config['CoarseNetwork'])
        self.refinement_network = RefinementNetwork(config['RefinementNetwork'])

        self.local_critic = LocalCritic(config['LocalCritic'])
        self.global_critic = GlobalCritic(config['GlobalCritic'])

        self.l1_loss = torch.nn.L1Loss()

        self.bbox_size = bbox_size

    def forward(self, x):
        coarse_output = self.coarse_network(x)
        refined_output = self.refinement_network(coarse_output)

        return [coarse_output, refined_output]

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, imgs, batch_idx, optimizer_idx):
        bbox = random_bbox_fixed(64, 64, input_shape=(256, 256))

        masks = torch.zeros(imgs.shape)[:, 0, :, :]
        masks[:, :, bbox.left:bbox.right, bbox.bottom:bbox.top] += 1.0

        x = imgs
        x[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top] = 0.0

        spatial_dis = spatial_discount(0.999, (imgs.shape[2], imgs.shape[3]), True)

        cn_output = self.coarse_network(x)
        rn_output = self.refinement_network(cn_output, masks)

        x_cn = cn_output * masks + x * (1 - masks)
        x_rn = rn_output * masks + x * (1 - masks)



        if optimizer_idx == 0:
            l1_losses = spatial_dis * (
                (
                        self.l1_loss(x_cn, imgs) + self.l1_loss(x_rn, imgs)
                )[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top]
            )
            adversarial_losses = 0

            loss = l1_losses + adversarial_losses
            loss.backward()

        if optimizer_idx == 1:
            pass

        if optimizer_idx == 2:
            pass

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            [list(self.coarse_network.parameters()) + list(self.refinement_network.parameters())],
            lr=self.hparams.opt_params['Generator']['lr'],
        )

        opt_lc = torch.optim.Adam(self.local_critic.parameters(),
                                  lr=self.hparams.opt_params['LocalCritic']['lr'])

        opt_gc = torch.optim.Adam(self.global_critic.parameters(),
                                  lr=self.hparams.opt_params['GlobalCritic']['lr'])

        return [opt_gen, opt_lc, opt_gc], []
