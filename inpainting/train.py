from collections import OrderedDict

import torch.nn.functional as F

import pytorch_lightning as pl
from inpainting.model import *

from inpainting.data import random_bbox_fixed
from inpainting.utils import spatial_discount

from torchvision.utils import make_grid
import numpy as np

import imageio


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

    def forward(self, x, mask):
        coarse_output = self.coarse_network(x)
        refined_output = self.refinement_network(coarse_output, mask)

        return refined_output

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, imgs, batch_idx, optimizer_idx):
        bbox = random_bbox_fixed(64, 64, input_shape=(256, 256))

        masks = torch.zeros(imgs.shape)
        masks[:, :, bbox.left:bbox.right, bbox.bottom:bbox.top] += 1.0
        masks = masks.float().cuda()

        x = imgs.float()
        x[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top] = 1.0

        # import ipdb; ipdb.set_trace()

        spatial_dis = spatial_discount(0.999, (64, 64), True).cuda()

        if optimizer_idx == 0:
            cn_output = self.coarse_network(x)
            rn_output = self.refinement_network(cn_output, masks)

            x_cn = cn_output * masks + x * (1 - masks)
            x_rn = rn_output * masks + x * (1 - masks)

            lc_preds_fake = self.local_critic(x_rn[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top])
            gc_preds_fake = self.global_critic(x_rn)

            lc_preds_real = self.local_critic(imgs.float()[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top])
            gc_preds_real = self.global_critic(imgs.float())

            loss = (lc_preds_fake - lc_preds_real).mean() + (gc_preds_fake - gc_preds_real).mean() + \
                   self.hparams.opt_params['lambdaGP'] * self.compute_gradient_penalty(imgs.float(), x_rn, bbox)

            self.logger.experiment.add_scalars('D_metrics', {'D_loss': loss.item()}, batch_idx)
            return loss

        if optimizer_idx == 1:
            cn_output = self.coarse_network(x)
            rn_output = self.refinement_network(cn_output, masks)

            x_cn = cn_output * masks + x * (1 - masks)
            x_rn = rn_output * masks + x * (1 - masks)

            l1_losses = (spatial_dis * (
                (torch.abs(x_cn - imgs.float()) + torch.abs(x_rn - imgs.float()))[:, :, bbox.left: bbox.right,
                bbox.bottom: bbox.top]
            )).mean() + torch.abs(rn_output * (1 - masks) - x * (1 - masks)).mean()

            lc_preds_fake = self.local_critic(x_rn[:, :, bbox.left: bbox.right, bbox.bottom: bbox.top])
            gc_preds_fake = self.global_critic(x_rn)

            adversarial_losses = -lc_preds_fake.mean() - gc_preds_fake.mean()

            loss = l1_losses + adversarial_losses * 0.001

            self.logger.experiment.add_scalars(
                'G_metrics',
                {'G_loss_l1': l1_losses.item(), 'G_loss_adv': adversarial_losses.item(), 'G_loss': loss.item()},
                batch_idx
            )

            if batch_idx % 100 == 0:
                self.logger.experiment.add_image('input', make_grid((x + 1) / 2), batch_idx)
                self.logger.experiment.add_image('mask', make_grid((masks + 1) / 2), batch_idx)
                self.logger.experiment.add_image('cn_output', make_grid((cn_output + 1) / 2), batch_idx)
                self.logger.experiment.add_image('rn_output', make_grid((rn_output + 1) / 2), batch_idx)
                self.logger.experiment.add_image('x_cn', make_grid((x_cn + 1) / 2), batch_idx)
                self.logger.experiment.add_image('x_rn', make_grid((x_rn + 1) / 2), batch_idx)

            return loss

    def compute_gradient_penalty(self, real, fake, bbox):
        alpha = torch.Tensor(np.random.random((real.size(0), 1, 1, 1))).to(self.device)

        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        interpolates_masked = (alpha * real + ((1 - alpha) * fake))[:, :, bbox.left: bbox.right,
                              bbox.bottom: bbox.top].requires_grad_(True)

        lc_interpolates = self.local_critic(interpolates_masked)
        gc_interpolates = self.global_critic(interpolates)

        fake_tensor = torch.Tensor(real.shape[0], 1).fill_(1.0).to(self.device)
        lc_gradients = torch.autograd.grad(
            outputs=lc_interpolates,
            inputs=interpolates_masked,
            grad_outputs=fake_tensor,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        lc_gradients = lc_gradients.view(lc_gradients.size(0), -1).to(self.device)

        gc_gradients = torch.autograd.grad(
            outputs=gc_interpolates,
            inputs=interpolates,
            grad_outputs=fake_tensor,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gc_gradients = gc_gradients.view(gc_gradients.size(0), -1).to(self.device)

        gradient_penalty = ((lc_gradients.norm(2, dim=1) - 1) ** 2).mean() + (
                (gc_gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def configure_optimizers(self):
        opt_D = torch.optim.RMSprop(
            list(self.local_critic.parameters()) + list(self.global_critic.parameters()),
            lr=self.hparams.opt_params['D']['lr'])

        opt_G = torch.optim.RMSprop(
            list(self.coarse_network.parameters()) + list(self.refinement_network.parameters()),
            lr=self.hparams.opt_params['G']['lr']
        )
        return [opt_D, opt_G], []
