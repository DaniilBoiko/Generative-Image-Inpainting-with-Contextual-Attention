import torch.functional as F

import pytorch_lightning as pl
from inpainting.model import *


class GAN(pl.LightningModule):

    def __init__(self, config, opt_params):
        super().__init__()
        self.save_hyperparameters()

        self.coarse_network = CoarseNetwork(config['CoarseNetwork'])
        self.refinement_network = RefinementNetwork(config['RefinementNetwork'])

        self.local_critic = LocalCritic(config['LocalCritic'])
        self.global_critic = GlobalCritic(config['GlobalCritic'])

    def forward(self, x):
        coarse_output = self.coarse_network(x)
        refined_output = self.refinement_network(coarse_output)

        return [coarse_output, refined_output]

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        if optimizer_idx == 0:
            cn_output = self.coarse_network(imgs)
            self.gan_output = self.refinement_network(cn_output)

            loss = None
            loss(self.cn_output)

            pass

        if optimizer_idx == 1:
            pass

        if optimizer_idx == 2:
            pass

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            [list(self.coarse_network.parameters()) + list(self.refinement_network.parameters())],
            lr=self.hparams.opt_params['CoarseNetwork'].lr,
            betas=self.hparams.opt_params['CoarseNetwork'].betas
        )

        opt_lc = torch.optim.Adam(self.local_critic.parameters(),
                                  lr=self.hparams.opt_params['LocalCritic'].lr,
                                  betas=self.hparams.opt_params['LocalCritic'].betas)

        opt_gc = torch.optim.Adam(self.global_critic.parameters(),
                                  lr=self.hparams.opt_params['GlobalCritic'].lr,
                                  betas=self.hparams.opt_params['GlobalCritic'].betas)

        return [opt_gen, opt_lc, opt_gc], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
