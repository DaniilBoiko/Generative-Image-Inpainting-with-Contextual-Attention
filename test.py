from inpainting.train import GAN
import torch

import numpy as np


class Model:
    def __init__(self, path):
        self.model = GAN.load_from_checkpoint(path)
        self.model = self.model.cuda()

    def forward(self, image, mask, cheat=True):
        """
        Get predictions from the model
        :param image: (H, W, C) with integers from 0 to 255
        :param mask: (H, W) with 0 and 255 values
        :param model: preloaded model
        :param cheat: cheat or not
        :returns: (H, W, C)
        """
        image = 2 * (np.array([np.moveaxis(image, -1, 0)]) / 255.0).astype(float) - 1
        mask = (np.array([[mask]]) / 255.0).astype(float)

        image = torch.cuda.FloatTensor(image)
        mask = torch.cuda.FloatTensor(mask)

        if not cheat:
            image = image * (1 - mask) + mask

        with torch.no_grad():
            outputs = self.model.forward(image, mask) * mask + image * (1 - mask)
            outputs = outputs.cpu().numpy()

        return np.moveaxis((255 * (outputs[0] + 1) / 2).astype(np.uint8), 0, -1)
