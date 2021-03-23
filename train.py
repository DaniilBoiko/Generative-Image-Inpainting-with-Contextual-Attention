import yaml

from inpainting.train import GAN
from inpainting.data import ImageDataset

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torchvision.utils import make_grid
import numpy as np
import imageio

config = yaml.load(open('model_config.yaml'))

PATH = 'data/test_256'

print(PATH)

image_loader = DataLoader(
    ImageDataset(PATH),
    batch_size=12, shuffle=True, num_workers=16, drop_last=True
)

imageio.imwrite('color_debug_2.png', (np.moveaxis(ImageDataset(PATH)[0], 0, -1) + 1) / 2)
imageio.imwrite('color_debug_1.png',
                (np.moveaxis(make_grid(next(iter(image_loader))).cpu().detach().numpy(), 0, -1) + 1) / 2)

gan = GAN(config['Model'], config['OptParams'])
trainer = pl.Trainer(gpus=1)
trainer.fit(gan, image_loader)
