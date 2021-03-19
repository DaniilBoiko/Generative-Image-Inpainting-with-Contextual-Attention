import yaml

from inpainting.train import GAN
from inpainting.data import ImageDataset

from torch.utils.data import DataLoader
import pytorch_lightning as pl

config = yaml.load(open('model_config.yaml'))

PATH = ''

image_loader = DataLoader(
    ImageDataset(PATH, max_mask_size=64),
    bs=4, shuffle=True, num_workers=16, drop_last=True
)

gan = GAN(config['Model'], config['OptParams'])
trainer = pl.Trainer()
trainer.fit(gan, image_loader)