from inpainting.train import GAN
import torch

import numpy as np

model = GAN.load_from_checkpoint('epoch=1-step=54749.ckpt')
model = model.cuda()


def get_predictions(image, mask, model):
    image = 2 * (np.array([np.moveaxis(image, -1, 0)]) / 255.0).astype(float) - 1
    mask = (np.array([[mask]]) / 255.0).astype(float)

    image = torch.cuda.FloatTensor(image)
    mask = torch.cuda.FloatTensor(mask)

    with torch.no_grad():
        outputs = model.forward(image, mask) * mask + image * (1 - mask)
        outputs = outputs.cpu().numpy()

    return np.moveaxis((255*(outputs[0]+1)/2).astype(np.uint8), 0, -1)


image = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
mask = (np.random.rand(256, 256) * 255).astype(np.uint8)

outputs = get_predictions(image, mask, model)
