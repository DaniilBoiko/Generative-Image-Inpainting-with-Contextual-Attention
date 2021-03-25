import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from .utils import cov


def compute_psnr(img, img_hat):
    """
    Computes PSNR, using formula from:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Assumes values to be in [0, 1] range
    """
    mse = torch.mean((img - img_hat) ** 2)

    return -10 * torch.log10(mse)


def compute_tv(img, img_hat):
    """
    Computes total variation reduction

    See details in:
    https://en.wikipedia.org/wiki/Total_variation_denoising
    """

    def tv(y):
        tv_h = torch.sum(torch.abs(y[:, 1:, :] - y[:, :-1, :]))
        tv_v = torch.sum(torch.abs(y[:, :, 1:] - y[:, :, -1]))

        return tv_h + tv_v

    img_tv = tv(img)
    img_hat_tv = tv(img_hat)

    return img_hat_tv / (img_tv + 1e-5)


def compute_metrics(img, img_hat):
    """
    Computes many different metrics
    Assumes values to be in [0, 1] range
    """
    metrics = {
        'l1': torch.mean(torch.abs(img - img_hat)),
        'l2': torch.mean((img - img_hat) ** 2),
        'PSNR': compute_psnr(img, img_hat),
        'TV': compute_tv(img, img_hat),
        'SSIM': ssim(
            img.cpu().numpy(), img_hat.cpu().numpy(),
            multichannel=True, data_range=1
        )
    }

    return metrics
