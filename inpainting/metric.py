import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from .utils import cov

EPSILON = 1e-5
FID_MODEL = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
FID_MODEL.fc = nn.Identity()
FID_MODEL.eval()

FID_MODEL_TRANSFORMS = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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

    return img_hat_tv / (img_tv + EPSILON)


def compute_fid(img, img_hat):
    img_transformed = FID_MODEL(FID_MODEL_TRANSFORMS(img).unsqueeze(0))[0]
    img_mu, img_sigma = torch.mean(img_transformed), cov(img_transformed)

    img_hat_transformed = FID_MODEL(FID_MODEL_TRANSFORMS(img_hat).unsqueeze(0))[0]
    img_hat_mu, img_hat_sigma = torch.mean(img_hat_transformed), cov(img_hat_transformed)

    ssdiff = torch.sum((img_mu - img_hat_mu) ** 2)
    covmean = torch.nan_to_num(torch.sqrt(torch.dot(img_transformed, img_hat_transformed)))

    return ssdiff + torch.trace(img_sigma + img_hat_sigma - 2 * covmean)


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
        ),
        'FID': compute_fid(img, img_hat)
    }

    return metrics
