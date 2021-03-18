import torch


def compute_psnr(img, img_hat):
    """
    Computes PSNR, using formula from:
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Assumes values to be in [0, 1] range
    """
    mse = torch.mean((img - img_hat) ** 2)

    return -10 * torch.log10(mse)


def compute_ssim(img, img_hat):
    pass


def compute_tv(img, img_hat):
    pass


def compute_FID(img, img_hat):
    pass


def compute_metrics(img, img_hat):
    """Computes many different metrics
    """
    metrics = {
        'l1': torch.mean(torch.abs(img - img_hat)),
        'l2': torch.mean((img - img_hat) ** 2),
        'PSNR': compute_psnr(img, img_hat),
        'TV': None,
        'SSIM': None,
        'FID': None
    }

    return metrics
