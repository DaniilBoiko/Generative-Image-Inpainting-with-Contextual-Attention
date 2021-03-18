import torch
from torch import nn

from .layers import ContextualAttention


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """
    Estimates covariance matrix like numpy.cov

    Copy-pasted from https://github.com/pytorch/pytorch/issues/19037
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def build_layers(config: [str], in_channels=3, input_size=256):
    layers = []

    need_flatten = True
    current_size = input_size

    n_of_channels = in_channels
    for line in config:
        params = None

        splitted_line = line.split('_')
        if len(splitted_line) == 2:
            layer_name, params = splitted_line
        else:
            layer_name = line

        converted_params = []
        if layer_name == 'conv' and params:
            name_cache = ''
            property_cache = ''
            previous = ''

            for symbol in params:
                if not symbol.isdigit() and previous.isdigit():
                    converted_params.append([name_cache, int(property_cache)])
                    name_cache = ''
                    property_cache = ''

                previous = symbol

                if symbol.isdigit():
                    property_cache += symbol

                if not symbol.isdigit():
                    name_cache += symbol

            if name_cache and property_cache:
                converted_params.append([name_cache, int(property_cache)])

        converted_params = {k: v for k, v in converted_params}

        if layer_name == 'conv':
            layers.append(
                nn.Conv2d(in_channels=n_of_channels,
                          out_channels=converted_params['C'],
                          kernel_size=converted_params['K'],
                          stride=converted_params['S'],
                          padding=converted_params['D'] if 'D' in converted_params else converted_params['K'] // 2,
                          dilation=converted_params['D'] if 'D' in converted_params else 1
                          )
            )
            layers.append(
                nn.ELU()
            )

            n_of_channels = converted_params['C']
            current_size /= converted_params['S']

        elif layer_name == 'ContextualAttentionLayer':
            layers.append(
                ContextualAttention()
            )
            pass

        elif layer_name == 'fc':
            if need_flatten:
                layers.append(
                    torch.nn.Flatten()
                )
                need_flatten = False

            layers.append(
                torch.nn.Linear(int(n_of_channels * current_size ** 2), 1)
            )

        elif layer_name == 'upscale':
            layers.append(
                torch.nn.Upsample(scale_factor=2)
            )

        else:
            raise ValueError

    return layers
