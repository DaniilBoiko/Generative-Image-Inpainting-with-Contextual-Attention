import torch
from torch import nn

from .utils import build_layers


class CoarseNetwork(nn.Module):
    def __init__(self, config):
        super(CoarseNetwork, self).__init__()
        self.layers = nn.Sequential(**config)

    def forward(self, x):
        x = self.layers(x)
        return torch.clamp(x, -1, 1)


class RefinmentNetwork(nn.Module):
    def __init__(self, config):
        super(RefinmentNetwork, self).__init__()

        self.layers = {}
        for group_name, sublayers in config.items():
            self.layers[group_name] = nn.Sequential(**build_layers(sublayers))

    def forward(self, x):
        conv_x = self.layers['Convolutional'](x)
        atten_x = self.layers['Attention'](x)

        x = torch.cat([conv_x, atten_x], dim=1)
        x = self.layers['Both'](x)
        x = torch.clamp(x, -1., 1.)

        return x


class LocalCritic(nn.Module):
    def __init__(self, config):
        super(LocalCritic, self).__init__()
        self.layers = build_layers(config)


class GlobalCritic(nn.Module):
    def __init__(self, config):
        super(GlobalCritic, self).__init__()
        self.layers = build_layers(config)
