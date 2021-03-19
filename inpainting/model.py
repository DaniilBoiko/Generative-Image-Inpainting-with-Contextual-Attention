import torch
from torch import nn

from .utils import build_layers


class CoarseNetwork(nn.Module):
    def __init__(self, config):
        super(CoarseNetwork, self).__init__()
        self.layers = nn.Sequential(*build_layers(config))

    def forward(self, x):
        x = self.layers(x)
        return torch.clamp(x, -1, 1)


class AttentionBranch(nn.Module):
    def __init__(self, config):
        super(AttentionBranch, self).__init__()
        self.layers = build_layers(config)

    def forward(self, x, masks):
        for layer in self.layers:
            if layer.__class__.__name__ != 'ContextualAttention':
                x = layer(x)
            else:
                x = layer(x, x, masks)

        return x


class RefinementNetwork(nn.Module):
    def __init__(self, config):
        super(RefinementNetwork, self).__init__()

        self.layers = {}
        for group_name, sublayers in config.items():
            if group_name != 'both':
                self.layers[group_name] = nn.Sequential(*build_layers(sublayers))

        self.layers['Convolutional'] = nn.Sequential(*build_layers(config['Convolutional']))
        self.layers['Attention'] = AttentionBranch(config['Attention'])

        self.layers['Both'] = nn.Sequential(*build_layers(
            config['Both'],
            in_channels=self.layers['Convolutional']._modules[
                            str(len(self.layers['Convolutional']._modules) - 2)
                        ].out_channels + self.layers['Attention'].layers[-2].out_channels
        ))

    def forward(self, x, mask):
        conv_x = self.layers['Convolutional'](x)
        atten_x = self.layers['Attention'](x, mask)

        x = torch.cat([conv_x, atten_x], dim=1)
        x = self.layers['Both'](x)
        x = torch.clamp(x, -1., 1.)

        return x

class LocalCritic(nn.Module):
    def __init__(self, config):
        super(LocalCritic, self).__init__()
        self.layers = nn.Sequential(*build_layers(config))

    def forward(self, x):
        return self.layers(x)

class GlobalCritic(nn.Module):
    def __init__(self, config):
        super(GlobalCritic, self).__init__()
        self.layers = nn.Sequential(*build_layers(config))

    def forward(self, x):
        return self.layers(x)
