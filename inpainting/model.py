import torch
from torch import nn

from .utils import build_layers


class CoarseNetwork(nn.Module):
    def __init__(self, config):
        super(CoarseNetwork, self).__init__()
        self.layers = nn.Sequential(*build_layers(config)[:-1])

    def forward(self, x):
        x = self.layers(x)
        return torch.clamp(x, -1, 1)


class AttentionBranch(nn.Module):
    def __init__(self, config):
        super(AttentionBranch, self).__init__()
        self.layers = torch.nn.ModuleList(build_layers(config))

    def forward(self, x, masks):
        for layer in self.layers:
            if layer.__class__.__name__ != 'ContextualAttention':
                x = layer(x)
            else:
                x = layer(x, x, masks)

        return x

    def cuda(self):
        for layer in self.layers:
            layer.cuda()

            if layer.__class__.__name__ == 'ContextualAttention':
                layer.use_cuda = True


class RefinementNetwork(nn.Module):
    def __init__(self, config):
        super(RefinementNetwork, self).__init__()

        self.layers = torch.nn.ModuleDict({})

        self.layers['Convolutional'] = nn.Sequential(*build_layers(config['Convolutional']))
        self.layers['Attention'] = AttentionBranch(config['Attention'])
        self.layers['Attention'].layers[12].use_cuda=False

        self.layers['Both'] = nn.Sequential(*build_layers(
            config['Both'],
            in_channels=self.layers['Convolutional']._modules[
                            str(len(self.layers['Convolutional']._modules) - 2)
                        ].out_channels + self.layers['Attention'].layers[-2].out_channels
        )[:-1])

    def forward(self, x, mask):
        convolutional_x = self.layers['Convolutional'](x)
        attention_x = self.layers['Attention'](x, mask)

        x = torch.cat([convolutional_x, attention_x], dim=1)
        x = self.layers['Both'](x)
        x = torch.clamp(x, -1, 1.)

        return x

    def cuda(self):
        for _, layer in self.layers.items():
            layer.cuda()


class LocalCritic(nn.Module):
    def __init__(self, config):
        super(LocalCritic, self).__init__()
        self.layers = nn.Sequential(*build_layers(config, input_size=64))

    def forward(self, x):
        return self.layers(x)


class GlobalCritic(nn.Module):
    def __init__(self, config):
        super(GlobalCritic, self).__init__()
        self.layers = nn.Sequential(*build_layers(config))

    def forward(self, x):
        return self.layers(x)
