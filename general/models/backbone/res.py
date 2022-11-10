import torch
from torch import nn
from general.config import cfg


class Skip(nn.Module):
    def __init__(self, c, pointwise=False, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(c, kernel_size=3, strides=strides)
        self.conv2 = nn.Conv2d(c, kernel_size=3)
        self.conv3 = nn.Conv2d(c, kernel_size=1, strides=strides) if pointwise else None

        self.bn1 = nn.BatchNorm2d()
        self.bn2 = nn.BatchNorm2d()

    def forward(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.conv3(Y) if self.conv3 else Y
        Y += X
        return nn.ReLU(Y)


class ResBlock(nn.Module):
    def __init__(self, c, depth, first=False, **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        First = lambda: Skip(c, pointwise=True, strides=2)
        self.layers = nn.Sequential([First if not (i or first) else Skip(c) for i in range(depth)])

    def forward(self, X):
        return self.layers(X)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layers = nn.Sequential(
            [
                ResBlock(c, d, first=(not i))
                for i, c, d in enumerate(zip(cfg.RESNET.CHANNELS, cfg.RESNET.DEPTHS))
            ]
        )
        self.layers.append(nn.GlobalAvgPool2d())

    def forward(self, X):
        return self.layers(X)
