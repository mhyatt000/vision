import torch
from torch import nn

"""
idim: input channels
odim: output channels
"""


class ConvBNA(nn.Module):
    """Conv with Batch Normalization and Activation"""

    def __init__(self, idim, odim, activation=nn.ReLU):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                idim,
                odim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(odim),
            activation(),
        )

    def forward(self, x):
        return self.seq(x)


class T1(ConvBNA):
    """Creates type 1 layer of SRNet."""

    def __init__(self, idim, odim):
        super().__init__(idim, odim)


class T2(nn.Module):
    """Creates type 2 layer of SRNet."""

    def __init__(self, idim, odim):
        super().__init__()

        self.seq = nn.Sequential(
            T1(idim, odim),
            ConvBNA(idim, odim, activation=nn.Identity),
        )

    def forward(self, x):
        return x + self.seq(x)


class T3(nn.Module):
    """Creates type 3 layer of SRNet."""

    def __init__(self, idim, odim):
        super().__init__()

        self.seq = nn.Sequential(
            T1(idim, odim),
            ConvBNA(odim, odim, activation=nn.Identity),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(
                idim,
                odim,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(odim),
        )

    def forward(self, x):
        return self.seq(x) + self.skip(x)


class T4(nn.Module):
    """Creates type 4 layer of SRNet."""

    def __init__(self, idim, odim):
        super().__init__()

        self.seq = nn.Sequential(
            T1(idim, odim),
            ConvBNA(odim, odim, activation=nn.Identity),
            nn.AdaptiveAvgPool2d(output_size=1),
        )

    def forward(self, x):
        return self.seq(x)


class SRNet(nn.Module):
    """This is SRNet model class."""

    def __init__(self, cfg):
        super().__init__()

        self.block1 = nn.Sequential(
            T1(3, 64),
            T1(64, 16),
        )
        self.block2 = nn.Sequential(
            T2(16, 16),
            T2(16, 16),
            T2(16, 16),
            T2(16, 16),
            T2(16, 16),
        )
        self.block3 = nn.Sequential(
            T3(16, 16),
            T3(16, 64),
            T3(64, 128),
            T3(128, 256),
        )
        self.block4 = T4(256, 512)

        self.seq = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )
        self.dense = nn.Linear(512, cfg.model.odim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.seq(x)
        return self.softmax(self.dense(x.view(x.size(0), -1)))


if __name__ == "__main__":
    image = torch.randn((1, 3, 256, 256))
    net = SRNet()
    print(net(image).shape)
