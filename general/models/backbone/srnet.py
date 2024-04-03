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
            ConvBNA(odim, odim, activation=nn.Identity),
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
    """This is SRNet model class.
    Default is 

    t1: [64, 16]
    t2: [16, 16, 16, 16, 16]
    t3: [16, 64, 128, 256]
    t4: [512]

    """

    def __init__(self, cfg):
        super().__init__()
        print(cfg.model)
        t1,t2,t3,t4 = cfg.model.t1, cfg.model.t2, cfg.model.t3, cfg.model.t4

        self.block1 = nn.Sequential(
            * [T1(3,t1[0])] + [T1(t1[i], t1[i+1]) for i in range(len(t1)-1)]
        )
        self.block2 = nn.Sequential(
            * [T2(t1[-1],t2[0])] + [T2(t2[i], t2[i+1]) for i in range(len(t2)-1)]
        )
        self.block3 = nn.Sequential(
            * [T3(t2[-1],t3[0])] + [T3(t3[i], t3[i+1]) for i in range(len(t3)-1)]
        )
        self.block4 = nn.Sequential(
            * [T4(t3[-1],t4[0])] + [T4(t4[i], t4[i+1]) for i in range(len(t4)-1)]
        )

        self.seq = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )
        self.dense = nn.Linear(t4[-1], cfg.model.odim)
        self.softmax = nn.LogSoftmax(dim=1)

    def embed(self, x):
        x = self.seq(x)
        return x

    def forward(self, x):
        x = self.embed(x)
        return self.softmax(self.dense(x.view(x.size(0), -1)))


import hydra
@hydra.main(config_path="../../../config", config_name="main")
def main(cfg):
    image = torch.randn((1, 3, 256, 256))
    net = SRNet(cfg)
    print(net)
    print(net(image).shape)

if __name__ == "__main__":
    main()
