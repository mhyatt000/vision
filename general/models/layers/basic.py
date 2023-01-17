""" some basic layers that will make things easier """

import torch
from torch import nn
from general.config import cfg


class Select(nn.Module):
    """
    selects from a list of features
    useful for modules that return a list of features
    but you only want one
    """

    def __init__(self, idx=-1):
        super(Select, self).__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, idim, hdim=None, odim=None, act=nn.GELU, drop=0.0):
        super(MLP, self).__init__()

        odim = odim or idim
        hdim = hdim or idim

        self.module = nn.Sequential(
            nn.Linear(idim, hdim),
            act(),
            nn.Linear(hdim, odim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.module(x)


# TODO: make variants that are cfg compatible and seperable from cfg


class _MLP(MLP):
    def __init__(self):
        super(_MLP, self).__init__(
            idim=cfg.MODEL.MLP.IDIM,
            hdim=cfg.MODEL.MLP.HDIM,
            odim=cfg.MODEL.MLP.ODIM,
            act=nn.GELU,
            drop=cfg.MODEL.MLP.DROPOUT,
        )


class CONV2D(nn.Conv2d):
    def __init__(self):
        super(CONV2D, self).__init__(
            cfg.MODEL.CONV.ICHANNEL,
            cfg.MODEL.CONV.OCHANNEL,
            cfg.MODEL.CONV.KERNEL,
        )
        self.reshape = (
            (lambda x: x.view(x.shape[0], -1)) if cfg.MODEL.CONV.RESHAPE else nn.Identity()
        )

    def forward(self, x):
        return self.reshape(super().forward(x))
