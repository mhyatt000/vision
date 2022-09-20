import warnings

from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from tqdm import tqdm

from argparse import ArgumentParser as AP

warnings.filterwarnings("ignore")


class G_block(nn.Module):
    def __init__(self, ic=3, oc=64, kernel_size=4, strides=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)

        self.t_conv2d = nn.ConvTranspose2d(ic, oc, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(oc)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.t_conv2d(X)))


class D_block(nn.Module):
    def __init__(self, ic=3, oc=64, kernel_size=4, strides=2, padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)

        self.conv2d = nn.Conv2d(ic, oc, kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(oc)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


class Generator(nn.Module):
    """default generator"""

    def __init__(self, *, c=64):

        super(Generator, self).__init__()

        self.main = nn.Sequential(
            G_block(ic=100, oc=c * 8, strides=1, padding=0),
            G_block(ic=c * 8, oc=c * 4),
            G_block(ic=c * 4, oc=c * 2),
            G_block(ic=c * 2, oc=c),
            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def weight_init(self):
        """docstring"""
        pass

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """default discriminator"""

    def __init__(self, *, c=64):

        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            D_block(ic=3, oc=c ),
            D_block(ic=c, oc=c * 2),
            D_block(ic=c * 2, oc=c * 4),
            D_block(ic=c * 4, oc=c * 8),
            nn.Conv2d(c * 8, 1, 4, 1, 0,  bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    def weight_init(self):
        """docstring"""
        pass
