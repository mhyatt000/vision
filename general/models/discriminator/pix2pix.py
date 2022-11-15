# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
# Defines the PatchGAN discriminator with the specified arguments.

from general.config import cfg
import torch
from torch import nn


class NLayerDiscriminator(nn.Module):
    def __init__( self, inc, onc=64, nlayers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, get_feat=False):
        super(NLayerDiscriminator, self).__init__()

        self.get_feat = get_feat
        self.nlayers = nlayers

        kw = 4  # kernel width
        pw = int(np.ceil((kw - 1.0) / 2))  # padding width
        conv = lambda i, o, s: nn.Conv2d(i, o, kernel_size=kw, stride=s, padding=pw)

        def mk_layer(i, o, s, norm=True):
            """abstracts layer building"""
            return [
                conv(i, o, s),
                norm_layer(o) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        layers = [mk_layer(inc, onc, 2, norm=False)]

        for _ in range(nlayers - 1):
            inc, onc = onc, min(onc * 2, 512)
            layers += [mk_layer(inc, onc, 2)]

        inc, onc = onc, min(onc * 2, 512)
        layers += [mk_layer(inc, onc, 1)]

        layers += [[conv(onc, 1, 1)]]

        if use_sigmoid:
            layers += [[nn.Sigmoid()]]

        if get_feat:
            for n, block in enumerate(layers):
                setattr(self, "block" + str(n), nn.Sequential(*block))
        else:
            stream = []
            for n, block in enumerate(layers):
                stream += block
            self.model = nn.Sequential(*stream)

    def forward(self, X):

        if self.get_feat:
            res = [X]
            for n in range(self.nlayers + 2):
                model = getattr(self, "block" + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(X)
