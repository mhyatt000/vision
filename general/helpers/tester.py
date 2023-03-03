import os
from general.results import plot
from general.toolbox import tqdm 

import matplotlib.pyplot as plt

from general.config import cfg
from general.data import build_loaders, build_loaderx
from general.helpers import Checkpointer, make_optimizer, make_scheduler
from general.losses import make_loss
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler


def gather(x):
    """simple all gather manuver"""

    if not cfg.distributed:
        return x

    _gather = [torch.zeros(x.shape, device=cfg.DEVICE) for _ in range(dist.get_world_size())]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)

class Tester:
    """manages and abstracts options from evaluation"""

    def __init__(self, model, loader, trainloader):

        # essentials
        self.model = model
        self.loader = loader
        self.trainloader = trainloader

    def embed(self, loader):
        """docstring"""

        allY, allYh = [],[]

        # TODO: fix this so its clean in toolbox.tqdm.py
        # decorator was to fix printnode problem but its clunky
        @tqdm.prog(len(loader), desc="Embed")
        def _embed(X,Y):
            X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
            Yh = self.model(X).view((Y.shape[0], -1))
            Y, Yh = gather(Y), gather(Yh)
            allY.append(Y)
            allYh.append(Yh)

        with torch.no_grad():
            for X, Y in loader:
                _embed(X,Y)

        return torch.cat(allY), torch.cat(allYh)

    def run(self):
        """docstring"""

        Y,Yh = self.embed(self.trainloader)
        centers = plot.make_centers(Y,Yh)

        Y,Yh = self.embed(self.loader)
        for p in cfg.EXP.PLOTS:
            plot.PLOTS[p](Y,Yh)
        
