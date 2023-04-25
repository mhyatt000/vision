import os
from general.results import plot
from general.toolbox import tqdm

import matplotlib.pyplot as plt

from general.config import cfg
from general.data import build_loaders
from general.helpers import Checkpointer, make_scheduler
from general.losses import make_loss
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn.functional as F


def gather(x):
    """simple all gather manuver"""

    if not cfg.distributed:
        return x

    _gather = [torch.zeros(x.shape, device=cfg.rank) for _ in range(dist.get_world_size())]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Tester:
    """manages and abstracts options from evaluation"""

    def __init__(self, model, loader, trainloader, *, criterion):
        # essentials
        self.model = model
        self.loader = loader
        self.trainloader = trainloader
        self.criterion = criterion

    def embed(self, loader):
        """docstring"""

        allY, allYh = [], []

        # TODO: fix this so its clean in toolbox.tqdm.py
        # decorator was to fix printnode problem but its clunky
        @tqdm.prog(len(loader), desc="Embed")
        def _embed(X, Y):

            X = X.to(cfg.rank, non_blocking=True)
            Y = Y.to(cfg.rank, non_blocking=True)

            with torch.no_grad():
                Yh = self.model(X).view((Y.shape[0], -1))
                Yh = F.normalize(Yh)
                Y, Yh = gather(Y), gather(Yh)
                allY.append(Y.cpu())
                allYh.append(Yh.cpu())

        for X, Y in loader:
            _embed(X, Y)

        torch.cuda.empty_cache()
        return torch.cat(allY), torch.cat(allYh)

    def get_centers(self):
        """get learned cls centers"""
        return F.normalize(self.criterion.module.weight).detach().cpu()

    def run(self):
        """docstring"""

        self.model.eval()
        Y, Yh = self.embed(self.trainloader)
        rknns = plot._RKNN(Y, Yh)

        Y, Yh = self.embed(self.loader)
        logits = self.criterion.module.apply_margin(Yh.to(cfg.rank),Y.to(cfg.rank)).detach().cpu()

        kwargs = {
            "rknns": rknns,
            "centers": self.get_centers(),
            "logits": logits,
        }

        if  cfg.master:
            for p in cfg.EXP.PLOTS:
                plot.PLOTS[p](Y, Yh, **kwargs)

