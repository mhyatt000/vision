import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler

from general.data import build_loaders
from general.helpers import Checkpointer, make_scheduler
from general.losses import make_loss
from general.results import out, plot
from general.toolbox import tqdm


def gather(x, device):
    """simple all gather manuver"""

    _gather = [
        torch.zeros(x.shape, device=device) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Tester:
    """manages and abstracts options from evaluation"""

    def __init__(self, cfg, trainer):
        # essentials
        self.cfg = cfg
        self.model = trainer.model
        self.loaders = trainer.loaders
        self.trainloader = self.loaders["train"]

        # extract from DDP
        self.criterion = trainer.criterion
        self.criterion = (
            self.criterion.module
            if "module" in self.criterion.__dict__
            else self.criterion
        )
        self.plot = trainer.plot

    def embed(self, loader):
        """docstring"""

        allY, allYh = [], []

        # TODO: fix this so its clean in toolbox.tqdm.py
        # decorator was to fix printnode problem but its clunky
        @tqdm.prog(self.cfg, len(loader), desc="Embed")
        def _embed(X, Y):
            X = X.to(self.cfg.rank, non_blocking=True)
            Y = Y.to(self.cfg.rank, non_blocking=True)

            with torch.no_grad():
                Yh = self.model(X).view((Y.shape[0], -1))
                Yh = F.normalize(Yh)
                if self.cfg.util.machine.dist:
                    Y, Yh = gather(Y, cfg.rank), gather(Yh, cfg.rank)
                allY.append(Y.cpu())
                allYh.append(Yh.cpu())

        for X, Y in loader:
            _embed(X, Y)

        torch.cuda.empty_cache()
        return torch.cat(allY), torch.cat(allYh)

    def get_centers(self):
        """get learned cls centers"""
        return F.normalize(self.criterion.weight).detach().cpu()

    def run(self):
        """docstring"""

        print("begin eval loop...")

        self.model.eval()
        Y, Yh = self.embed(self.trainloader)
        rknns = self.plot.calcs["rknn"](Y, Yh)

        Y, Yh = self.embed(self.loader)

        kwargs = {
            "rknns": rknns,
        }

        if cfg.loss.body in ["ARC", "PFC"]:
            kwargs["centers"] = self.get_centers()
        if cfg.loss.body == "ARC":
            logits = (
                self.criterion.apply_margin(Yh.to(cfg.rank), Y.to(cfg.rank))
                .detach()
                .cpu()
            )
            kwargs["logits"] = logits

        folder = out.get_path()

        # while there's not a png for all plots...
        # keeps dist.barrier() from timing out
        while not all(
            [
                any([p in x for x in os.listdir(folder)])
                for p in [p.lower() for p in cfg.exp.plots]
            ]
        ):
            if cfg.master:
                for p in cfg.exp.plots:
                    self.plot.plots[p](Y, Yh, **kwargs)
            else:
                time.sleep(2)
