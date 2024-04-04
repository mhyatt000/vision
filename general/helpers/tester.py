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
from general.results import PlotManager, out
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
        self.testloader = self.loaders["test"]

        # extract from DDP
        self.criterion = trainer.criterion
        self.criterion = (
            self.criterion.module
            if "module" in self.criterion.__dict__
            else self.criterion
        )
        self.plot = trainer.plot

    def embed(self, testloader):
        """docstring"""

        allY, allYh = [], []
        allembed= []

        # TODO: fix this so its clean in toolbox.tqdm.py
        # decorator was to fix printnode problem but its clunky
        @tqdm.prog(self.cfg, len(testloader), desc="Embed")
        def _embed(batch):

            todev = lambda a: a.to(cfg.rank, non_blocking=True)

            x = todev(batch['x'])
            label = todev(batch['label'])

            with torch.no_grad():
                Yh = self.model(x).view((label.shape[0], -1))
                embed = self.model.embed(x).view((label.shape[0], -1))
                embed = F.normalize(Yh)

                if self.cfg.util.machine.dist:
                    label, Yh, embed = gather(label, cfg.rank), gather(Yh, cfg.rank), gather(embed, cfg.rank)
                allY.append(label.cpu())
                allYh.append(Yh.cpu())
                allembed.append(embed.cpu())

        for batch in testloader:
            _embed(batch)

        torch.cuda.empty_cache()
        allout = {
            "Y": torch.cat(allY),
            "Yh": torch.cat(allYh),
            "embed": torch.cat(allembed),
        }
        return allout

    def get_centers(self):
        """get learned cls centers"""
        return F.normalize(self.criterion.weight).detach().cpu()

    def run(self):
        """docstring"""

        print("begin eval loop...")

        self.model.eval()

        # kwargs = { "rknns": rknns }
        kwargs = {
            "testloader": self.testloader,
            "model": self.model,
        }

        # Y, Yh = self.embed(self.trainloader)
        # rknns = self.plot.calcs["rknn"](Y, Yh)  # rknn centers depend on train data

        allout = self.embed(self.testloader)

        """
        if self.cfg.loss.body in ["ARC", "PFC"]:
            kwargs["centers"] = self.get_centers()
        if self.cfg.loss.body == "ARC":
            logits = (
                self.criterion.apply_margin(Yh.to(self.cfg.rank), Y.to(self.cfg.rank))
                .detach()
                .cpu()
            )
            kwargs["logits"] = logits
        """

        if self.cfg.master:
            self.plot(**allout, **kwargs)
            done = torch.tensor([1], dtype=torch.int)  # 1 indicates completion
        else:
            done = torch.tensor([0], dtype=torch.int)

        if self.cfg.util.machine.dist:
            dist.broadcast(done, src=0)
            if done.item() == 1:  # all nodes wait until master is done
                pass
