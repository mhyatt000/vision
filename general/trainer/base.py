import os
import time

import matplotlib.pyplot as plt
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint_sequential

from general.data import build_loaders
from general.helpers import Checkpointer, Stopper, make_scheduler
from general.losses import make_loss
from general.models import build_model
from general.optim import make_optimizer
from general.results import PlotManager, out
from general.toolbox import gpu, tqdm


def gather(x):
    """simple all gather manuver"""

    if not cfg.util.machine.dist:
        return x

    _gather = [
        torch.zeros(x.shape, device=cfg.rank) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Trainer:
    """manages and abstracts options from the training loop"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterion = make_loss(cfg)

        params = [{"params": self.model.parameters()}]
        if cfg.loss.body in ["ARC", "PFC"]:
            params.append({"params": self.criterion.parameters()})

        self.optimizer = make_optimizer(cfg, params)
        self.scheduler = make_scheduler(cfg, self.optimizer)
        self.scaler = GradScaler(growth_interval=100)  # default is 2k
        self.ckp = Checkpointer(cfg, self)
        self.stopper = Stopper(cfg.solver.patience)

        self.loaders = build_loaders(cfg)
        self.loader = self.loaders["train"]

        self.clip = lambda: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        # loader dataset is a subset of the dataset class
        self.plot = PlotManager(cfg, classes=self.loader.dataset.dataset.classes)

        """ what is ema -> exponential moving average """

        # state
        self.epoch, self.nstep = 0, 0
        self.losses, self.accs, self.lrs = [], [], []
        self.best_epoch = 0

        print(gpu.gpu_utilization())
        print("init trainer")
        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update_step(self):
        """docstring"""
        self.nstep += 1

        if self.nstep % self.cfg.solver.checkpoint_period == 0:
            self.plot.show_loss(self.losses, lr=self.lrs)
            self.plot.show_accuracy(self.accs)
            self.ckp.save()

    def update_epoch(self):
        """update after the training loop like housekeeping"""

        if self.cfg.solver.scheduler.step_by == "EPOCH":
            self.step_scheduler(self.loss)
        self.epoch += 1

    def calc_accuracy(self, Yh, Y):
        if self.cfg.loss.body != "ce":
            # find the argmax of the output (Yh) and compare to the target (Y)
            acc = float(
                (torch.argmax(Yh, dim=1) == torch.argmax(Y, dim=1)).sum() / Yh.shape[0]
            )
            self.accs.append(acc)
            return
        with torch.no_grad():
            acc = float(
                (torch.argmax(Yh, dim=1) == torch.argmax(Y, dim=1)).sum() / Yh.shape[0]
            )
            self.accs.append(acc)

    def back_pass(self, loss):
        if self.cfg.solver.amp:
            self.scaler.scale(loss).backward()
            # self.scaler.scale(loss.to(cfg.rank)).backward()
        else:
            loss.to(self.cfg.rank).backward()

    def backpropagation(self):
        if self.cfg.solver.amp:
            self.scaler.unscale_(self.optimizer)  # must unscale before clipping
        self.clip()
        if self.cfg.solver.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def step_scheduler(self, loss):
        """docstring"""
        if self.cfg.solver.scheduler.body == "PATIENT":
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def step(self, batch):
        """training step with adaptive gradient accumulation"""

        # @gpu.timer()
        # def sendit(X,Y):
        # return X , Y
        # X,Y = sendit(X,Y)

        todev = lambda a: a.to(self.cfg.rank, non_blocking=True)

        x = todev(batch['x'])
        label = todev(batch['label'])

        output = self.model(x)
        loss = self.criterion(output, label)
        # dist.all_reduce(loss, op=dist.ReduceOp.MAX)

        self.calc_accuracy(output, label)

        loss /= self.cfg.solver.grad_acc_every
        self.back_pass(loss)
        self.loss = float(loss.detach())

        if self.cfg.solver.scheduler.step_by == "ITER":
            self.step_scheduler(self.loss)

        self.losses.append(self.loss)
        self.lrs.append(self.get_lr())

        # only update every k steps
        if self.nstep % self.cfg.solver.grad_acc_every:
            return
        else:
            self.backpropagation()

    def get_lr(self):
        """docstring"""
        return (
            self.scheduler._last_lr[0]
            if "_last_lr" in self.scheduler.__dict__
            else self.cfg.solver.optim.lr
        )

    def display(self):
        """displays training status"""

        return " | ".join(
            [
                f"loss: {self.loss:.4f}",
                f"best: {self.stopper.get_best():.4f}",
                f"accuracy: {self.accs[-1]:.2f}",
                f"patience: {self.stopper.get_patience()}",
                f"lr: {self.get_lr():.2e}",
                # f'lr: {self.optimizer.param_groups[0]['lr']:.2e }',
                f"amp: {self.scaler.get_scale():.1e}",
                f"{gpu.gpu_utilization()}",
            ]
        )

    def run(self):
        """trains model"""

        print("begin training loop...")

        steps_left = self.cfg.solver.max_iter - self.nstep

        # @torch.autocast(cfg.solver.amp)
        @tqdm.prog(self.cfg, steps_left)
        def _step(batch):
            self.step(batch)
            return self.display()

        # for epoch in range(self.epoch, cfg.solver.MAX_EPOCH-1):
        while self.nstep < self.cfg.solver.max_iter:
            if self.cfg.util.machine.dist:
                self.loader.sampler.set_epoch(self.epoch)
            for batch in self.loader:
                _step(batch)
                self.update_step()
                if self.stopper(self.losses[-1]):
                    return
            self.update_epoch()
