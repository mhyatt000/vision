import os
from general.toolbox import gpu, tqdm
from general.results import plot

import matplotlib.pyplot as plt

from general.config import cfg
from general.data import build_loaders 
from general.helpers import Checkpointer, make_optimizer, make_scheduler, Stopper
from general.losses import make_loss
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler


def gather(x):
    """simple all gather manuver"""

    if not cfg.distributed:
        return x

    _gather = [
        torch.zeros(x.shape, device=cfg.DEVICE) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Trainer:
    """manages and abstracts options from the training loop"""

    def __init__(self, model, loader):

        # essentials
        self.model = model
        self.criterion = make_loss()

        params = [{"params": model.parameters()}]
        if cfg.LOSS.BODY == "AAM":
            params.append({"params": self.criterion.parameters()})

        self.optimizer = make_optimizer(params)
        self.scheduler = make_scheduler(self.optimizer)
        self.scaler = GradScaler(growth_interval=100)  # default is 2k
        self.ckp = Checkpointer(self)
        self.stopper = Stopper()
        self.loader = loader

        self.clip = lambda: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        """what is ema"""

        # state
        self.epoch, self.nstep = 0, 0
        self.losses, self.accs = [], []
        self.best_epoch = 0

        print(gpu.gpu_utilization())
        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update_step(self):
        """docstring"""
        self.nstep += 1

        if self.nstep % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            plot.show_loss(self.losses)
            plot.show_accuracy(self.accs)
            self.ckp.save()

    def update_epoch(self):
        """update after the training loop like housekeeping"""

        self.epoch += 1


    def calc_accuracy(self, Yh, Y):

        if cfg.LOSS.BODY != "CE":
            self.accs.append(-1)
            return
        with torch.no_grad():
            acc = float(
                (torch.argmax(Yh, dim=1) == torch.argmax(Y, dim=1)).sum() / Yh.shape[0]
            )
            self.accs.append(acc)

    def back_pass(self, loss):
        if cfg.AMP:
            self.scaler.scale(loss.to(cfg.DEVICE)).backward()
        else:
            loss.backward()

    def backpropagation(self):
        if cfg.AMP:
            self.scaler.unscale_(self.optimizer)  # must unscale before clipping
        self.clip()
        if cfg.AMP:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

    def step(self, X, Y):
        """training step with adaptive gradient accumulation"""

        Yh = self.model(X)
        loss = self.criterion(Yh, Y)
        self.calc_accuracy(Yh, Y)

        self.loss = float(loss.detach())
        self.losses.append(self.loss)
        loss /= cfg.SOLVER.GRAD_ACC_EVERY

        self.back_pass(loss)

        # only update every k steps
        if self.nstep % cfg.SOLVER.GRAD_ACC_EVERY:
            return
        else:
            self.backpropagation()

    def rebuild_loader(self):
        """rebuild a loader with half the batch_size ... hasnt been working tho"""

        # TODO: batch_size or GPU batch_size?
        # torch.cuda.empty_cache()
        # cfg.LOADER.BATCH_SIZE = cfg.LOADER.BATCH_SIZE // 2
        # cfg.SOLVER.GRAD_ACC_EVERY *= 2
        # self.loader = build_loaders()['train']

        # rebuild the same loader w sam hparam
        # half the batch size

    def run(self):
        """trains model"""

        print("begin training loop...")

        steps_left = cfg.SOLVER.MAX_ITER - self.nstep

        # @torch.autocast(cfg.AMP)
        @tqdm.prog(steps_left)
        def _step(X, Y):
            self.step(X, Y)
            desc = f"loss: {self.loss:.4f} | accuracy: {self.accs[-1]:.2f} | lr: {self.scheduler.get_last_lr()[0]:.2e} | amp: {self.scaler.get_scale():.1e} | {gpu.gpu_utilization()}"
            return desc

        # for epoch in range(self.epoch, cfg.SOLVER.MAX_EPOCH-1):
        while self.nstep < cfg.SOLVER.MAX_ITER:

            if cfg.distributed:
                self.loader.sampler.set_epoch(self.epoch)
            for X, Y in self.loader:
                _step(X, Y)
                self.update_step()
                if self.stopper(self.losses[-1]):
                    return
            self.update_epoch()

        # except torch.cuda.OutOfMemoryError as ex:
        # raise ex # TODO: shouldnt this work?
        # self.rebuild_loader()
        # self.run()
