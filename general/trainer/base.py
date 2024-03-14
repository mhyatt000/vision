import os
import time
from general.toolbox import gpu, tqdm
from general.results import plot

import matplotlib.pyplot as plt

from general.config import cfg
from general.data import build_loaders
from general.helpers import Checkpointer, make_scheduler, Stopper
from general.optim import make_optimizer
from general.losses import make_loss
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint_sequential


def gather(x):
    """simple all gather manuver"""

    if not cfg.distributed:
        return x

    _gather = [torch.zeros(x.shape, device=cfg.rank) for _ in range(dist.get_world_size())]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Trainer:
    """manages and abstracts options from the training loop"""

    def __init__(self):

        self.model = build_model()
        self.criterion = make_loss()

        params = [{"params": model.parameters()}]
        if cfg.LOSS.BODY in ["ARC", "PFC"]:
            params.append({"params": self.criterion.parameters()})

        self.optimizer = make_optimizer(params)
        self.scheduler = make_scheduler(self.optimizer)
        self.scaler = GradScaler(growth_interval=100)  # default is 2k
        self.ckp = Checkpointer(self)
        self.stopper = Stopper()

        self.loaders = build_loaders()
        self.loader = loader['train']

        self.clip = lambda: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        """ what is ema -> exponential moving average """

        # state
        self.epoch, self.nstep = 0, 0
        self.losses, self.accs, self.lrs = [], [], []
        self.best_epoch = 0

        print(gpu.gpu_utilization())
        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update_step(self):
        """docstring"""
        self.nstep += 1

        if self.nstep % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            plot.show_loss(self.losses, lr=self.lrs)
            plot.show_accuracy(self.accs)
            self.ckp.save()

    def update_epoch(self):
        """update after the training loop like housekeeping"""

        if cfg.SCHEDULER.STEP_BY == "EPOCH":
            self.step_scheduler(self.loss)
        self.epoch += 1

    def calc_accuracy(self, Yh, Y):
        if cfg.LOSS.BODY != "CE":
            self.accs.append(-1)
            return
        with torch.no_grad():
            acc = float((torch.argmax(Yh, dim=1) == torch.argmax(Y, dim=1)).sum() / Yh.shape[0])
            self.accs.append(acc)

    def back_pass(self, loss):
        if cfg.AMP:
            self.scaler.scale(loss).backward()
            # self.scaler.scale(loss.to(cfg.rank)).backward()
        else:
            loss.to(cfg.rank).backward()

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

    def step_scheduler(self, loss):
        """docstring"""
        if cfg.SCHEDULER.BODY == "PATIENT":
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def step(self, X, Y):
        """training step with adaptive gradient accumulation"""

        # @gpu.timer()
        # def sendit(X,Y):
        # return X , Y
        # X,Y = sendit(X,Y)

        X = X.to(cfg.rank, non_blocking=True)
        Y = Y.to(cfg.rank, non_blocking=True)

        out_dict = self.model(X)
        embed, output = out_dict.values()

        # Yh = checkpoint_sequential(self.model, 4, X) # gradient checkpointing
        loss = self.criterion(output, Y)
        # dist.all_reduce(loss, op=dist.ReduceOp.MAX)

        self.calc_accuracy(output, Y)
        self.back_pass(loss)

        self.loss = float(loss.detach())

        if cfg.SCHEDULER.STEP_BY == "ITER":
            self.step_scheduler(self.loss)

        self.losses.append(self.loss)
        self.lrs.append(self.get_lr())
        loss /= cfg.SOLVER.GRAD_ACC_EVERY

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

    def get_lr(self):
        """docstring"""
        return self.scheduler._last_lr[0] if '_last_lr' in self.scheduler.__dict__ else cfg.SOLVER.OPTIM.LR

    def display():
        """displays training status"""

        return = " | ".join(
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

        steps_left = cfg.SOLVER.MAX_ITER - self.nstep

        # @torch.autocast(cfg.AMP)
        @tqdm.prog(steps_left)
        def _step(X, Y):
            self.step(X, Y)
            return self.display()

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

        """this was going to be flexible batch size"""
        # except torch.cuda.OutOfMemoryError as ex:
        # raise ex # TODO: shouldnt this work?
        # self.rebuild_loader()
        # self.run()
