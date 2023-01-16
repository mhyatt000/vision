import os

import matplotlib.pyplot as plt
from tqdm import tqdm

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


printnode = not (cfg.rank and cfg.distributed)


def prog(length):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "tqdm"):
                wrapper.tqdm = tqdm(total=length) #, leave=False)
            result = func(*args, **kwargs)
            wrapper.tqdm.set_description(result)
            wrapper.tqdm.update()

            return result

        return wrapper if printnode else func  # no tqdm if not printnode

    return decorator


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
        self.amp = GradScaler(growth_interval=100)  # default is 2k
        self.ckp = Checkpointer(self)

        self.loader = loader
        # self.loaders = build_loaders()

        """what is ema"""

        # TODO: you need to make this into an object to handle it all nicely
        # early stopping

        # self.use_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1
        # self.patience = 0
        # self.max_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE
        # self.best = 0.0
        # self.best_epoch = 0

        # self.use_decay = cfg.SOLVER.WEIGHT_DECAY_SCHEDULE
        # self.milestone_tgt = 0

        # state
        self.epoch, self.batch = 0, 0
        self.losses, self.accs = [], [0]
        self.best_epoch = 0

        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update(self):
        """update after the training loop like housekeeping"""

        self.patient()
        self.ckp.save()

        self.epoch += 1
        self.batch = 0

    def patient(self):
        """given the eval result should we terminate the loop"""

        return

        """
        if self.accs[-1] < self.best:
            self.patience += 1
        else:
            self.patience = 0
            self.best = self.accs[-1]
            self.best_epoch = self.epoch
            self.ckp.save()
        if self.use_patience and self.patience >= self.max_patience:
            print()
            print(f"Auto Termination at {self.epoch}, current best {self.best}")
            quit()
        """

    """
    def init_decay():
        # Adapt the weight decay
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, "milestones"):
            milestone_target = 0
            for i, milstone in enumerate(list(scheduler.milestones)):
                if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                    milestone_target = i + 1
    """

    def step(self, X, Y):

        Yh = self.model(X)
        print(Yh.shape)
        quit()
        loss = self.criterion(Yh, Y)
        clip = lambda : torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        if cfg.AMP:
            self.amp.scale(loss.to(cfg.DEVICE)).backward()
            self.amp.unscale_(self.optimizer)
            clip()
            self.amp.step(self.optimizer)
            self.amp.update()
            self.optimizer.zero_grad()

        else:
            loss.backward()
            clip()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()

        # TODO: make robust ... if amp then use amp else regular
        # TODO make msg messenger obj to handle reporting
        # and documenting (maybe a graph?)

        self.loss = float(loss)
        self.losses.append(self.loss)
        self.scheduler.step()

    def run(self):
        """trains model"""

        print("begin training loop...")
        # loader = self.loaders["train"]

        nsteps = (cfg.SOLVER.MAX_EPOCH - self.epoch)*len(self.loader)

        # @torch.autocast(cfg.AMP)
        @prog(nsteps)
        def _step(X, Y):
            self.step(X, Y)
            desc = f'{self.epoch}/{cfg.SOLVER.MAX_EPOCH} | loss: {self.loss:.4f} | lr: {self.scheduler.get_last_lr()[0]:.4f} | amp: {self.amp.get_scale():.1e} '
            return desc

        for epoch in range(self.epoch, cfg.SOLVER.MAX_EPOCH):

            if cfg.distributed:
                self.loader.sampler.set_epoch(self.epoch)
            for X, Y in self.loader:
                _step(X, Y)

            self.update()
