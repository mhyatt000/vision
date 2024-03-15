import itertools
import json
import os
import random
import time

import numpy as np
import torch
from torch import distributed as dist

from general.data.loader import build_loaders
from general.models import build_model
from general.results import out
from general.trainer import Trainer

from .tester import Tester


def setup_seed(cfg, deterministic=True):
    torch.manual_seed(cfg.exp.seed)
    torch.cuda.manual_seed_all(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    random.seed(cfg.exp.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.exp.seed)

    # deterministic is slower, more reproducible
    if cfg.util.machine.dist:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = deterministic


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            return


class Experiment:
    """docstring"""

    def __init__(self, cfg):
        self.cfg = cfg
        mkdir(cfg.exp.out)
        print("init exp")

        self.trainer = Trainer(self.cfg)
        self.tester = Tester(  # self.trainer.__dict__
            self.trainer.model,
            self.trainer.loaders,
            criterion=self.trainer.criterion,
        )

    def run(self):
        if self.cfg.exp.train:
            self.trainer.run()
        if self.cfg.exp.test:
            self.tester.run()


EXPERIMENTS = {
    "base": Experiment,
    # "series": SeriesExperiment,
}


def build_experiment(cfg):
    print(cfg.exp)
    return EXPERIMENTS[cfg.exp.body](cfg)
