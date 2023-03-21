from general.config import cfg
import itertools
import json
import numpy as np
import random
import os
import torch
from general.data.loader import build_loaders
from general.models import build_model
from .tester import Tester
from .trainer import Trainer
from general.results import out


def setup_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # deterministic is slower, more reproducible
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

    def __init__(self):
        mkdir(cfg.OUT)
        print("init exp")
        loaders = build_loaders()
        print("loaders are built")
        model = build_model()
        print("model is built")
        self.trainer = Trainer(model, loaders["train"])
        self.tester = Tester(
            self.trainer.model, loaders["test"], trainloader=loaders["train"]
        )

    def run(self):
        if cfg.EXP.TRAIN:
            self.trainer.run()
        if cfg.EXP.TEST:
            self.tester.run()


class Split5x2Experiment(Experiment):
    """5 Seeds of 2-fold cross validation"""

    def __init__(self):
        self.versions = os.path.join(cfg.OUT,'versions.json')
        pass

    def mk_versions(self):
        "allows different versions of the experiment to be saved to different folders"

        mkdir(cfg.OUT)
        if os.path.exists(self.versions): # dont remake if you can start in the middle
            return

        exp = []
        ordered = lambda x: x.reverse() if cfg.EXP.REVERSE else x
        
        LO_combos = [None]
        if cfg.LOADER.LEAVE_N_OUT:
            classes =[ [i for i in range(cfg.LOADER.NCLASSES)] for _ in range(cfg.LOADER.LEAVE_N_OUT)]
            LO_combos = list(itertools.product(*classes))
            LO_combos = [sorted(set(x)) for x in LO_combos]
            temp = []
            for LO in LO_combos:
                if len(LO) == cfg.LOADER.LEAVE_N_OUT and not LO in temp:
                    temp.append(LO)
            LO_combos = temp

        for LO in LO_combos:
            for seed,swap in itertools.product(ordered([0,1,2,3,4]), ordered([False,True])):
                e = {'LO':LO, 'seed':seed, 'swap':swap}
                exp.append(e)
                mkdir(os.path.join(cfg.OUT,out.d2s(e)))

        with open(self.versions,'w') as file:
            json.dump(exp,file)

    def pop_versions(self):
        """pop the first experiment version off the file"""
        if not cfg.world_rank: # only rank 0 pops
            with open(self.versions,'r') as file:
                exp = json.load(file)[1:]
            with open(self.versions,'w') as file:
                json.dump(exp,file)
         

    # TODO: can you generalize for many iterations of any hparam? ie: LO
    def run(self):
        self.mk_versions()

        while True:
            version = out.get_exp_version()
            if not version:
                break
            cfg.SEED = version['seed']
            setup_seed(cfg.SEED)

            super().__init__()
            super().run()
            self.pop_versions()

EXPERIMENTS = {
    "DEFAULT": Experiment,
    "5x2": Split5x2Experiment,
}


def build_experiment():
    print(cfg.EXP)
    return EXPERIMENTS[cfg.EXP.BODY]()
