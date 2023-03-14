from general.config import cfg
from general.data.loader import build_loaders
from general.models import build_model
from .tester import Tester
from .trainer import Trainer


class Experiment:
    """docstring"""

    def __init__(self):
        print("init exp")
        model = build_model()
        print("model is built")
        loaders = build_loaders()
        print("loaders are built")
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
        pass

    # TODO: can you generalize for many iterations of any hparam? ie: LO
    def run(self):
        for seed in range(5):
            for swap in [0, 1]:
                cfg.SEED = seed
                cfg.LOADER.SWAP = swap
                setup_seed(cfg.SEED)
                super().__init__()
                super().run()


EXPERIMENTS = {
    "DEFAULT": Experiment,
    "5x2": Split5x2Experiment,
}


def build_experiment():
    print(cfg.EXP)
    return EXPERIMENTS[cfg.EXP.BODY]()
