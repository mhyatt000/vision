from general.config import cfg
from general.data.loader import build_loaders
from general.models import build_model
from .newtrainer import Trainer

#TODO: how to manage evaluations

class Experiment:
    """docstring"""

    def __init__(self):
        model = build_model()
        loaders = build_loaders()
        self.trainer = Trainer(model, loaders['train'])

    def run(self):
        self.trainer.run()


class Split5x2Experiment(Experiment):
    """docstring"""

    def __init__(self):
        super().__init__()

    def run():
        for split in range(5):
            for swap in [0, 1]:
                set_seed(cfg.seed + split)
                model = build_model()
                loaders = build_loaders(swap=swap)
                self.trainer = Trainer(model, loaders['train'])
                super().run()
