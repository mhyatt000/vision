import os

from general.results import out as results
from general.config import cfg
import torch


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            return


class Checkpointer:
    """
    manages and saves snapshots
    """

    def __init__(self, trainer):

        self.trainer = trainer
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.scheduler = self.trainer.scheduler
        self.criterion = self.trainer.criterion

        # paths
        self.psnap = os.path.join(results.get_path(), "snapshot.pt")

        # state
        self.remember = [
            "nstep",
            "epoch",
            "losses",
            "accs",
            # "best",
            # "best_epoch",
        ]

        self.states = [
            "criterion",
            # "optimizer",
            "scaler",
            "scheduler",
        ]

    def save(self):
        if cfg.rank or (self.trainer.epoch % cfg.SOLVER.CHECKPOINT_PER_EPOCH):
            return

        snap = {a.upper(): getattr(self.trainer, a) for a in self.remember}
        snap.update(
            {a.upper(): getattr(self.trainer, a).state_dict() for a in self.states}
        )
        mod = self.trainer.model.module if cfg.distributed else self.trainer.model
        snap["MODEL"] = mod.state_dict()
        torch.save(snap, self.psnap)

    def load(self):

        if not os.path.exists(self.psnap):
            print(f"No snapshot found for {self.psnap}")
            mkdir(results.get_path())
            return

        print("Loading Snapshot")

        snap = torch.load(self.psnap)

        mod = self.trainer.model.module if cfg.distributed else self.trainer.model
        mod.load_state_dict(snap["MODEL"])
        for k, v in snap.items():
            if k.lower() in self.remember + self.states:
                if not type(v) in [list, int]:  # try:
                    temp = getattr(self.trainer, k.lower())
                    temp.load_state_dict(v)
                    setattr(self.trainer, k.lower(), temp)
                else:  # except Exception as ex:
                    # raise ex
                    setattr(self.trainer, k.lower(), v)
        del snap

        print(f"Resuming training from snapshot at step {self.trainer.nstep+1}")
        print(self.psnap)
