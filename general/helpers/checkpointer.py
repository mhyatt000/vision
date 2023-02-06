import os

from general.config import cfg
import torch


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.system(f"mkdir {path}")
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
        self.path = os.path.join(cfg.ROOT, "experiments", cfg.config_name)
        self.psnap = os.path.join(self.path, "snapshot.pt")

        # state
        self.remember = [
            "epoch",
            "losses",
            "accs",
            # "best",
            # "best_epoch",
        ]
        self.states = [
            "criterion",
            "optimizer",
            "scaler",
            "scheduler",
        ]

    def save(self):
        if cfg.rank or (self.trainer.epoch % cfg.SOLVER.CHECKPOINT_PER_EPOCH):
            return

        snap = {a.upper(): getattr(self.trainer, a) for a in self.remember}
        snap.update({a.upper(): getattr(self.trainer, a).state_dict() for a in self.states})
        mod = self.trainer.model.module if cfg.distributed else self.trainer.model
        snap["MODEL"] = mod.state_dict()
        torch.save(snap, self.psnap)

    def load(self):

        if not os.path.exists(self.psnap):
            print("No snapshot found")
            mkdir(self.path)
            return

        print("Loading Snapshot")

        snap = torch.load(self.psnap)

        mod = self.trainer.model.module if cfg.distributed else self.trainer.model
        mod.load_state_dict(snap["MODEL"])
        del snap["MODEL"]
        for k, v in snap.items():
            try:
                getattr(self.trainer, k.lower()).load_state_dict(v)
            except:
                setattr(self.trainer, k.lower(), v)

        print(f"Resuming training from snapshot at epoch {self.trainer.epoch}")
        print(self.psnap)
