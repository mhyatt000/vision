from general.config import cfg
from torch import optim

from general.losses import make_loss
# from general.solver import make_lr_scheduler, make_optimizer


OPTIM = {
    "ADAM": optim.Adam,
    "SGD": optim.SGD,
}


def make_optimizer(model):
    return OPTIM[cfg.OPTIM.BODY](
        model.parameters(),
        lr=cfg.OPTIM.LR,
        betas=cfg.OPTIM.BETAS,
    )


class Trainer:
    """manages and abstracts options from the training loop"""

    def __init__(self):

        # optimization
        self.optimizer = make_optimizer(model)
        # self.scheduler = make_lr_scheduler(self.optimizer)
        self.loss = make_loss()

        """what is ema"""

        # early stopping 
        self.use_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1
        self.patience = 0
        self.max_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE
        self.best = 0.0

        self.use_decay = cfg.SOLVER.WEIGHT_DECAY_SCHEDULE
        self.milestone_tgt = 0


        # snapshots and checkpoints
        self.snap = cfg.config_file.replace('configs','snapshots')
        self.ckt = cfg.config_file.replace('configs','checkpoints')


    def update(self):
        """update after the training loop"""

    def is_patient(self, result, *, iteration=None):
        """given the eval result should we terminate the loop"""

        if self.use_patience:
            if result < self.best:
                self.patience += 1
            else:
                self.patience = 0
                self.best = result
                # checkpointer.save("model_best", **arguments)

            print("Previous Best", self.best)
            print("Patience Counter", self.patience)
            print("Eval Result", result)

            if self.patience >= self.max_patience:
                if is_main_process():
                    n = "\n\n\n\n"
                    print(f"{n}Auto Termination at {iteration or 'NA'}, current best {best,n}")
                return False
        return True

    def init_decay():
        """docstring"""

        # Adapt the weight decay
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, "milestones"):
            milestone_target = 0
            for i, milstone in enumerate(list(scheduler.milestones)):
                if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                    milestone_target = i + 1


    def _save_ckp(self, epoch):
        """docstring"""

        if self.gpu_id or (epoch % cfg.save_every):
            return

        ckp = self.model.module if cfg.distributed else self.model
        ckp = ckp.state_dict()
        torch.save(ckp, "ckp.pt")
        print(f'Epoch {epoch} | saved model')

