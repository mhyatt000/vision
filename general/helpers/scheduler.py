from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR, _LRScheduler
from general.config import cfg


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):

        nbatch = cfg.LOADER.SIZE // (cfg.LOADER.BATCH_SIZE * cfg.world_size)
        self.max_steps = nbatch * cfg.SOLVER.MAX_EPOCH 
        self.warmup_steps = nbatch * cfg.SCHEDULER.WARMUP

        self.base_lr = cfg.OPTIM.LR
        self.warmup_lr_init = 0.0001
        self.power = 2

        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):

        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()

        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


def make_scheduler(optimizer):

    schedulers = {
        "Step": lambda optim: StepLR(optim, step_size=1, gamma=0.95),
        "Poly": PolyScheduler,
    }

    return schedulers[cfg.SCHEDULER.BODY](optimizer)


# 1/10 every 10 epochs is 0.8
# 1/3 every 10 epochs is 0.9