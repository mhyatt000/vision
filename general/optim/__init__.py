import torch
from torch import optim
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO

from .lamb import LAMB

optimIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "lamb": LAMB,
    "zero": ZeRO,
}


def make_optimizer(cfg, params):
    # TODO: add support for *args where args are objects to be optimized
    # ie: make_optim(*args, params):
    # assert not (args and params)

    print(cfg.solver.optim)

    kwargs = dict(
        params=params,
        lr=cfg.solver.optim.lr,
        weight_decay=float(cfg.solver.optim.decay),
    )

    if cfg.solver.optim.body == "SGD":
        kwargs["momentum"] = cfg.solver.optim.momentum
    if cfg.solver.optim.body == "ADAM":
        kwargs["betas"] = cfg.solver.optim.betas
    if cfg.solver.optim.body == "ZERO":
        kwargs["optimizer_class"] = optim.Adam

    optimizer = optimIZERS[cfg.solver.optim.body](**kwargs)
    return optimizer
