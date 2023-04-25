from general.config import cfg

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

from .arcloss import CombinedMarginLoss
from .pfc import PFC
from .arc2 import ArcFace

LOSSES = {
    "ARC": ArcFace, # PFC,
    "CE": nn.CrossEntropyLoss,
}


def make_loss():
    print(cfg.LOSS)
    print()

    loss = LOSSES[cfg.LOSS.BODY]()
    if cfg.LOSS.BODY in ["ARC"]:
        loss.to(cfg.rank).train()

        if cfg.distributed:
            loss = DDP( loss, device_ids=[cfg.rank])
            loss.train()

    return loss
