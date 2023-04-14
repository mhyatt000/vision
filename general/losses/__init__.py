from general.config import cfg
from torch import nn

from .arcloss import CombinedMarginLoss
from .pfc import PFC
from .angular_sm import AngularPenaltySM
from .arc2 import ArcFace

LOSSES = {
    "ARC": ArcFace, # PFC,
    "ANGULAR_SM": AngularPenaltySM,
    "CE": nn.CrossEntropyLoss,
}


def make_loss():
    loss = LOSSES[cfg.LOSS.BODY]()
    if cfg.LOSS.BODY in ["ARC", "ANGULAR_SM"]:
        loss.to(cfg.rank).train()
    return loss
