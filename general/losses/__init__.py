from general.config import cfg
from torch import nn

from . import arcloss
from .partial_fc_v2 import PartialFC_V2 as PFC

LOSSES = {
    "AAM": PFC,  # arcloss.CombinedMarginLoss, implements AAM
    "CE": nn.CrossEntropyLoss,
}


def make_loss():
    l = LOSSES[cfg.LOSS.BODY]()
    if cfg.LOSS.BODY == "AAM":
        l.train().to(cfg.DEVICE)
    return l
