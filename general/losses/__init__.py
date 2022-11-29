from torch import nn

from general.config import cfg

from . import arcloss
from .partial_fc_v2 import PartialFC_V2 as PFC

LOSSES = {
    "AAM": PFC, # arcloss.CombinedMarginLoss, implements AAM
    "CE": nn.CrossEntropyLoss,
    "PFC": PFC,
}


def make_loss():
    return LOSSES[cfg.LOSS.BODY]()
