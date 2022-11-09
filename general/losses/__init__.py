from torch import nn

from general.config import cfg

from . import arcloss

LOSSES = {
    "AAM": arcloss.CombinedMarginLoss,
    "CE": nn.CrossEntropyLoss,
}


def make_loss():
    return LOSSES[cfg.LOSS.BODY]()
