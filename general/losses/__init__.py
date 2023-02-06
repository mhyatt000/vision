from general.config import cfg
from torch import nn

from .arcloss import CombinedMarginLoss
from .partial_fc_v2 import PartialFC_V2 as PFC2


class PFC(PFC2):
    def __init__(self):
        super(PFC,self).__init__(
            margin_loss=CombinedMarginLoss(),
            embedding_size=cfg.LOSS.PFC.EMBED_DIM,
            num_classes=cfg.LOSS.PFC.NCLASSES,
            sample_rate=cfg.LOSS.PFC.SAMPLE_RATE,
            fp16=cfg.AMP,
        )


LOSSES = {
    "AAM": PFC,  # arcloss.CombinedMarginLoss, implements AAM
    "CE": nn.CrossEntropyLoss,
}


def make_loss():
    loss = LOSSES[cfg.LOSS.BODY]()
    if cfg.LOSS.BODY == "AAM":
        loss.to(cfg.DEVICE).train()
    return loss
