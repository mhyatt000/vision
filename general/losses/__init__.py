from general.config import cfg

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

# from .arcloss import CombinedMarginLoss
# from .pfc import PFC
from .arc2 import ArcFace

from .pfc_original import PartialFC_V2
from .arc_original import CombinedMarginLoss

class PFC(PartialFC_V2):
    def __init__(self):
        super(PFC, self).__init__(
            margin_loss=CombinedMarginLoss(),
            embedding_size=cfg.LOSS.PFC.EMBED_DIM,
            num_classes=cfg.LOSS.PFC.NCLASSES,
            sample_rate=cfg.LOSS.PFC.SAMPLE_RATE,
            fp16=cfg.AMP,
        )


LOSSES = {
    "PFC": PFC,
    "ARC": ArcFace, 
    "CE": nn.CrossEntropyLoss,
}

def make_loss():
    print(cfg.LOSS)
    print()

    loss = LOSSES[cfg.LOSS.BODY]()
    if cfg.LOSS.BODY in ["ARC","PFC"]:
        loss.to(cfg.rank).train()

        if cfg.distributed and cfg.LOSS.BODY == 'ARC':
            loss = DDP( loss, device_ids=[cfg.rank])
            loss.train()

    return loss
