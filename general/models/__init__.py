from general.config import cfg
from torch.nn.parallel import DistributedDataParallel as DDP

from . import backbone, head, lang, layers, rpn
from .backbone import resnet, swint, vit, ffcresnet, iresnet
from .vlrcnn import VLRCNN

models = {
    "VLRCNN": VLRCNN,
    "SWINT": swint.SwinTransformer,
    "RESNET": resnet.ResNet,
    "VIT": vit.VIT,
    "FFCR": ffcresnet.FFCR,
    "IRESNET": iresnet.IResNet,
}


def build_model():
    model = models[cfg.MODEL.BODY]().to(cfg.rank)

    if cfg.distributed:
        model = DDP(
            model,
            device_ids=[cfg.rank],
            output_device=cfg.rank,
            # broadcast_buffers=False,
        )

    if cfg.EXP.TRAIN:
        model.train()
    return model
