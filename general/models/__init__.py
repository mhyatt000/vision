from general.config import cfg
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from . import backbone, head, layers, rpn # lang
from .backbone import ffcresnet, iresnet, resnet, swint, vit, srnet, resnet_from_scratch
from .vlrcnn import VLRCNN
from .layers import basic
from . import custom 

def sequence():
    """allows user to define a sequence of models"""
    return nn.Sequential(*[models[mod]() for mod in cfg.MODEL.SEQ.SEQUENCE])

def CUSTOM():
    return custom.CUSTOM[cfg.MODEL.CUSTOM.KEY]()


models = {
    "VLRCNN": VLRCNN,
    "SWINT": swint.SwinTransformer,
    "RESNET": resnet.ResNet, 
    "RESNET_FROM_SCRATCH": resnet_from_scratch.ResNet, 
    "VIT": vit.VIT,
    "FFCR": ffcresnet.FFCR,
    "IRESNET": iresnet.IResNet,
    "SEQUENTIAL": sequence,
    "MLP" : basic._MLP,
    "SELECT" : basic.Select,
    "CONV" : basic.CONV2D,
    "SRNET": srnet.SRNet,
    "CUSTOM": CUSTOM,
}


def build_model():
    model = models[cfg.MODEL.BODY]().to(cfg.rank)

    if cfg.distributed:
        model = DDP(
            model,
            device_ids=[cfg.rank],
            output_device=cfg.rank,
            # broadcast_buffers=False,
            # find_unused_parameters=True,
        )

    if cfg.EXP.TRAIN:
        model.train()
    return model
