from general.config import cfg

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
    return models[cfg.MODEL.BODY]().to(cfg.rank)
