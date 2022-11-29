from general.config import cfg

from . import backbone, head, lang, layers, rpn
from .backbone import resnet, swint, vit, ffc_resnet
from .vlrcnn import VLRCNN

models = {
    "VLRCNN": VLRCNN,
    "SWINT": swint.SwinTransformer,
    "RESNET": resnet.ResNet,
    "VIT": vit.VIT,
    "FFCR": ffc_resnet.FFCR,
}


def build_model():
    return models[cfg.MODEL.BODY]().to(cfg.rank)
