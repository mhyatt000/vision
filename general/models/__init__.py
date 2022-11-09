from . import backbone, head, layers, lang, rpn
from .vlrcnn import VLRCNN

from .backbone import vit, swint
from general.config import cfg

models = {
    "VLRCNN": VLRCNN,
    "SWINT": swint.SwinTransformer,
    "RESNET": None,
    "VIT": vit.VIT,
}


def build_model():
    return models[cfg.MODEL.BODY]()
