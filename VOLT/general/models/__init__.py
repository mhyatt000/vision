from . import backbone, detector, head, layers, lm, rpn
from .vlrcnn import VLRCNN


def build_model(cfg):
    """returns a general vision language RCNN"""

    return VLRCNN(cfg)
