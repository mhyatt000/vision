from . import backbone, head, layers, lang, rpn
from .vlrcnn import VLRCNN

def build_model():
    return VLRCNN()
