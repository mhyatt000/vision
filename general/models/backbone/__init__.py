from general.config import cfg

from . import fbnet, resnet, vswin
from .vswin import SwinTransformer3D as VSWIN


def build_backbone():
    """returns a vision backbone"""

    " for registering modules via registry.py "
    # assert (
        # cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES
    # ), f"cfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    # return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

    assert "VIDEO-SWIN" == cfg.MODEL.VISION.BODY, "must be VIDEO-SWIN"
    return VSWIN()

