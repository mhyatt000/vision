from .vswin import SwinTransformer3D as VSWIN

from . import vswin
from . import resnet


def build_backbone(cfg):
    """returns a vision backbone"""

    " for registering modules via registry.py "
    # assert (
        # cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES
    # ), f"cfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    # return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

    assert "VIDEO-SWIN" == cfg.MODEL.BACKBONE.BODY, "must be VIDEO-SWIN"
    return VSWIN(cfg)

