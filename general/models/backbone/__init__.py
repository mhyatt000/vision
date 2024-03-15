from . import resnet, vswin, vit, ffcresnet, iresnet  # fbnet
from .vswin import SwinTransformer3D as VSWIN


def build_backbone():
    """returns a vision backbone"""

    " for registering modules via registry.py "
    # assert (
    # cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES
    # ), f"cfg.MODEL.BACKBONE.CONV_BODY: {cfg.MODEL.BACKBONE.CONV_BODY} are not registered in registry"
    # return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

    # assert "VIDEO-SWIN" == cfg.model.vision.body, "must be VIDEO-SWIN"
    return VSWIN()
