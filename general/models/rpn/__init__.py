from .vldyhead import VLDyHeadModule
from general.config import cfg
# from .rpn import RPNModule
# from .retina import RetinaNetModule
# from .fcos import FCOSModule
# from .atss import ATSSModule
# from .dyhead import DyHeadModule

RPN_ARCH = {
        # "RPN": RPNModule,
        # "RETINA": RetinaNetModule,
        # "FCOS": FCOSModule,
        # "ATSS": ATSSModule,
        # "DYHEAD": DyHeadModule,
    "VLDYHEAD": VLDyHeadModule,
}


def build_rpn():
    """returns a region proposal network"""

    rpn_arch = RPN_ARCH[cfg.MODEL.RPN_ARCHITECTURE]
    return rpn_arch()
